"""
Proximal Policy Optimization (PPO)

Schulman et al., 2017: "Proximal Policy Optimization Algorithms"

Key idea: Clip the policy ratio to prevent too large updates.
L_CLIP = E[min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)]

where r(θ) = π_θ(a|s) / π_θ_old(a|s)

Pros:
- More stable than vanilla REINFORCE
- Natural trust region without complex computation
- Works well in practice

Cons:
- Requires multiple epochs per batch (sample efficiency vs compute)
- Still can drift from pretrained policy without KL penalty
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
from .rewards import compute_sudoku_reward


def ppo_step(
    policy_agent,
    puzzles: torch.Tensor,
    solutions_gt: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    act_coef: float = 0.1,
    ppo_epochs: int = 4,
    reward_fn: Optional[callable] = None,
    kl_coef: float = 0.0,
    reference_agent: Optional[torch.nn.Module] = None,
) -> Dict[str, Any]:
    """
    PPO training step with optional KL penalty and ACT head training.
    
    Args:
        policy_agent: Policy model
        puzzles: [B, 16] input puzzles
        solutions_gt: [B, 16] ground truth solutions
        optimizer: Optimizer for policy
        clip_eps: Clipping parameter ε (typically 0.1-0.3)
        entropy_coef: Entropy bonus coefficient
        act_coef: ACT head loss coefficient
        ppo_epochs: Number of optimization epochs per batch
        reward_fn: Custom reward function
        kl_coef: Optional KL penalty coefficient
        reference_agent: Optional frozen reference policy for KL
        
    Returns:
        Dict with metrics
    """
    B = puzzles.shape[0]
    reward_fn = reward_fn or compute_sudoku_reward
    
    policy_agent.train()
    
    # ===== Phase 1: Collect experience with OLD policy =====
    with torch.no_grad():
        old_logits, _ = policy_agent.backbone(puzzles)
        old_probs = F.softmax(old_logits, dim=-1)
        
        # Sample actions
        old_dist = torch.distributions.Categorical(old_probs)
        actions = old_dist.sample()  # [B, 16]
        old_log_probs = old_dist.log_prob(actions)  # [B, 16]
        
        # Compute rewards
        rewards = reward_fn(puzzles, actions)
        
        # Compute advantages
        baseline = rewards.mean()
        advantages = rewards - baseline
        # Normalize advantages for stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # ===== Phase 2: PPO optimization epochs =====
    total_policy_loss = 0
    total_clip_fraction = 0
    total_kl = 0
    total_act_loss = 0
    
    for epoch in range(ppo_epochs):
        # Forward pass with CURRENT policy including q_logits for ACT head
        logits, _, (q_halt, q_continue) = policy_agent.backbone(puzzles, return_q_logits=True)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Log probabilities of the actions taken
        log_probs = dist.log_prob(actions)
        
        # Policy ratio: π_new(a|s) / π_old(a|s)
        ratio = torch.exp(log_probs - old_log_probs)  # [B, 16]
        ratio_mean = ratio.mean(dim=1)  # [B]
        
        # Clipped surrogate objective
        surr1 = ratio_mean * advantages
        surr2 = torch.clamp(ratio_mean, 1 - clip_eps, 1 + clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Track clipping
        clip_fraction = ((ratio_mean < 1 - clip_eps) | (ratio_mean > 1 + clip_eps)).float().mean()
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        
        # ACT head loss: train q_halt to predict if solution is valid (R=1)
        # Same approach as pretraining: binary signal for "is solution perfect?"
        # q_halt is [B] scalar (from TRM Q-head using position 0)
        halt_target = (rewards > 0.99).float()  # R=1 means valid
        act_loss = F.binary_cross_entropy_with_logits(q_halt, halt_target)
        total_act_loss += act_loss.item()
        
        # Optional KL penalty
        kl_loss = torch.tensor(0.0, device=puzzles.device)
        if kl_coef > 0 and reference_agent is not None:
            with torch.no_grad():
                ref_logits, _ = reference_agent.backbone(puzzles)
                ref_probs = F.softmax(ref_logits, dim=-1)
            kl = (probs * (torch.log(probs + 1e-8) - torch.log(ref_probs + 1e-8))).sum(dim=-1).mean()
            kl_loss = kl_coef * kl
            total_kl += kl.item()
        
        # Total loss with ACT head
        total_loss = policy_loss + kl_loss - entropy_coef * entropy + act_coef * act_loss
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_agent.parameters(), 0.5)
        optimizer.step()
        
        total_policy_loss += policy_loss.item()
        total_clip_fraction += clip_fraction.item()
    
    # Compute accuracy
    with torch.no_grad():
        final_logits, _ = policy_agent.backbone(puzzles)
        greedy_actions = F.softmax(final_logits, dim=-1).argmax(dim=-1)
        empty_mask = (puzzles == 1)
        accuracy = ((greedy_actions == solutions_gt) & empty_mask).float().sum() / empty_mask.float().sum()
    
    return {
        'policy_loss': total_policy_loss / ppo_epochs,
        'clip_fraction': total_clip_fraction / ppo_epochs,
        'entropy': entropy.item(),
        'avg_reward': rewards.mean().item(),
        'accuracy': accuracy.item(),
        'act_loss': total_act_loss / ppo_epochs,
        'kl_divergence': total_kl / ppo_epochs if kl_coef > 0 else 0.0,
        'method': 'PPO' + ('+KL' if kl_coef > 0 else ''),
    }

