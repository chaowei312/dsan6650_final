"""
REINFORCE with KL Divergence Penalty

Adds KL(π_current || π_reference) penalty to prevent catastrophic forgetting.
This is the key technique used in RLHF (InstructGPT, Claude, etc.)

Loss = -E[log π(a|s) · (R - baseline)] + β * KL(π || π_ref) - entropy_coef * H(π)

The KL term keeps the current policy close to the reference (pretrained) policy.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
import copy
from .rewards import compute_sudoku_reward


def create_reference_policy(policy_agent) -> torch.nn.Module:
    """
    Create a frozen copy of the policy to use as reference.
    
    Args:
        policy_agent: The current policy model
        
    Returns:
        Frozen copy of the policy
    """
    reference = copy.deepcopy(policy_agent)
    for param in reference.parameters():
        param.requires_grad = False
    reference.eval()
    return reference


def compute_kl_divergence(
    current_probs: torch.Tensor,
    reference_probs: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute KL(current || reference) for each position.
    
    Args:
        current_probs: [B, T, V] current policy probabilities
        reference_probs: [B, T, V] reference policy probabilities
        
    Returns:
        kl: [B] mean KL divergence per sample
    """
    # KL(P || Q) = sum(P * log(P/Q))
    log_ratio = torch.log(current_probs + eps) - torch.log(reference_probs + eps)
    kl = (current_probs * log_ratio).sum(dim=-1)  # [B, T]
    return kl.mean(dim=1)  # [B]


def reinforce_kl_step(
    policy_agent,
    reference_agent,
    puzzles: torch.Tensor,
    solutions_gt: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    kl_coef: float = 0.1,
    entropy_coef: float = 0.01,
    act_coef: float = 0.1,
    reward_fn: Optional[callable] = None,
    adaptive_kl: bool = False,
    target_kl: float = 0.01,
) -> Dict[str, Any]:
    """
    REINFORCE with KL penalty and ACT head training.
    
    Args:
        policy_agent: Current policy model
        reference_agent: Frozen reference policy (from create_reference_policy)
        puzzles: [B, 16] input puzzles
        solutions_gt: [B, 16] ground truth solutions
        optimizer: Optimizer for policy
        kl_coef: Coefficient for KL penalty (β)
        entropy_coef: Coefficient for entropy bonus
        act_coef: Coefficient for ACT head loss
        reward_fn: Custom reward function
        adaptive_kl: If True, adjust kl_coef based on actual KL
        target_kl: Target KL for adaptive adjustment
        
    Returns:
        Dict with metrics
    """
    B = puzzles.shape[0]
    reward_fn = reward_fn or compute_sudoku_reward
    
    policy_agent.train()
    
    # Forward pass - current policy with q_logits for ACT head
    logits, _, (q_halt, q_continue) = policy_agent.backbone(puzzles, return_q_logits=True)
    probs = F.softmax(logits, dim=-1)
    
    # Forward pass - reference policy (no grad)
    with torch.no_grad():
        ref_logits, _ = reference_agent.backbone(puzzles)
        ref_probs = F.softmax(ref_logits, dim=-1)
    
    # Sample actions from current policy
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample()
    log_probs = dist.log_prob(actions)
    
    # Compute rewards
    with torch.no_grad():
        rewards = reward_fn(puzzles, actions)
    
    # REINFORCE loss
    baseline = rewards.mean()
    advantages = rewards - baseline
    policy_loss = -(log_probs.mean(dim=1) * advantages).mean()
    
    # KL divergence penalty
    kl_div = compute_kl_divergence(probs, ref_probs).mean()
    
    # Entropy bonus
    entropy = dist.entropy().mean()
    
    # ACT head loss: train q_halt to be high when reward is high
    # q_halt and q_continue are already [B] scalars (from TRM Q-head using position 0)
    halt_target = (rewards > 0.5).float()
    halt_logits = q_halt - q_continue  # [B]
    act_loss = F.binary_cross_entropy_with_logits(halt_logits, halt_target)
    
    # Total loss with KL penalty and ACT loss
    total_loss = policy_loss + kl_coef * kl_div - entropy_coef * entropy + act_coef * act_loss
    
    # Optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_agent.parameters(), 1.0)
    optimizer.step()
    
    # Adaptive KL coefficient adjustment
    new_kl_coef = kl_coef
    if adaptive_kl:
        if kl_div.item() > target_kl * 1.5:
            new_kl_coef = min(kl_coef * 1.5, 1.0)  # Increase penalty
        elif kl_div.item() < target_kl * 0.5:
            new_kl_coef = max(kl_coef / 1.5, 0.001)  # Decrease penalty
    
    # Compute accuracy
    with torch.no_grad():
        greedy_actions = probs.argmax(dim=-1)
        empty_mask = (puzzles == 1)
        accuracy = ((greedy_actions == solutions_gt) & empty_mask).float().sum() / empty_mask.float().sum()
    
    return {
        'policy_loss': policy_loss.item(),
        'kl_divergence': kl_div.item(),
        'entropy': entropy.item(),
        'avg_reward': rewards.mean().item(),
        'accuracy': accuracy.item(),
        'act_loss': act_loss.item(),
        'kl_coef': new_kl_coef,
        'method': 'REINFORCE+KL',
    }

