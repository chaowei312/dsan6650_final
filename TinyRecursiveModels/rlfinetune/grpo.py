"""
Group Relative Policy Optimization (GRPO)

DeepSeek-Math, 2024: Uses group-based relative ranking instead of absolute rewards.

Key idea: For each prompt, sample K solutions and rank them.
- Best solution gets positive advantage
- Worst solution gets negative advantage
- Uses relative ordering, not absolute reward values

Advantages:
- No need for reward model training (uses relative comparisons)
- Reduces reward hacking
- More stable than absolute rewards

Formula:
L_GRPO = -E[log π(a|s) · (rank_normalized_advantage)]
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
from .rewards import compute_sudoku_reward


def grpo_step(
    policy_agent,
    puzzles: torch.Tensor,
    solutions_gt: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    num_samples: int = 4,
    entropy_coef: float = 0.01,
    act_coef: float = 0.1,
    reward_fn: Optional[callable] = None,
    kl_coef: float = 0.0,
    reference_agent: Optional[torch.nn.Module] = None,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """
    GRPO training step with ACT head training.
    
    Args:
        policy_agent: Policy model
        puzzles: [B, 16] input puzzles
        solutions_gt: [B, 16] ground truth solutions
        optimizer: Optimizer
        num_samples: K - number of solutions to sample per puzzle
        entropy_coef: Entropy bonus coefficient
        act_coef: ACT head loss coefficient
        reward_fn: Reward function
        kl_coef: Optional KL penalty coefficient
        reference_agent: Optional frozen reference for KL
        temperature: Sampling temperature (higher = more exploration)
        
    Returns:
        Dict with metrics
    """
    B = puzzles.shape[0]
    K = num_samples
    reward_fn = reward_fn or compute_sudoku_reward
    device = puzzles.device
    
    policy_agent.train()
    
    # Forward pass with q_logits for ACT head training
    logits, _, (q_halt, q_continue) = policy_agent.backbone(puzzles, return_q_logits=True)
    probs = F.softmax(logits / temperature, dim=-1)  # [B, 16, V]
    
    # ===== Sample K solutions per puzzle =====
    all_actions = []
    all_log_probs = []
    all_rewards = []
    
    dist = torch.distributions.Categorical(probs)
    
    for k in range(K):
        actions = dist.sample()  # [B, 16]
        log_probs = dist.log_prob(actions)  # [B, 16]
        
        with torch.no_grad():
            rewards = reward_fn(puzzles, actions)  # [B]
        
        all_actions.append(actions)
        all_log_probs.append(log_probs.mean(dim=1))  # [B] - mean log prob per sample
        all_rewards.append(rewards)
    
    # Stack: [K, B]
    all_log_probs = torch.stack(all_log_probs, dim=0)  # [K, B]
    all_rewards = torch.stack(all_rewards, dim=0)  # [K, B]
    
    # ===== Compute group-relative advantages =====
    # For each puzzle (column), rank the K samples
    # Best gets +1, worst gets -1, middle gets intermediate values
    
    # Transpose to [B, K] for easier per-puzzle operations
    rewards_t = all_rewards.t()  # [B, K]
    log_probs_t = all_log_probs.t()  # [B, K]
    
    # Compute ranks within each group (0 = worst, K-1 = best)
    # argsort of argsort gives ranks
    ranks = rewards_t.argsort(dim=1).argsort(dim=1).float()  # [B, K]
    
    # Normalize ranks to [-1, +1]
    # rank 0 → -1, rank K-1 → +1
    normalized_ranks = 2 * ranks / (K - 1) - 1  # [B, K]
    
    # Use normalized ranks as advantages
    advantages = normalized_ranks
    
    # ===== Policy gradient with group-relative advantages =====
    # Flatten for loss computation
    policy_loss = -(log_probs_t * advantages).mean()
    
    # Entropy bonus (from original distribution)
    entropy = dist.entropy().mean()
    
    # ACT head loss: train q_halt to predict if solution is valid (R=1)
    # Same approach as pretraining: binary signal for "is solution perfect?"
    # q_halt is [B] scalar (from TRM Q-head using position 0)
    mean_reward = all_rewards.mean(dim=0)  # [B]
    halt_target = (mean_reward > 0.99).float()  # R=1 means valid
    act_loss = F.binary_cross_entropy_with_logits(q_halt, halt_target)
    
    # Optional KL penalty
    kl_loss = torch.tensor(0.0, device=device)
    if kl_coef > 0 and reference_agent is not None:
        with torch.no_grad():
            ref_logits, _ = reference_agent.backbone(puzzles)
            ref_probs = F.softmax(ref_logits, dim=-1)
        kl = (probs * (torch.log(probs + 1e-8) - torch.log(ref_probs + 1e-8))).sum(dim=-1).mean()
        kl_loss = kl_coef * kl
    
    # Total loss with ACT head
    total_loss = policy_loss + kl_loss - entropy_coef * entropy + act_coef * act_loss
    
    # Optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_agent.parameters(), 1.0)
    optimizer.step()
    
    # Compute metrics
    with torch.no_grad():
        # Best and worst rewards in each group
        best_rewards = all_rewards.max(dim=0)[0]  # [B]
        worst_rewards = all_rewards.min(dim=0)[0]  # [B]
        mean_rewards = all_rewards.mean(dim=0)  # [B]
        
        # Accuracy using greedy
        greedy_actions = probs.argmax(dim=-1)
        empty_mask = (puzzles == 1)
        accuracy = ((greedy_actions == solutions_gt) & empty_mask).float().sum() / empty_mask.float().sum()
    
    return {
        'policy_loss': policy_loss.item(),
        'entropy': entropy.item(),
        'avg_reward': mean_rewards.mean().item(),
        'best_reward': best_rewards.mean().item(),
        'worst_reward': worst_rewards.mean().item(),
        'reward_spread': (best_rewards - worst_rewards).mean().item(),
        'accuracy': accuracy.item(),
        'act_loss': act_loss.item(),
        'kl_divergence': kl_loss.item() / kl_coef if kl_coef > 0 else 0.0,
        'method': 'GRPO' + ('+KL' if kl_coef > 0 else ''),
    }


def grpo_step_with_consensus(
    policy_agent,
    puzzles: torch.Tensor,
    solutions_gt: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    num_samples: int = 8,
    consensus_threshold: float = 0.6,
    entropy_coef: float = 0.01,
    reward_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    GRPO with consensus bonus - reward solutions that agree with the group.
    
    Args:
        policy_agent: Policy model
        puzzles: [B, 16] input puzzles
        solutions_gt: [B, 16] ground truth
        optimizer: Optimizer
        num_samples: K samples per puzzle
        consensus_threshold: Fraction of samples that must agree for bonus
        entropy_coef: Entropy bonus
        reward_fn: Base reward function
        
    Returns:
        Dict with metrics
    """
    B = puzzles.shape[0]
    K = num_samples
    reward_fn = reward_fn or compute_sudoku_reward
    device = puzzles.device
    
    policy_agent.train()
    
    # Forward pass
    logits, _ = policy_agent.backbone(puzzles)
    probs = F.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    
    # Sample K solutions per puzzle
    all_actions = []
    all_log_probs = []
    all_base_rewards = []
    
    for k in range(K):
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        with torch.no_grad():
            rewards = reward_fn(puzzles, actions)
        
        all_actions.append(actions)
        all_log_probs.append(log_probs.mean(dim=1))
        all_base_rewards.append(rewards)
    
    # Stack actions: [K, B, 16]
    all_actions = torch.stack(all_actions, dim=0)
    all_log_probs = torch.stack(all_log_probs, dim=0)  # [K, B]
    all_base_rewards = torch.stack(all_base_rewards, dim=0)  # [K, B]
    
    # ===== Compute consensus for each cell =====
    # Mode (most common value) for each position
    with torch.no_grad():
        # For each position, find the most common prediction
        consensus_rewards = torch.zeros(K, B, device=device)
        
        for b in range(B):
            for pos in range(16):
                if puzzles[b, pos] == 1:  # Only for empty cells
                    # Get all predictions for this position
                    preds = all_actions[:, b, pos]  # [K]
                    
                    # Count occurrences of each value
                    for k in range(K):
                        agreement = (preds == preds[k]).float().mean()
                        if agreement >= consensus_threshold:
                            consensus_rewards[k, b] += 0.1  # Bonus for consensus
    
    # Combine base rewards with consensus bonus
    total_rewards = all_base_rewards + consensus_rewards
    
    # Rank-based advantages
    rewards_t = total_rewards.t()  # [B, K]
    log_probs_t = all_log_probs.t()  # [B, K]
    
    ranks = rewards_t.argsort(dim=1).argsort(dim=1).float()
    advantages = 2 * ranks / (K - 1) - 1
    
    # Policy gradient
    policy_loss = -(log_probs_t * advantages).mean()
    entropy = dist.entropy().mean()
    total_loss = policy_loss - entropy_coef * entropy
    
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_agent.parameters(), 1.0)
    optimizer.step()
    
    # Metrics
    with torch.no_grad():
        greedy_actions = probs.argmax(dim=-1)
        empty_mask = (puzzles == 1)
        accuracy = ((greedy_actions == solutions_gt) & empty_mask).float().sum() / empty_mask.float().sum()
    
    return {
        'policy_loss': policy_loss.item(),
        'entropy': entropy.item(),
        'avg_reward': all_base_rewards.mean().item(),
        'consensus_bonus': consensus_rewards.mean().item(),
        'accuracy': accuracy.item(),
        'method': 'GRPO+Consensus',
    }

