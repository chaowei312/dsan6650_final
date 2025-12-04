"""
REINFORCE (Vanilla Policy Gradient)

The simplest policy gradient method. 
Williams, 1992: "Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning"

Pros:
- Simple to implement
- Unbiased gradient estimate

Cons:
- High variance
- No mechanism to prevent catastrophic forgetting
- Can diverge from pretrained policy quickly
"""

import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .rewards import compute_sudoku_reward


def reinforce_step(
    policy_agent,
    puzzles: torch.Tensor,
    solutions_gt: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    entropy_coef: float = 0.01,
    act_coef: float = 0.1,
    reward_fn: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Single REINFORCE training step with ACT head training.
    
    Loss = -E[log π(a|s) · (R - baseline)] - entropy_coef * H(π) + act_coef * ACT_loss
    
    ACT head is trained to predict when to halt:
    - High reward → q_halt should be high (halt, prediction is good)
    - Low reward → q_continue should be high (continue thinking)
    
    Args:
        policy_agent: Policy model with backbone
        puzzles: [B, 16] input puzzles
        solutions_gt: [B, 16] ground truth solutions (for accuracy metric)
        optimizer: Optimizer for policy
        entropy_coef: Coefficient for entropy bonus
        act_coef: Coefficient for ACT head loss
        reward_fn: Custom reward function (default: compute_sudoku_reward)
        
    Returns:
        Dict with metrics: policy_loss, entropy, avg_reward, accuracy, act_loss
    """
    B = puzzles.shape[0]
    reward_fn = reward_fn or compute_sudoku_reward
    
    policy_agent.train()
    
    # Forward pass with q_logits for ACT head training
    logits, _, (q_halt, q_continue) = policy_agent.backbone(puzzles, return_q_logits=True)
    probs = F.softmax(logits, dim=-1)
    
    # Sample actions from policy
    dist = torch.distributions.Categorical(probs)
    actions = dist.sample()  # [B, 16]
    log_probs = dist.log_prob(actions)  # [B, 16]
    
    # Compute rewards (no gradient needed)
    with torch.no_grad():
        rewards = reward_fn(puzzles, actions)  # [B]
    
    # REINFORCE with mean baseline (variance reduction)
    baseline = rewards.mean()
    advantages = rewards - baseline
    
    # Policy gradient loss: -E[log π(a|s) · A]
    policy_loss = -(log_probs.mean(dim=1) * advantages).mean()
    
    # Entropy bonus for exploration
    entropy = dist.entropy().mean()
    
    # ACT head loss: train q_halt to predict if solution is valid (R=1)
    # Same approach as pretraining: binary signal for "is solution perfect?"
    # q_halt is [B] scalar (from TRM Q-head using position 0)
    
    # Target: halt when reward == 1.0 (valid solution, no violations)
    halt_target = (rewards > 0.99).float()  # R=1 means valid
    
    # BCE loss for halt decision: sigmoid(q_halt) should match halt_target
    act_loss = F.binary_cross_entropy_with_logits(q_halt, halt_target)
    
    # Total loss
    total_loss = policy_loss - entropy_coef * entropy + act_coef * act_loss
    
    # Optimize
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_agent.parameters(), 1.0)
    optimizer.step()
    
    # Compute accuracy (greedy evaluation)
    with torch.no_grad():
        greedy_actions = probs.argmax(dim=-1)
        empty_mask = (puzzles == 1)
        accuracy = ((greedy_actions == solutions_gt) & empty_mask).float().sum() / empty_mask.float().sum()
    
    return {
        'policy_loss': policy_loss.item(),
        'entropy': entropy.item(),
        'avg_reward': rewards.mean().item(),
        'accuracy': accuracy.item(),
        'act_loss': act_loss.item(),
        'method': 'REINFORCE',
    }

