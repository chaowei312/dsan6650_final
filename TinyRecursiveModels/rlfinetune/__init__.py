"""
RL Fine-tuning Methods for Sudoku Policy

This module provides various RL algorithms for fine-tuning a pre-trained
supervised model, with different strategies to prevent catastrophic forgetting.

Methods:
- REINFORCE: Vanilla policy gradient (baseline)
- REINFORCE_KL: REINFORCE with KL divergence penalty
- PPO: Proximal Policy Optimization with clipped objective
- GRPO: Group Relative Policy Optimization (DeepSeek style)
"""

from .reinforce import reinforce_step
from .reinforce_kl import reinforce_kl_step, create_reference_policy
from .ppo import ppo_step
from .grpo import grpo_step
from .rewards import compute_sudoku_reward, compute_dense_reward

__all__ = [
    'reinforce_step',
    'reinforce_kl_step', 
    'create_reference_policy',
    'ppo_step',
    'grpo_step',
    'compute_sudoku_reward',
    'compute_dense_reward',
]

