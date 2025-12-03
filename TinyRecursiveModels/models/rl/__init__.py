"""
Reinforcement Learning components for hypothesis-driven self-correction.

This module contains:
- heads.py: Q-head and V-head for action-value and state-value estimation
- gating.py: Hypothesis gating mechanism (LOCK/HYPOTHESIS/REJECT)
- rewards.py: Reward computation including entropy-based difficulty rewards
"""

from .heads import QHead, VHead, DuelingHead
from .gating import HypothesisGate, CellStatus
from .rewards import RewardComputer, EntropyComputer

__all__ = [
    'QHead',
    'VHead', 
    'DuelingHead',
    'HypothesisGate',
    'CellStatus',
    'RewardComputer',
    'EntropyComputer',
]

