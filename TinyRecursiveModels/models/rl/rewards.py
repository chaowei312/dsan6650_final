"""
Reward computation for hypothesis-driven self-correction.

Implements:
1. Entropy-based difficulty measurement
2. Self-consistency rewards (Q calibration)
3. Commitment rewards (LOCK/REJECT transitions)
4. Terminal rewards (difficulty-scaled success)
5. Efficiency bonus (bits/cycle)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math

from .gating import CellStatus


class EntropyComputer(nn.Module):
    """
    Compute information-theoretic metrics for Sudoku puzzles.
    
    Measures puzzle difficulty based on constraint propagation entropy.
    """
    
    def __init__(self, grid_size: int = 4, vocab_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.vocab_size = vocab_size
        self.box_size = int(math.sqrt(grid_size))
        
    def get_candidates(
        self,
        grid: torch.Tensor,
        row: int,
        col: int,
    ) -> torch.Tensor:
        """
        Get possible values for a cell based on Sudoku constraints.
        
        Args:
            grid: Puzzle grid [batch_size, grid_size, grid_size]
            row, col: Cell position
            
        Returns:
            Candidate mask [batch_size, vocab_size] (True = valid candidate)
        """
        batch_size = grid.shape[0]
        device = grid.device
        
        # Start with all candidates
        candidates = torch.ones((batch_size, self.vocab_size), dtype=torch.bool, device=device)
        
        # Remove values in same row
        for c in range(self.grid_size):
            val = grid[:, row, c]
            valid = val > 0
            candidates[valid, val[valid].long() - 1] = False
            
        # Remove values in same column
        for r in range(self.grid_size):
            val = grid[:, r, col]
            valid = val > 0
            candidates[valid, val[valid].long() - 1] = False
            
        # Remove values in same box
        box_row = (row // self.box_size) * self.box_size
        box_col = (col // self.box_size) * self.box_size
        for r in range(box_row, box_row + self.box_size):
            for c in range(box_col, box_col + self.box_size):
                val = grid[:, r, c]
                valid = val > 0
                candidates[valid, val[valid].long() - 1] = False
                
        return candidates
    
    def compute_cell_entropy(
        self,
        grid: torch.Tensor,
        row: int,
        col: int,
    ) -> torch.Tensor:
        """
        Compute entropy for a single cell.
        
        H(cell) = log2(|candidates|) if not filled, 0 otherwise
        
        Args:
            grid: Puzzle grid [batch_size, grid_size, grid_size]
            row, col: Cell position
            
        Returns:
            Entropy [batch_size]
        """
        # If cell is filled, entropy is 0
        is_filled = grid[:, row, col] > 0
        
        # Get candidates
        candidates = self.get_candidates(grid, row, col)
        num_candidates = candidates.sum(dim=-1).float()
        
        # Entropy = log2(candidates), clamp to avoid log(0)
        entropy = torch.log2(num_candidates.clamp(min=1))
        entropy = torch.where(is_filled, torch.zeros_like(entropy), entropy)
        
        return entropy
    
    def compute_total_entropy(self, grid: torch.Tensor) -> torch.Tensor:
        """
        Compute total puzzle entropy.
        
        H_total = sum of cell entropies
        
        Args:
            grid: Puzzle grid [batch_size, grid_size, grid_size]
            
        Returns:
            Total entropy [batch_size]
        """
        total_entropy = torch.zeros(grid.shape[0], device=grid.device)
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                total_entropy += self.compute_cell_entropy(grid, row, col)
                
        return total_entropy
    
    def compute_information_gain(
        self,
        grid_before: torch.Tensor,
        grid_after: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute information gain from one state to another.
        
        IG = H(before) - H(after)
        
        Args:
            grid_before: Grid before action [B, G, G]
            grid_after: Grid after action [B, G, G]
            
        Returns:
            Information gain [batch_size]
        """
        h_before = self.compute_total_entropy(grid_before)
        h_after = self.compute_total_entropy(grid_after)
        
        return h_before - h_after


class RewardComputer(nn.Module):
    """
    Compute all reward components for hypothesis-driven RL.
    
    Reward Structure:
    1. r_commit: Self-consistency reward for commitments
    2. r_ce: Cross-entropy improvement reward
    3. r_terminal: Difficulty-scaled terminal reward
    4. r_efficiency: Bits per cycle bonus
    """
    
    def __init__(
        self,
        grid_size: int = 4,
        vocab_size: int = 4,
        # Reward coefficients
        commit_correct_confident: float = 1.5,
        commit_correct_uncertain: float = 0.7,
        commit_wrong_confident: float = -2.0,
        commit_wrong_uncertain: float = -0.5,
        reject_correct: float = -0.3,
        reject_wrong: float = 0.3,
        terminal_success_base: float = 5.0,
        terminal_failure: float = -3.0,
        efficiency_bonus: float = 0.5,
        step_penalty: float = -0.01,
        # Self-consistency coefficient
        consistency_weight: float = 0.5,
    ):
        super().__init__()
        
        self.entropy_computer = EntropyComputer(grid_size, vocab_size)
        self.grid_size = grid_size
        self.vocab_size = vocab_size
        
        # Store coefficients
        self.r_commit = {
            'correct_confident': commit_correct_confident,
            'correct_uncertain': commit_correct_uncertain,
            'wrong_confident': commit_wrong_confident,
            'wrong_uncertain': commit_wrong_uncertain,
        }
        # Rejection rewards: Only penalize rejecting correct, neutral for rejecting wrong
        # This prevents gaming by rejecting everything
        self.r_reject = {
            'correct': reject_correct,  # -0.3: Bad to reject correct
            'wrong': 0.0,               # Neutral: Don't reward rejection
        }
        self.r_terminal = {
            'success_base': terminal_success_base,
            'failure': terminal_failure,
        }
        self.efficiency_bonus = efficiency_bonus
        self.step_penalty = step_penalty
        self.consistency_weight = consistency_weight
        
    def compute_commit_reward(
        self,
        q_values: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        locked_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reward for commitment (LOCK) decisions.
        
        r_commit = r_base(correct/wrong) + consistency_bonus
        
        Self-consistency bonus:
        - High Q + correct = +0.5 (confident and right)
        - High Q + wrong = -0.5 (overconfident)
        - Low Q + correct = -0.2 (underconfident)
        - Low Q + wrong = +0.2 (correctly uncertain)
        
        Args:
            q_values: Q values [B, L]
            predictions: Model predictions [B, L]
            labels: Ground truth [B, L]
            locked_mask: Which cells were just locked [B, L]
            
        Returns:
            rewards: Per-cell rewards [B, L]
            info: Statistics dict
        """
        device = q_values.device
        batch_size, seq_len = q_values.shape
        
        # Check correctness
        is_correct = (predictions == labels) & locked_mask
        is_wrong = (predictions != labels) & locked_mask
        
        # High/low confidence (Q > 0 = confident)
        is_confident = q_values > 0
        
        # Base rewards
        rewards = torch.zeros_like(q_values)
        
        # Correct + confident
        rewards = torch.where(
            is_correct & is_confident,
            torch.full_like(rewards, self.r_commit['correct_confident']),
            rewards
        )
        
        # Correct + uncertain
        rewards = torch.where(
            is_correct & ~is_confident,
            torch.full_like(rewards, self.r_commit['correct_uncertain']),
            rewards
        )
        
        # Wrong + confident (worst case)
        rewards = torch.where(
            is_wrong & is_confident,
            torch.full_like(rewards, self.r_commit['wrong_confident']),
            rewards
        )
        
        # Wrong + uncertain
        rewards = torch.where(
            is_wrong & ~is_confident,
            torch.full_like(rewards, self.r_commit['wrong_uncertain']),
            rewards
        )
        
        # Self-consistency bonus
        # Reward alignment between Q and actual correctness
        correct_float = is_correct.float() * 2 - 1  # +1 if correct, -1 if wrong
        consistency = q_values * correct_float  # Positive if aligned
        consistency_bonus = self.consistency_weight * consistency * locked_mask.float()
        
        rewards = rewards + consistency_bonus
        
        info = {
            'num_correct_confident': (is_correct & is_confident).sum(),
            'num_correct_uncertain': (is_correct & ~is_confident).sum(),
            'num_wrong_confident': (is_wrong & is_confident).sum(),
            'num_wrong_uncertain': (is_wrong & ~is_confident).sum(),
            'consistency_bonus_mean': consistency_bonus[locked_mask].mean() if locked_mask.any() else torch.tensor(0.0),
        }
        
        return rewards, info
    
    def compute_reject_reward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        rejected_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reward for rejection decisions.
        
        Rejecting wrong predictions is good, rejecting correct is bad.
        
        Args:
            predictions: Predictions before rejection [B, L]
            labels: Ground truth [B, L]
            rejected_mask: Which cells were rejected [B, L]
            
        Returns:
            rewards: Per-cell rewards [B, L]
            info: Statistics dict
        """
        was_correct = (predictions == labels) & rejected_mask
        was_wrong = (predictions != labels) & rejected_mask
        
        rewards = torch.zeros_like(predictions, dtype=torch.float)
        
        # Rejecting correct prediction (bad)
        rewards = torch.where(
            was_correct,
            torch.full_like(rewards, self.r_reject['correct']),
            rewards
        )
        
        # Rejecting wrong prediction (good)
        rewards = torch.where(
            was_wrong,
            torch.full_like(rewards, self.r_reject['wrong']),
            rewards
        )
        
        info = {
            'num_reject_correct': was_correct.sum(),
            'num_reject_wrong': was_wrong.sum(),
        }
        
        return rewards, info
    
    def compute_terminal_reward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        status: torch.Tensor,
        initial_entropy: torch.Tensor,
        num_cycles: torch.Tensor,
        max_cycles: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute terminal reward with difficulty scaling.
        
        r_terminal = r_base × (1 + H_0 / H_max) × efficiency_multiplier
        
        Args:
            predictions: Final predictions [B, L]
            labels: Ground truth [B, L]
            status: Final cell status [B, L]
            initial_entropy: Initial puzzle entropy [B]
            num_cycles: Number of cycles used [B]
            max_cycles: Maximum allowed cycles
            
        Returns:
            rewards: Per-batch terminal rewards [B]
            info: Statistics dict
        """
        batch_size = predictions.shape[0]
        device = predictions.device
        
        # Check if puzzle is solved
        non_given_mask = status != CellStatus.GIVEN
        all_correct = ((predictions == labels) | ~non_given_mask).all(dim=-1)
        
        # Base reward
        base_reward = torch.where(
            all_correct,
            torch.full((batch_size,), self.r_terminal['success_base'], device=device),
            torch.full((batch_size,), self.r_terminal['failure'], device=device),
        )
        
        # Difficulty scaling: harder puzzles get higher rewards
        # Normalize entropy to [0, 1] range (H_max ≈ grid_size^2 * log2(vocab_size))
        h_max = (self.grid_size ** 2) * math.log2(self.vocab_size)
        difficulty_multiplier = 1 + initial_entropy / h_max
        
        # Efficiency multiplier: faster solutions get bonus
        # efficiency = (max_cycles - used_cycles + 1) / max_cycles
        efficiency_multiplier = (max_cycles - num_cycles.float() + 1) / max_cycles
        
        # Combine (only apply multipliers to success)
        terminal_reward = torch.where(
            all_correct,
            base_reward * difficulty_multiplier * (1 + self.efficiency_bonus * efficiency_multiplier),
            base_reward,  # Failure reward not scaled
        )
        
        info = {
            'num_solved': all_correct.sum(),
            'difficulty_multiplier_mean': difficulty_multiplier.mean(),
            'efficiency_multiplier_mean': efficiency_multiplier.mean(),
            'terminal_reward_mean': terminal_reward.mean(),
        }
        
        return terminal_reward, info
    
    def compute_step_reward(
        self,
        status: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute per-step penalty to encourage efficiency.
        
        Small negative reward per cycle for each unfilled/hypothesis cell.
        
        Args:
            status: Current cell status [B, L]
            
        Returns:
            Step penalty [B]
        """
        # Count cells that aren't done yet
        not_done = (status == CellStatus.UNFILLED) | (status == CellStatus.HYPOTHESIS)
        not_done_count = not_done.sum(dim=-1).float()
        
        # Small penalty per undone cell
        return self.step_penalty * not_done_count
    
    def compute_ce_improvement_reward(
        self,
        logits_before: torch.Tensor,
        logits_after: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute reward for cross-entropy improvement.
        
        r_ce = CE(before) - CE(after) (positive if improved)
        
        Args:
            logits_before: Logits before H cycle [B, L, V]
            logits_after: Logits after H cycle [B, L, V]
            labels: Ground truth [B, L]
            mask: Valid cells mask [B, L]
            
        Returns:
            Per-cell improvement rewards [B, L]
            info: Statistics dict
        """
        # Compute CE loss before and after
        ce_before = F.cross_entropy(
            logits_before.view(-1, logits_before.shape[-1]),
            labels.view(-1),
            reduction='none'
        ).view(labels.shape)
        
        ce_after = F.cross_entropy(
            logits_after.view(-1, logits_after.shape[-1]),
            labels.view(-1),
            reduction='none'
        ).view(labels.shape)
        
        # Improvement (positive = got better)
        improvement = (ce_before - ce_after) * mask.float()
        
        info = {
            'ce_before_mean': ce_before[mask].mean() if mask.any() else torch.tensor(0.0),
            'ce_after_mean': ce_after[mask].mean() if mask.any() else torch.tensor(0.0),
            'ce_improvement_mean': improvement[mask].mean() if mask.any() else torch.tensor(0.0),
        }
        
        return improvement, info
    
    def compute_trajectory_rewards(
        self,
        trajectory: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        initial_entropy: torch.Tensor,
        max_cycles: int,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute all rewards for a full trajectory.
        
        Args:
            trajectory: Dict with status_history, prediction_history, q_history, logits_history
            labels: Ground truth [B, L]
            initial_entropy: Initial puzzle entropy [B]
            max_cycles: Maximum H cycles
            
        Returns:
            Total rewards per timestep [B, T]
            info: Aggregated statistics
        """
        status_history = trajectory['status_history']  # [B, T+1, L]
        prediction_history = trajectory['prediction_history']  # [B, T+1, L]
        q_history = trajectory['q_history']  # [B, T, L]
        
        batch_size = labels.shape[0]
        num_steps = q_history.shape[1]
        device = labels.device
        
        all_rewards = []
        all_info = {}
        
        for t in range(num_steps):
            step_rewards = torch.zeros((batch_size, labels.shape[1]), device=device)
            
            # Current and next status
            status_t = status_history[:, t]
            status_tp1 = status_history[:, t + 1]
            pred_t = prediction_history[:, t]
            pred_tp1 = prediction_history[:, t + 1]
            q_t = q_history[:, t]
            
            # Detect transitions
            just_locked = (status_tp1 == CellStatus.LOCKED) & (status_t != CellStatus.LOCKED)
            just_rejected = (status_t == CellStatus.HYPOTHESIS) & (status_tp1 == CellStatus.UNFILLED)
            
            # Commitment rewards
            commit_rewards, commit_info = self.compute_commit_reward(
                q_values=q_t,
                predictions=pred_tp1,
                labels=labels,
                locked_mask=just_locked,
            )
            step_rewards = step_rewards + commit_rewards
            
            # Rejection rewards
            reject_rewards, reject_info = self.compute_reject_reward(
                predictions=pred_t,
                labels=labels,
                rejected_mask=just_rejected,
            )
            step_rewards = step_rewards + reject_rewards
            
            # Step penalty
            step_penalty = self.compute_step_reward(status_tp1)
            step_rewards = step_rewards + step_penalty.unsqueeze(-1)
            
            all_rewards.append(step_rewards)
            
            # Aggregate info
            for k, v in {**commit_info, **reject_info}.items():
                if k not in all_info:
                    all_info[k] = []
                all_info[k].append(v)
        
        # Stack rewards [B, T, L]
        all_rewards = torch.stack(all_rewards, dim=1)
        
        # Terminal reward (last step only)
        final_status = status_history[:, -1]
        final_predictions = prediction_history[:, -1]
        num_cycles = torch.tensor(num_steps, device=device).expand(batch_size)
        
        terminal_rewards, terminal_info = self.compute_terminal_reward(
            predictions=final_predictions,
            labels=labels,
            status=final_status,
            initial_entropy=initial_entropy,
            num_cycles=num_cycles,
            max_cycles=max_cycles,
        )
        
        # Add terminal reward to last timestep (broadcast over cells)
        all_rewards[:, -1] = all_rewards[:, -1] + terminal_rewards.unsqueeze(-1) / labels.shape[1]
        
        # Aggregate info
        for k, v in all_info.items():
            if isinstance(v[0], torch.Tensor):
                # Ensure all tensors on same device
                all_info[k] = torch.stack([t.to(device) for t in v]).sum()
            else:
                all_info[k] = sum(v)
        all_info.update(terminal_info)
        
        return all_rewards, all_info


class GAEComputer:
    """
    Generalized Advantage Estimation for PPO.
    
    A_t = sum_{l=0}^{inf} (gamma * lambda)^l * delta_{t+l}
    where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    """
    
    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE advantages and returns.
        
        Args:
            rewards: Rewards per timestep [B, T] or [B, T, L]
            values: Value estimates [B, T+1] or [B, T+1, L]
            dones: Episode done flags [B, T]
            
        Returns:
            advantages: GAE advantages [B, T] or [B, T, L]
            returns: Discounted returns [B, T] or [B, T, L]
        """
        T = rewards.shape[1]
        
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = values[:, t + 1]
                next_done = dones[:, t] if dones.dim() > 1 else dones
            else:
                next_value = values[:, t + 1]
                next_done = dones[:, t] if dones.dim() > 1 else torch.zeros_like(dones[:, 0])
            
            # Handle done flag dimension
            if next_done.dim() < rewards.dim() - 1:
                next_done = next_done.unsqueeze(-1)
                
            delta = rewards[:, t] + self.gamma * next_value * (1 - next_done.float()) - values[:, t]
            advantages[:, t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - next_done.float()) * last_gae
            
        returns = advantages + values[:, :-1]
        
        return advantages, returns
    
    def normalize_advantages(self, advantages: torch.Tensor) -> torch.Tensor:
        """Normalize advantages to have mean 0 and std 1."""
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

