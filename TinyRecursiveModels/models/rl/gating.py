"""
Hypothesis gating mechanism for discrete decision making.

Implements the LOCK/HYPOTHESIS/REJECT state machine:
- UNFILLED: Cell is empty (given puzzle constraint)
- LOCKED: Model is confident, prediction committed
- HYPOTHESIS: Model is uncertain, tentative prediction
- REJECTED: Model believes prediction is wrong, reset to UNFILLED

Uses straight-through estimator for gradient flow through discrete decisions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, NamedTuple
from enum import IntEnum


class CellStatus(IntEnum):
    """Status of each cell in the puzzle."""
    GIVEN = 0      # Original puzzle constraint (never changes)
    UNFILLED = 1   # Empty cell, needs prediction
    HYPOTHESIS = 2 # Tentative prediction, might change
    LOCKED = 3     # Confident prediction, committed
    REJECTED = 4   # Was wrong, reset to unfilled


class GatingThresholds(NamedTuple):
    """Thresholds for discrete gating decisions."""
    lock_threshold: float = 0.6      # Q > this -> LOCK
    reject_threshold: float = -0.4   # Q < this -> REJECT
    # Between reject and lock -> HYPOTHESIS


class HypothesisGate(nn.Module):
    """
    Discrete hypothesis gating mechanism.
    
    Takes Q values and current status, outputs new status and gating decisions.
    Uses straight-through estimator for backpropagation.
    
    State Transitions:
    - UNFILLED + predict -> HYPOTHESIS (always start uncertain)
    - HYPOTHESIS + high Q -> LOCKED (commit)
    - HYPOTHESIS + low Q -> REJECTED -> UNFILLED (reset)
    - LOCKED + very low Q -> UNLOCKED -> UNFILLED (self-correction)
    """
    
    def __init__(
        self,
        lock_threshold: float = 0.6,
        reject_threshold: float = -0.4,
        unlock_threshold: float = -0.7,  # Must be very wrong to unlock
        allow_unlock: bool = True,
        temperature: float = 1.0,  # For soft decisions during training
    ):
        super().__init__()
        
        self.thresholds = GatingThresholds(
            lock_threshold=lock_threshold,
            reject_threshold=reject_threshold,
        )
        self.unlock_threshold = unlock_threshold
        self.allow_unlock = allow_unlock
        self.temperature = temperature
        
    def forward(
        self,
        q_values: torch.Tensor,
        status: torch.Tensor,
        predictions: torch.Tensor,
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply gating decisions based on Q values.
        
        Args:
            q_values: Q values from Q-head [batch_size, seq_len]
            status: Current cell status [batch_size, seq_len]
            predictions: Current predictions [batch_size, seq_len]
            training: Whether in training mode (uses soft gates)
            
        Returns:
            new_status: Updated cell status [batch_size, seq_len]
            new_predictions: Updated predictions (reset if rejected)
            info: Dict with gating statistics
        """
        batch_size, seq_len = q_values.shape
        device = q_values.device
        
        # Initialize outputs
        new_status = status.clone()
        new_predictions = predictions.clone()
        
        # Masks for different current states
        is_given = (status == CellStatus.GIVEN)
        is_unfilled = (status == CellStatus.UNFILLED)
        is_hypothesis = (status == CellStatus.HYPOTHESIS)
        is_locked = (status == CellStatus.LOCKED)
        
        # Gating decisions based on Q values
        should_lock = q_values > self.thresholds.lock_threshold
        should_reject = q_values < self.thresholds.reject_threshold
        should_hypothesis = ~should_lock & ~should_reject
        should_unlock = q_values < self.unlock_threshold
        
        if training:
            # Soft gating with straight-through estimator
            lock_probs, reject_probs, hyp_probs = self._soft_gate(q_values)
            
            # Apply soft gates (for gradient flow)
            # Actual decisions are still discrete
            gate_info = {
                'lock_prob': lock_probs,
                'reject_prob': reject_probs,
                'hypothesis_prob': hyp_probs,
            }
        else:
            gate_info = {}
        
        # State transitions (discrete)
        # UNFILLED -> HYPOTHESIS (when model makes a prediction)
        # This happens externally when model outputs a prediction
        
        # HYPOTHESIS -> LOCKED (when confident)
        new_status = torch.where(
            is_hypothesis & should_lock,
            torch.full_like(status, CellStatus.LOCKED),
            new_status
        )
        
        # HYPOTHESIS -> REJECTED -> UNFILLED (when wrong)
        rejected_mask = is_hypothesis & should_reject
        new_status = torch.where(
            rejected_mask,
            torch.full_like(status, CellStatus.UNFILLED),
            new_status
        )
        # Reset predictions for rejected cells
        new_predictions = torch.where(
            rejected_mask,
            torch.zeros_like(predictions),  # 0 = empty
            new_predictions
        )
        
        # LOCKED -> UNFILLED (self-correction, optional)
        if self.allow_unlock:
            unlock_mask = is_locked & should_unlock
            new_status = torch.where(
                unlock_mask,
                torch.full_like(status, CellStatus.UNFILLED),
                new_status
            )
            new_predictions = torch.where(
                unlock_mask,
                torch.zeros_like(predictions),
                new_predictions
            )
        else:
            unlock_mask = torch.zeros_like(is_locked)
        
        # Never modify GIVEN cells
        new_status = torch.where(is_given, status, new_status)
        new_predictions = torch.where(is_given, predictions, new_predictions)
        
        # Compute statistics
        info = {
            'num_locked': (new_status == CellStatus.LOCKED).sum(),
            'num_hypothesis': (new_status == CellStatus.HYPOTHESIS).sum(),
            'num_unfilled': (new_status == CellStatus.UNFILLED).sum(),
            'num_rejected': rejected_mask.sum(),
            'num_unlocked': unlock_mask.sum() if self.allow_unlock else torch.tensor(0),
            **gate_info,
        }
        
        return new_status, new_predictions, info
    
    def _soft_gate(
        self, 
        q_values: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute soft gating probabilities for gradient flow.
        
        Uses temperature-scaled softmax over [reject, hypothesis, lock] options.
        """
        # Create pseudo-logits for 3 options
        # Higher Q -> higher lock prob, lower Q -> higher reject prob
        lock_logit = (q_values - self.thresholds.lock_threshold) / self.temperature
        reject_logit = (self.thresholds.reject_threshold - q_values) / self.temperature
        hyp_logit = torch.zeros_like(q_values)  # Baseline
        
        # Stack and softmax
        logits = torch.stack([reject_logit, hyp_logit, lock_logit], dim=-1)
        probs = F.softmax(logits, dim=-1)
        
        reject_probs = probs[..., 0]
        hyp_probs = probs[..., 1]
        lock_probs = probs[..., 2]
        
        return lock_probs, reject_probs, hyp_probs
    
    def apply_straight_through(
        self,
        q_values: torch.Tensor,
        hard_decisions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply straight-through estimator.
        
        Forward: use hard decisions
        Backward: gradient flows through soft Q values
        
        Args:
            q_values: Soft Q values (for gradient)
            hard_decisions: Discrete gating decisions
            
        Returns:
            Tensor that acts like hard_decisions in forward,
            but has gradients from q_values in backward.
        """
        # Straight-through: hard in forward, soft in backward
        return hard_decisions + q_values - q_values.detach()


class TerminationChecker(nn.Module):
    """
    Check if episode should terminate.
    
    Episode ends when:
    1. All non-GIVEN cells are LOCKED
    2. Maximum H cycles reached
    3. No progress for N cycles (optional)
    """
    
    def __init__(
        self,
        max_cycles: int = 8,
        patience: int = 2,  # Terminate if no locks for this many cycles
    ):
        super().__init__()
        self.max_cycles = max_cycles
        self.patience = patience
        
    def forward(
        self,
        status: torch.Tensor,
        cycle: int,
        prev_locked_count: Optional[torch.Tensor] = None,
        no_progress_count: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Check termination conditions.
        
        Args:
            status: Current cell status [batch_size, seq_len]
            cycle: Current H cycle number
            prev_locked_count: Locked count from previous cycle
            no_progress_count: Cycles without new locks
            
        Returns:
            done: Whether episode is done [batch_size]
            new_no_progress_count: Updated no-progress counter
            current_locked_count: Current locked count
        """
        batch_size = status.shape[0]
        device = status.device
        
        # Count locked cells
        current_locked_count = (status == CellStatus.LOCKED).sum(dim=-1)
        
        # All cells locked (excluding GIVEN)
        non_given = (status != CellStatus.GIVEN).sum(dim=-1)
        all_locked = current_locked_count >= non_given
        
        # Max cycles reached
        max_reached = torch.full((batch_size,), cycle >= self.max_cycles, device=device)
        
        # No progress tracking
        if prev_locked_count is not None and no_progress_count is not None:
            made_progress = current_locked_count > prev_locked_count
            new_no_progress_count = torch.where(
                made_progress,
                torch.zeros_like(no_progress_count),
                no_progress_count + 1
            )
            patience_exceeded = new_no_progress_count >= self.patience
        else:
            new_no_progress_count = torch.zeros((batch_size,), dtype=torch.long, device=device)
            patience_exceeded = torch.zeros((batch_size,), dtype=torch.bool, device=device)
        
        # Combine conditions
        done = all_locked | max_reached | patience_exceeded
        
        return done, new_no_progress_count, current_locked_count


class HypothesisStateManager:
    """
    Manages the full hypothesis state across H cycles.
    
    Tracks:
    - Cell status (GIVEN/UNFILLED/HYPOTHESIS/LOCKED)
    - Predictions
    - Q value history (for reward computation)
    - Transition events (for reward assignment)
    """
    
    def __init__(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        given_mask: torch.Tensor,  # Which cells are given
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        
        # Initialize status: GIVEN where given, UNFILLED elsewhere
        self.status = torch.where(
            given_mask,
            torch.full((batch_size, seq_len), CellStatus.GIVEN, device=device),
            torch.full((batch_size, seq_len), CellStatus.UNFILLED, device=device),
        )
        
        # Track predictions (0 = empty)
        self.predictions = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        
        # History for reward computation
        self.q_history = []
        self.status_history = [self.status.clone()]
        self.prediction_history = [self.predictions.clone()]
        
    def update(
        self,
        new_predictions: torch.Tensor,
        q_values: torch.Tensor,
        gating_module: HypothesisGate,
    ) -> Dict[str, torch.Tensor]:
        """
        Update state after one H cycle.
        
        Args:
            new_predictions: Model's new predictions [B, L]
            q_values: Q values for each cell [B, L]
            gating_module: HypothesisGate for state transitions
            
        Returns:
            Dict with transition info for reward computation
        """
        # Store Q values
        self.q_history.append(q_values.clone())
        
        # For unfilled cells, set to hypothesis if model made a prediction
        has_prediction = new_predictions > 0
        unfilled_mask = self.status == CellStatus.UNFILLED
        
        # UNFILLED + prediction -> HYPOTHESIS
        self.status = torch.where(
            unfilled_mask & has_prediction,
            torch.full_like(self.status, CellStatus.HYPOTHESIS),
            self.status
        )
        self.predictions = torch.where(
            unfilled_mask & has_prediction,
            new_predictions,
            self.predictions
        )
        
        # Apply gating (HYPOTHESIS -> LOCKED/REJECTED)
        new_status, updated_predictions, info = gating_module(
            q_values=q_values,
            status=self.status,
            predictions=self.predictions,
        )
        
        # Compute transitions for reward
        transitions = {
            'locked_this_cycle': (new_status == CellStatus.LOCKED) & (self.status != CellStatus.LOCKED),
            'rejected_this_cycle': info['num_rejected'] > 0,
            'unlocked_this_cycle': info.get('num_unlocked', torch.tensor(0)) > 0,
        }
        
        # Update state
        self.status = new_status
        self.predictions = updated_predictions
        
        # Store history
        self.status_history.append(self.status.clone())
        self.prediction_history.append(self.predictions.clone())
        
        return {**info, **transitions}
    
    def get_trajectory(self) -> Dict[str, torch.Tensor]:
        """Get full trajectory for training."""
        return {
            'status_history': torch.stack(self.status_history, dim=1),  # [B, T+1, L]
            'prediction_history': torch.stack(self.prediction_history, dim=1),  # [B, T+1, L]
            'q_history': torch.stack(self.q_history, dim=1) if self.q_history else None,  # [B, T, L]
        }

