"""
Rule-Based RL Agent with Multi-Sample PPO using Original TRM Backbone

Multi-Sample Approach:
1. Sample K actions from policy
2. Evaluate each action by checking Sudoku rules (instant reward)
3. V-head trained on average value of sampled actions
4. Q-head trained on best action value
5. Advantage = Q - V for PPO

Uses the same TRM architecture as the main project for fair comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, NamedTuple
from dataclasses import dataclass, field
import math

# Import original TRM components
from models.recursive_reasoning.trm import (
    TinyRecursiveReasoningModel_ACTV1Config,
    TinyRecursiveReasoningModel_ACTV1_Inner,
    TinyRecursiveReasoningModel_ACTV1InnerCarry,
)
from .env import ActionType, check_valid_move, check_valid_move_batched, count_violations_batched, is_solved_batched


@dataclass
class PolicyOutput:
    """Output from policy forward pass."""
    logits: torch.Tensor           # [B, seq_len, vocab_size]
    confidence: torch.Tensor       # [B, seq_len] max softmax prob
    entropy: torch.Tensor          # [B, seq_len] per-cell entropy
    action_type: torch.Tensor      # [B] FILL or UNFILL
    selected_cell: torch.Tensor    # [B] which cell
    selected_digit: torch.Tensor   # [B] which digit (for FILL)
    log_prob: torch.Tensor         # [B] log prob of full action
    value: torch.Tensor            # [B] state value V(s)


@dataclass
class MultiSampleOutput:
    """Output from multi-sample forward pass."""
    # Selected action (best from samples)
    action_type: torch.Tensor      # [B] FILL or UNFILL
    selected_cell: torch.Tensor    # [B] which cell
    selected_digit: torch.Tensor   # [B] which digit (for FILL)
    log_prob: torch.Tensor         # [B] log prob of selected action
    
    # Value estimates
    V: torch.Tensor                # [B] average value over K samples
    Q: torch.Tensor                # [B] value of best action
    advantage: torch.Tensor        # [B] Q - V
    
    # Sample info
    sample_rewards: torch.Tensor   # [B, K] rewards for each sample
    sample_valid: torch.Tensor     # [B, K] which samples were valid
    best_idx: torch.Tensor         # [B] index of best sample
    
    # Group consensus info (NEW!)
    consensus_score: torch.Tensor  # [B] avg consensus across cells (0-1)
    cell_entropy: torch.Tensor     # [B, num_cells] entropy per cell
    mode_digit: torch.Tensor       # [B] most common digit in group
    mode_count: torch.Tensor       # [B] how many chose the mode
    
    # For training
    logits: torch.Tensor           # [B, seq_len, vocab_size]
    confidence: torch.Tensor       # [B, seq_len]


class TRMBackbone(nn.Module):
    """
    Wrapper around original TRM for Rule-RL.
    
    Uses TinyRecursiveReasoningModel_ACTV1_Inner with exact same architecture
    as the supervised baseline.
    """
    
    def __init__(
        self,
        vocab_size: int = 6,
        hidden_size: int = 128,
        num_heads: int = 4,
        L_layers: int = 2,
        L_cycles: int = 4,   # 98% architecture
        H_cycles: int = 2,   # 98% architecture
        expansion: float = 4.0,
        seq_len: int = 16,
        batch_size: int = 64,
        detach_early_cycles: bool = False,  # Full gradients through all H-cycles
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles
        
        # Create TRM config matching original architecture
        config = TinyRecursiveReasoningModel_ACTV1Config(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            expansion=expansion,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            H_layers=0,  # Ignored
            L_layers=L_layers,
            pos_encodings="rope",
            rms_norm_eps=1e-5,
            rope_theta=10000.0,
            puzzle_emb_ndim=0,
            puzzle_emb_len=0,
            num_puzzle_identifiers=1,
            halt_max_steps=H_cycles,
            halt_exploration_prob=0.0,
            forward_dtype="bfloat16",
            detach_early_cycles=detach_early_cycles,  # Full gradients if False
        )
        
        self.config = config
        self.trm = TinyRecursiveReasoningModel_ACTV1_Inner(config)
        self.forward_dtype = self.trm.forward_dtype
        
    def forward(self, x: torch.Tensor, return_q_logits: bool = False):
        """
        Forward pass using original TRM.
        
        Args:
            x: Input tokens [B, seq_len]
            return_q_logits: If True, also return q_halt and q_continue logits for ACT head training
            
        Returns:
            logits: [B, seq_len, vocab_size]
            hidden: [B, seq_len, hidden_size] for value head
            (optional) q_logits: Tuple of (q_halt, q_continue) for ACT head
        """
        B = x.shape[0]
        device = x.device
        
        # Create batch dict for TRM
        batch = {
            "inputs": x,
            "puzzle_identifiers": torch.zeros(B, dtype=torch.long, device=device),
        }
        
        # Initialize carry
        carry = self.trm.empty_carry(B)
        carry.z_H = carry.z_H.to(device)
        carry.z_L = carry.z_L.to(device)
        
        # Reset carry to initial values
        reset_flag = torch.ones(B, dtype=torch.bool, device=device)
        carry = self.trm.reset_carry(reset_flag, carry)
        
        # Forward through TRM (returns new_carry, logits, q_logits, aux_loss)
        new_carry, logits, q_logits, aux_loss = self.trm(carry, batch)
        
        # Get hidden states for value head
        hidden = new_carry.z_H.float()  # Convert from bfloat16
        
        if return_q_logits:
            return logits.float(), hidden, q_logits
        return logits.float(), hidden


class PPOAgent(nn.Module):
    """
    Multi-Sample PPO Agent using Original TRM Backbone.
    
    Key innovation: Sample K actions, evaluate all with rule checking,
    train V on average, Q on best, use Q-V as advantage for PPO.
    """
    
    def __init__(
        self,
        grid_size: int = 4,
        hidden_size: int = 128,
        num_heads: int = 4,
        L_layers: int = 2,
        L_cycles: int = 4,   # 98% architecture
        H_cycles: int = 2,   # 98% architecture
        batch_size: int = 64,
        temperature: float = 1.0,
        num_samples: int = 8,  # K samples for multi-sample
        # Reward values for instant rule checking
        reward_valid: float = 3.0,
        reward_invalid: float = -1.0,
        detach_early_cycles: bool = False,  # Full gradients through all H-cycles
    ):
        super().__init__()
        
        self.grid_size = grid_size
        self.seq_len = grid_size * grid_size
        self.vocab_size = grid_size + 2
        self.temperature = temperature
        self.hidden_size = hidden_size
        self.num_samples = num_samples
        self.detach_early_cycles = detach_early_cycles
        self.reward_valid = reward_valid
        self.reward_invalid = reward_invalid
        
        # Original TRM Backbone
        self.backbone = TRMBackbone(
            vocab_size=self.vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            L_layers=L_layers,
            L_cycles=L_cycles,
            H_cycles=H_cycles,
            seq_len=self.seq_len,
            batch_size=batch_size,
            detach_early_cycles=detach_early_cycles,
        )
        
        # V-head: Predicts AVERAGE value over sampled actions
        # V(s) = E_a~π[Q(s,a)] ≈ (1/K) Σ r(s, aᵢ)
        self.V_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Q-head: Predicts value of BEST action
        # Q(s, a*) where a* = argmax_a r(s, a)
        self.Q_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )
        
        # Action type head (FILL vs UNFILL)
        self.action_type_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 2),
        )
    
    def forward(
        self,
        board: torch.Tensor,
        fill_mask: torch.Tensor,
        unfill_mask: torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """
        Select action given current state.
        """
        B = board.shape[0]
        device = board.device
        
        # Forward through TRM backbone
        logits, hidden = self.backbone(board)
        
        # Softmax and confidence
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)
        
        # Value estimate (mean pooling) - use V_head
        value = self.V_head(hidden.mean(dim=1)).squeeze(-1)
        
        # Decide action type
        global_hidden = hidden.mean(dim=1)
        action_type_logits = self.action_type_head(global_hidden)
        
        # Mask invalid action types
        can_fill = fill_mask.any(dim=-1)
        can_unfill = unfill_mask.any(dim=-1)
        
        action_type_mask = torch.stack([can_fill, can_unfill], dim=-1).float()
        action_type_logits = action_type_logits - 1e9 * (1 - action_type_mask)
        
        if deterministic:
            action_type = action_type_logits.argmax(dim=-1)
        else:
            action_type_probs = F.softmax(action_type_logits, dim=-1)
            action_type = torch.multinomial(action_type_probs, 1).squeeze(-1)
        
        # Select cell and digit
        selected_cell = torch.zeros(B, dtype=torch.long, device=device)
        selected_digit = torch.zeros(B, dtype=torch.long, device=device)
        log_prob = torch.zeros(B, device=device)
        
        for b in range(B):
            if action_type[b] == ActionType.FILL:
                # FILL: Select most confident empty cell
                cell_scores = confidence[b].clone()
                cell_scores[~fill_mask[b]] = -float('inf')
                
                if deterministic:
                    cell = cell_scores.argmax()
                else:
                    valid_scores = cell_scores.clone()
                    valid_scores[valid_scores == -float('inf')] = -1e9
                    cell_probs = F.softmax(valid_scores, dim=-1)
                    cell = torch.multinomial(cell_probs, 1).squeeze()
                
                selected_cell[b] = cell
                
                # Select digit - ONLY from valid digits (tokens 2 to vocab_size-1)
                # Token 0 = padding, Token 1 = empty, Tokens 2+ = actual digits
                digit_logits = logits[b, cell].clone()
                # Mask out invalid tokens (0=padding, 1=empty)
                digit_logits[0] = -float('inf')  # padding
                digit_logits[1] = -float('inf')  # empty
                
                if deterministic:
                    digit = digit_logits.argmax()
                else:
                    digit_probs = F.softmax(digit_logits / self.temperature, dim=-1)
                    digit = torch.multinomial(digit_probs, 1).squeeze()
                
                selected_digit[b] = digit
                
                # Log prob
                log_p_type = F.log_softmax(action_type_logits[b], dim=-1)[ActionType.FILL]
                valid_scores = confidence[b].clone()
                valid_scores[~fill_mask[b]] = -1e9
                log_p_cell = F.log_softmax(valid_scores, dim=-1)[cell]
                log_p_digit = F.log_softmax(digit_logits / self.temperature, dim=-1)[digit]
                log_prob[b] = log_p_type + log_p_cell + log_p_digit
                
            else:
                # UNFILL: Select least confident filled cell
                cell_scores = -confidence[b].clone()
                cell_scores[~unfill_mask[b]] = -float('inf')
                
                if deterministic:
                    cell = cell_scores.argmax()
                else:
                    valid_scores = cell_scores.clone()
                    valid_scores[valid_scores == -float('inf')] = -1e9
                    cell_probs = F.softmax(valid_scores, dim=-1)
                    cell = torch.multinomial(cell_probs, 1).squeeze()
                
                selected_cell[b] = cell
                selected_digit[b] = 0
                
                log_p_type = F.log_softmax(action_type_logits[b], dim=-1)[ActionType.UNFILL]
                valid_scores = -confidence[b].clone()
                valid_scores[~unfill_mask[b]] = -1e9
                log_p_cell = F.log_softmax(valid_scores, dim=-1)[cell]
                log_prob[b] = log_p_type + log_p_cell
        
        return PolicyOutput(
            logits=logits,
            confidence=confidence,
            entropy=entropy,
            action_type=action_type,
            selected_cell=selected_cell,
            selected_digit=selected_digit,
            log_prob=log_prob,
            value=value,
        )
    
    def forward_multi_sample(
        self,
        board: torch.Tensor,
        fill_mask: torch.Tensor,
        unfill_mask: torch.Tensor,
        K: Optional[int] = None,
    ) -> MultiSampleOutput:
        """
        Multi-sample forward pass.
        
        1. Sample K actions from policy
        2. Evaluate each action by checking Sudoku rules
        3. Compute V = average reward, Q = best reward
        4. Return best action and advantage = Q - V
        
        Args:
            board: [B, seq_len] current board state
            fill_mask: [B, seq_len] which cells can be filled
            unfill_mask: [B, seq_len] which cells can be unfilled
            K: number of samples (default: self.num_samples)
        """
        K = K or self.num_samples
        B = board.shape[0]
        device = board.device
        
        # Forward through backbone once
        logits, hidden = self.backbone(board)
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        
        # Get V and Q estimates from heads (for training targets)
        global_hidden = hidden.mean(dim=1)
        V_pred = self.V_head(global_hidden).squeeze(-1)
        Q_pred = self.Q_head(global_hidden).squeeze(-1)
        
        # ================================================================
        # VECTORIZED SAMPLING - Sample K actions for all B boards at once
        # ================================================================
        
        # Prepare cell selection scores: mask out non-fillable cells
        cell_scores = confidence.clone()  # [B, seq_len]
        cell_scores[~fill_mask] = -1e9
        cell_log_probs = F.log_softmax(cell_scores, dim=-1)  # [B, seq_len]
        cell_probs = F.softmax(cell_scores, dim=-1)  # [B, seq_len]
        
        # Sample K cells for each batch item: [B, K]
        # Use replacement=True to allow same cell multiple times
        sample_cells = torch.multinomial(cell_probs, K, replacement=True)  # [B, K]
        
        # Gather digit logits for sampled cells: [B, K, vocab_size]
        # sample_cells is [B, K], logits is [B, seq_len, vocab_size]
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, K)  # [B, K]
        sampled_digit_logits = logits[batch_idx, sample_cells]  # [B, K, vocab_size]
        
        # Mask invalid digits (0=pad, 1=empty)
        sampled_digit_logits[:, :, 0] = -1e9
        sampled_digit_logits[:, :, 1] = -1e9
        
        # Sample digits: reshape to [B*K, vocab], sample, reshape back
        # Use higher temperature during training for exploration
        explore_temp = self.temperature * 2.0  # Double temperature for more exploration
        digit_probs = F.softmax(sampled_digit_logits / explore_temp, dim=-1)  # [B, K, vocab]
        
        # Epsilon-greedy: 10% random actions to prevent policy collapse
        epsilon = 0.1
        num_valid_digits = self.grid_size  # 4 for 4x4
        uniform_probs = torch.zeros_like(digit_probs)
        uniform_probs[:, :, 2:2+num_valid_digits] = 1.0 / num_valid_digits  # uniform over valid digits
        digit_probs = (1 - epsilon) * digit_probs + epsilon * uniform_probs
        
        digit_probs_flat = digit_probs.view(B * K, -1)
        sample_digits_flat = torch.multinomial(digit_probs_flat, 1).squeeze(-1)  # [B*K]
        sample_digits = sample_digits_flat.view(B, K)  # [B, K]
        
        # ================================================================
        # VECTORIZED VALIDITY CHECK - Check all B*K moves at once!
        # ================================================================
        boards_expanded = board.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)  # [B*K, seq_len]
        cells_flat = sample_cells.view(B * K)  # [B*K]
        digits_flat = sample_digits.view(B * K)  # [B*K]
        
        # Binary valid check (for metrics)
        sample_valid_flat = check_valid_move_batched(boards_expanded, cells_flat, digits_flat, self.grid_size)
        sample_valid = sample_valid_flat.view(B, K)  # [B, K]
        
        # ================================================================
        # VIOLATION-BASED REWARD - Penalize constraint violations
        # ================================================================
        # Apply each action to get resulting boards
        boards_after_action = boards_expanded.clone()
        boards_after_action[torch.arange(B * K, device=device), cells_flat] = digits_flat
        
        # Count violations BEFORE and AFTER action
        violations_before = count_violations_batched(boards_expanded, self.grid_size)
        violations_after = count_violations_batched(boards_after_action, self.grid_size)
        
        # Reward = reduction in violations (positive = good)
        # If action reduces violations: +reward
        # If action creates violations: -penalty
        violation_change = violations_before - violations_after  # [B*K], positive = improvement
        
        # Normalize by max possible violations (~36 for 4x4: each digit could violate all 12 constraints)
        # Scale to roughly [-1, 1] range
        max_violations = self.grid_size * 3 * self.grid_size  # 48 for 4x4
        
        # Base reward: normalized violation change
        base_reward = violation_change / 3.0  # Scale so 1 violation change = ~0.33
        
        # Penalty for absolute violations (encourages clean boards)
        violation_penalty = -violations_after / max_violations  # [0, -1]
        
        # Big bonus for solving!
        solved = is_solved_batched(boards_after_action, self.grid_size).float()
        solve_bonus = solved * 5.0  # +5 for solving!
        
        # Combine: change + penalty + solve bonus
        sample_rewards = (base_reward + violation_penalty * 0.5 + solve_bonus).view(B, K)  # [B, K]
        
        # ================================================================
        # VECTORIZED LOG PROBS
        # ================================================================
        # Cell log probs: gather from cell_log_probs [B, seq_len]
        sample_cell_log_probs = cell_log_probs.gather(1, sample_cells)  # [B, K]
        
        # Digit log probs: need to gather from [B, K, vocab] using [B, K] indices
        digit_log_probs = F.log_softmax(sampled_digit_logits / self.temperature, dim=-1)  # [B, K, vocab]
        sample_digit_log_probs = digit_log_probs.gather(2, sample_digits.unsqueeze(-1)).squeeze(-1)  # [B, K]
        
        sample_log_probs = sample_cell_log_probs + sample_digit_log_probs  # [B, K]
        
        # ================================================================
        # GROUP CONSENSUS ANALYSIS - Vectorized!
        # ================================================================
        num_digits = self.grid_size + 2  # 0-5 for 4x4 (pad, empty, 1-4)
        seq_len = board.shape[1]
        
        # Build digit histogram per cell using scatter_add: [B, seq_len, num_digits]
        digit_counts = torch.zeros(B, seq_len, num_digits, device=device)
        
        # Create indices for scatter: [B, K] -> need to scatter into [B, seq_len, num_digits]
        # First create combined index: batch_idx * seq_len * num_digits + cell * num_digits + digit
        flat_idx = (
            torch.arange(B, device=device).unsqueeze(1) * seq_len * num_digits +
            sample_cells * num_digits +
            sample_digits
        )  # [B, K]
        
        # Only count valid digits (>= 2)
        valid_digit_mask = sample_digits >= 2
        ones = torch.ones_like(flat_idx, dtype=torch.float32)
        ones[~valid_digit_mask] = 0
        
        digit_counts_flat = digit_counts.view(-1)
        digit_counts_flat.scatter_add_(0, flat_idx.view(-1), ones.view(-1))
        digit_counts = digit_counts_flat.view(B, seq_len, num_digits)
        
        # Compute entropy per cell: H = -sum(p * log(p))
        cell_totals = digit_counts.sum(dim=-1, keepdim=True).clamp(min=1)
        cell_probs_dist = digit_counts / cell_totals
        log_probs_safe = torch.log(cell_probs_dist + 1e-10)
        cell_entropy = -(cell_probs_dist * log_probs_safe).sum(dim=-1)  # [B, seq_len]
        
        # Cells that were sampled
        cells_sampled = cell_totals.squeeze(-1) > 0  # [B, seq_len]
        
        # Normalize entropy: max = log(4) for uniform over 4 digits
        max_entropy = torch.log(torch.tensor(max(num_digits - 2, 1), dtype=torch.float32, device=device))
        norm_entropy = cell_entropy / (max_entropy + 1e-8)  # [0, 1]
        
        # Consensus = 1 - avg_entropy (high when group agrees)
        sampled_entropy = (norm_entropy * cells_sampled.float()).sum(dim=1)
        num_sampled = cells_sampled.sum(dim=1).float().clamp(min=1)
        consensus_score = 1.0 - (sampled_entropy / num_sampled)  # [B]
        
        # Find mode digit across all samples per batch - vectorized
        # Create histogram per batch: [B, num_digits]
        digit_hist = torch.zeros(B, num_digits, device=device)
        batch_digit_idx = (
            torch.arange(B, device=device).unsqueeze(1) * num_digits + sample_digits
        )  # [B, K]
        digit_hist_flat = digit_hist.view(-1)
        digit_hist_flat.scatter_add_(0, batch_digit_idx.view(-1), torch.ones(B * K, device=device))
        digit_hist = digit_hist_flat.view(B, num_digits)
        digit_hist[:, :2] = 0  # ignore pad/empty
        mode_digit = digit_hist.argmax(dim=1)
        mode_count = digit_hist.max(dim=1).values.long()
        
        # CONSENSUS BONUS - DISABLED (can cause "confident wrong" behavior)
        # The consensus bonus was encouraging groupthink, which reinforces biases
        # positive_reward = (sample_rewards > 0).float()  # [B, K]
        # consensus_bonus = consensus_score.unsqueeze(1) * positive_reward * 0.1  # [B, K]
        # sample_rewards = sample_rewards + consensus_bonus
        
        # Compute V = average reward over samples
        V_empirical = sample_rewards.mean(dim=1)
        
        # Compute Q = best reward (select best action)
        Q_empirical, best_idx = sample_rewards.max(dim=1)
        
        # Advantage = Q - V
        advantage = Q_empirical - V_empirical
        
        # Select best action
        selected_cell = sample_cells[torch.arange(B, device=device), best_idx]
        selected_digit = sample_digits[torch.arange(B, device=device), best_idx]
        selected_log_prob = sample_log_probs[torch.arange(B, device=device), best_idx]
        
        return MultiSampleOutput(
            action_type=torch.zeros(B, dtype=torch.long, device=device),
            selected_cell=selected_cell,
            selected_digit=selected_digit,
            log_prob=selected_log_prob,
            V=V_empirical,
            Q=Q_empirical,
            advantage=advantage,
            sample_rewards=sample_rewards,
            sample_valid=sample_valid,
            best_idx=best_idx,
            consensus_score=consensus_score,
            cell_entropy=cell_entropy,
            mode_digit=mode_digit,
            mode_count=mode_count,
            logits=logits,
            confidence=confidence,
        )
    
    def compute_multi_sample_loss(
        self,
        board: torch.Tensor,
        fill_mask: torch.Tensor,
        K: int = 8,
        old_log_probs: Optional[torch.Tensor] = None,
        clip_range: float = 0.2,
        kl_coef: float = 0.01,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO-style loss for multi-sample RL.
        
        1. V-head loss: MSE between V_pred and V_empirical (average reward)
        2. Q-head loss: MSE between Q_pred and Q_empirical (best reward)
        3. Policy loss: PPO clipped objective with group-relative advantages
        4. KL divergence penalty: Prevent policy from drifting too far
        
        This is similar to GRPO (DeepSeek-R1) but for Sudoku.
        """
        B = board.shape[0]
        device = board.device
        
        # Get multi-sample output
        output = self.forward_multi_sample(board, fill_mask, fill_mask, K)
        
        # Get predictions from heads
        _, hidden = self.backbone(board)
        global_hidden = hidden.mean(dim=1)
        V_pred = self.V_head(global_hidden).squeeze(-1)
        Q_pred = self.Q_head(global_hidden).squeeze(-1)
        
        # V-head loss: predict average reward
        V_loss = F.mse_loss(V_pred, output.V.detach())
        
        # Q-head loss: predict VALID RATE instead of max reward
        # This varies by state difficulty (unlike max reward which is constant)
        # Q now learns: "how easy is this state?" 
        valid_rate_target = output.sample_valid.float().mean(dim=1)  # [B]
        Q_loss = F.mse_loss(Q_pred, valid_rate_target.detach())
        
        # Group-relative advantages (GRPO style)
        # Normalize within batch: (adv - mean) / std
        advantages = output.advantage.detach()
        if advantages.numel() > 1 and advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = advantages - advantages.mean()
        
        # PPO clipped policy loss
        if old_log_probs is not None:
            # Full PPO: use ratio with old policy
            ratio = torch.exp(output.log_prob - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
        else:
            # REINFORCE-style (first iteration or no old_log_probs)
            policy_loss = -(output.log_prob * advantages).mean()
        
        # KL divergence penalty (GRPO style)
        # KL(π || π_ref) prevents policy from drifting too far
        if ref_log_probs is not None:
            # KL = log(π) - log(π_ref) = new_log_prob - ref_log_prob
            kl_div = (output.log_prob - ref_log_probs).mean()
        else:
            # Approximate KL using entropy change (self-KL regularization)
            # Higher entropy = more exploration = good
            kl_div = torch.tensor(0.0, device=device)
        
        # Entropy bonus (encourages exploration)
        probs = F.softmax(output.logits / self.temperature, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss (GRPO style)
        # L = L_V + L_Q + L_policy + β × KL - α × entropy
        # Q now predicts valid_rate (varies) instead of max reward (constant)
        entropy_coef = 0.05  # Increased from 0.01 for more exploration
        total_loss = V_loss + Q_loss + policy_loss + kl_coef * kl_div - entropy_coef * entropy
        
        # Compute metrics
        avg_reward = output.sample_rewards.mean().item()
        
        metrics = {
            'V_loss': V_loss.item(),
            'Q_loss': Q_loss.item(),
            'policy_loss': policy_loss.item(),
            'kl_div': kl_div.item() if isinstance(kl_div, torch.Tensor) else kl_div,
            'entropy': entropy.item(),
            'V_empirical': output.V.mean().item(),
            'advantage': output.advantage.mean().item(),
            'valid_rate': output.sample_valid.float().mean().item(),  # Binary: any NEW conflict?
            'avg_reward': avg_reward,  # Violation-based reward
            'consensus': output.consensus_score.mean().item(),
        }
        
        return total_loss, metrics
    
    def evaluate_actions(
        self,
        boards: torch.Tensor,
        fill_masks: torch.Tensor,
        unfill_masks: torch.Tensor,
        action_types: torch.Tensor,
        cells: torch.Tensor,
        digits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate log probabilities for given actions (for PPO update)."""
        B = boards.shape[0]
        device = boards.device
        
        logits, hidden = self.backbone(boards)
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        
        values = self.V_head(hidden.mean(dim=1)).squeeze(-1)
        
        global_hidden = hidden.mean(dim=1)
        action_type_logits = self.action_type_head(global_hidden)
        
        can_fill = fill_masks.any(dim=-1)
        can_unfill = unfill_masks.any(dim=-1)
        action_type_mask = torch.stack([can_fill, can_unfill], dim=-1).float()
        action_type_logits = action_type_logits - 1e9 * (1 - action_type_mask)
        
        log_probs = torch.zeros(B, device=device)
        entropies = torch.zeros(B, device=device)
        
        for b in range(B):
            log_p_type = F.log_softmax(action_type_logits[b], dim=-1)[action_types[b]]
            
            if action_types[b] == ActionType.FILL:
                cell_scores = confidence[b].clone()
                cell_scores[~fill_masks[b]] = -1e9
                log_p_cell = F.log_softmax(cell_scores, dim=-1)[cells[b]]
                
                # Mask out invalid digits (0=padding, 1=empty)
                digit_logits = logits[b, cells[b]].clone()
                digit_logits[0] = -float('inf')
                digit_logits[1] = -float('inf')
                log_p_digit = F.log_softmax(digit_logits / self.temperature, dim=-1)[digits[b]]
                
                log_probs[b] = log_p_type + log_p_cell + log_p_digit
                
                digit_probs = F.softmax(digit_logits / self.temperature, dim=-1)
                entropies[b] = -(digit_probs * torch.log(digit_probs + 1e-8)).sum()
            else:
                cell_scores = -confidence[b].clone()
                cell_scores[~unfill_masks[b]] = -1e9
                log_p_cell = F.log_softmax(cell_scores, dim=-1)[cells[b]]
                
                log_probs[b] = log_p_type + log_p_cell
                entropies[b] = 0.0
        
        return log_probs, entropies, values
    
    def get_value(self, board: torch.Tensor) -> torch.Tensor:
        """Get state value estimate (V)."""
        _, hidden = self.backbone(board)
        return self.V_head(hidden.mean(dim=1)).squeeze(-1)
    
    def get_Q(self, board: torch.Tensor) -> torch.Tensor:
        """Get Q value estimate (best action value)."""
        _, hidden = self.backbone(board)
        return self.Q_head(hidden.mean(dim=1)).squeeze(-1)
    
    def get_log_probs_for_grpo(
        self,
        board: torch.Tensor,
        fill_mask: torch.Tensor,
        cells: torch.Tensor,
        digits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probabilities for given (cell, digit) actions.
        Used for GRPO KL divergence calculation.
        """
        B = board.shape[0]
        device = board.device
        
        logits, hidden = self.backbone(board)
        scaled_logits = logits / self.temperature
        probs = F.softmax(scaled_logits, dim=-1)
        confidence = probs.max(dim=-1).values
        
        log_probs = torch.zeros(B, device=device)
        
        for b in range(B):
            # Cell selection log prob
            cell_scores = confidence[b].clone()
            cell_scores[~fill_mask[b]] = -1e9
            log_p_cell = F.log_softmax(cell_scores, dim=-1)[cells[b]]
            
            # Digit selection log prob
            digit_logits = logits[b, cells[b]].clone()
            digit_logits[0] = -1e9
            digit_logits[1] = -1e9
            log_p_digit = F.log_softmax(digit_logits / self.temperature, dim=-1)[digits[b]]
            
            log_probs[b] = log_p_cell + log_p_digit
        
        return log_probs


# Backwards compatibility
RuleRLAgent = PPOAgent
