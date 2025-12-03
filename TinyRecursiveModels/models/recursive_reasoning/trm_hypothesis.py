"""
TRM with Hypothesis-Driven Self-Correction.

Extends the base TRM model with:
1. Per-cell Q-head for gating decisions
2. V-head for advantage computation
3. Hypothesis state management
4. RL-compatible forward pass with trajectory collection

This model implements the discrete LOCK/HYPOTHESIS/REJECT mechanism
for iterative reasoning with self-correction.
"""

from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass, field
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.layers import (
    rms_norm, LinearSwish, SwiGLU, Attention, 
    RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
)
from models.sparse_embedding import CastedSparseEmbedding
from models.rl.heads import QHead, VHead, DuelingHead
from models.rl.gating import HypothesisGate, CellStatus, HypothesisStateManager, TerminationChecker
from models.rl.rewards import RewardComputer, EntropyComputer, GAEComputer

IGNORE_LABEL_ID = -100


class TRMHypothesisConfig(BaseModel):
    """Configuration for TRM with Hypothesis mechanism."""
    
    # Basic dimensions
    batch_size: int
    seq_len: int
    vocab_size: int
    hidden_size: int
    
    # Puzzle embeddings
    puzzle_emb_ndim: int = 0
    puzzle_emb_len: int = 16
    num_puzzle_identifiers: int = 1
    
    # Architecture
    H_cycles: int = 6
    L_cycles: int = 6
    L_layers: int = 2
    expansion: float = 4.0
    num_heads: int = 4
    pos_encodings: str = "rope"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # RL heads
    q_intermediate_size: Optional[int] = None
    v_intermediate_size: Optional[int] = None
    v_aggregation: str = "mean"
    head_dropout: float = 0.1
    
    # Gating thresholds
    lock_threshold: float = 0.6
    reject_threshold: float = -0.4
    unlock_threshold: float = -0.7
    allow_unlock: bool = True
    
    # Training
    max_H_cycles: int = 8
    forward_dtype: str = "bfloat16"
    detach_early_cycles: bool = True  # Only backprop through last cycle
    
    # Grid info (for Sudoku)
    grid_size: int = 4


@dataclass
class TRMHypothesisCarry:
    """State carried between H cycles."""
    z_H: torch.Tensor
    z_L: torch.Tensor
    status: torch.Tensor
    predictions: torch.Tensor
    cycle: int = 0


@dataclass
class TRMHypothesisTrajectory:
    """Trajectory data for RL training."""
    # States
    z_H_history: List[torch.Tensor] = field(default_factory=list)
    status_history: List[torch.Tensor] = field(default_factory=list)
    prediction_history: List[torch.Tensor] = field(default_factory=list)
    
    # Outputs
    logits_history: List[torch.Tensor] = field(default_factory=list)
    q_history: List[torch.Tensor] = field(default_factory=list)
    v_history: List[torch.Tensor] = field(default_factory=list)
    
    # Actions and rewards (filled during training)
    rewards: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dict for reward computation."""
        return {
            'status_history': torch.stack(self.status_history, dim=1),
            'prediction_history': torch.stack(self.prediction_history, dim=1),
            'q_history': torch.stack(self.q_history, dim=1) if self.q_history else None,
            'logits_history': torch.stack(self.logits_history, dim=1) if self.logits_history else None,
        }


class TRMHypothesisBlock(nn.Module):
    """Single transformer block for TRM."""
    
    def __init__(self, config: TRMHypothesisConfig):
        super().__init__()
        self.config = config
        
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps
        
    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post-norm attention
        hidden_states = rms_norm(
            hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states),
            variance_epsilon=self.norm_eps
        )
        # Post-norm MLP
        hidden_states = rms_norm(
            hidden_states + self.mlp(hidden_states),
            variance_epsilon=self.norm_eps
        )
        return hidden_states


class TRMHypothesisReasoningModule(nn.Module):
    """L-level reasoning module (stack of transformer blocks)."""
    
    def __init__(self, layers: List[TRMHypothesisBlock]):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        input_injection: torch.Tensor,
        cos_sin: CosSin,
    ) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin)
        return hidden_states


class TRMHypothesis(nn.Module):
    """
    TRM with Hypothesis-Driven Self-Correction.
    
    Key Features:
    1. Per-cell Q-head for gating (LOCK/HYPOTHESIS/REJECT)
    2. V-head for advantage computation
    3. Discrete gating with straight-through estimator
    4. Trajectory collection for PPO training
    """
    
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TRMHypothesisConfig(**config_dict)
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        
        # Embedding scale
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale
        
        # Token embeddings
        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size, 
            self.config.hidden_size, 
            init_std=embed_init_std, 
            cast_to=self.forward_dtype
        )
        
        # Output heads
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        
        # RL heads
        self.q_head = QHead(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.q_intermediate_size,
            dropout=self.config.head_dropout,
        )
        self.v_head = VHead(
            hidden_size=self.config.hidden_size,
            intermediate_size=self.config.v_intermediate_size,
            aggregation=self.config.v_aggregation,
            dropout=self.config.head_dropout,
        )
        
        # Puzzle embeddings (optional)
        self.puzzle_emb_len = 0
        if self.config.puzzle_emb_ndim > 0:
            self.puzzle_emb_len = (
                self.config.puzzle_emb_len if self.config.puzzle_emb_len > 0 
                else -(self.config.puzzle_emb_ndim // -self.config.hidden_size)
            )
            self.puzzle_emb = CastedSparseEmbedding(
                self.config.num_puzzle_identifiers,
                self.config.puzzle_emb_ndim,
                batch_size=self.config.batch_size,
                init_std=0,
                cast_to=self.forward_dtype
            )
            
        # Position encodings
        total_seq_len = self.config.seq_len + self.puzzle_emb_len
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=self.config.hidden_size // self.config.num_heads,
                max_position_embeddings=total_seq_len,
                base=self.config.rope_theta
            )
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(
                total_seq_len, 
                self.config.hidden_size, 
                init_std=embed_init_std, 
                cast_to=self.forward_dtype
            )
            
        # Reasoning layers
        self.L_level = TRMHypothesisReasoningModule(
            layers=[TRMHypothesisBlock(self.config) for _ in range(self.config.L_layers)]
        )
        
        # Initial states
        self.H_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        self.L_init = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True
        )
        
        # Gating module
        self.gating = HypothesisGate(
            lock_threshold=self.config.lock_threshold,
            reject_threshold=self.config.reject_threshold,
            unlock_threshold=self.config.unlock_threshold,
            allow_unlock=self.config.allow_unlock,
        )
        
        # Termination checker
        self.termination = TerminationChecker(
            max_cycles=self.config.max_H_cycles,
            patience=2,
        )
        
    def _input_embeddings(
        self, 
        inputs: torch.Tensor, 
        puzzle_identifiers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute input embeddings with optional puzzle embedding."""
        embedding = self.embed_tokens(inputs.to(torch.int32))
        
        if self.config.puzzle_emb_ndim > 0 and puzzle_identifiers is not None:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
                
            embedding = torch.cat(
                (puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding),
                dim=-2
            )
            
        if self.config.pos_encodings == "learned":
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))
            
        return self.embed_scale * embedding
    
    def _get_cos_sin(self) -> Optional[CosSin]:
        """Get rotary position embeddings."""
        if hasattr(self, 'rotary_emb'):
            return self.rotary_emb()
        return None
    
    def initial_carry(
        self, 
        batch_size: int, 
        given_mask: torch.Tensor,
        device: torch.device,
    ) -> TRMHypothesisCarry:
        """Initialize carry state for new puzzles."""
        seq_len = self.config.seq_len + self.puzzle_emb_len
        
        # Initialize hidden states
        z_H = self.H_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).clone()
        z_L = self.L_init.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1).clone()
        
        # Initialize status: GIVEN where given, UNFILLED elsewhere
        status = torch.where(
            given_mask,
            torch.full((batch_size, self.config.seq_len), CellStatus.GIVEN, device=device),
            torch.full((batch_size, self.config.seq_len), CellStatus.UNFILLED, device=device),
        )
        
        # Initialize predictions (0 = empty)
        predictions = torch.zeros((batch_size, self.config.seq_len), dtype=torch.long, device=device)
        
        return TRMHypothesisCarry(
            z_H=z_H.to(device),
            z_L=z_L.to(device),
            status=status,
            predictions=predictions,
            cycle=0,
        )
    
    def forward_one_cycle(
        self,
        carry: TRMHypothesisCarry,
        input_embeddings: torch.Tensor,
        cos_sin: Optional[CosSin],
        with_grad: bool = True,
    ) -> Tuple[TRMHypothesisCarry, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one H cycle of reasoning.
        
        Args:
            carry: Current state
            input_embeddings: Token embeddings
            cos_sin: Position embeddings
            with_grad: Whether to compute gradients
            
        Returns:
            new_carry: Updated state
            logits: Output logits [B, L, V]
            q_values: Q values [B, L]
            v_value: V value [B]
        """
        z_H, z_L = carry.z_H, carry.z_L
        
        context = torch.no_grad() if not with_grad else torch.enable_grad()
        
        with context:
            # L cycles (inner reasoning loop)
            for _ in range(self.config.L_cycles):
                z_L = self.L_level(z_L, z_H + input_embeddings, cos_sin=cos_sin)
            
            # H update
            z_H = self.L_level(z_H, z_L, cos_sin=cos_sin)
        
        # Output heads (always with grad for RL)
        # Slice off puzzle embedding tokens for output
        z_H_output = z_H[:, self.puzzle_emb_len:] if self.puzzle_emb_len > 0 else z_H
        
        logits = self.lm_head(z_H_output)
        q_values = self.q_head(z_H_output)
        v_value = self.v_head(z_H_output)
        
        new_carry = TRMHypothesisCarry(
            z_H=z_H,
            z_L=z_L,
            status=carry.status,
            predictions=carry.predictions,
            cycle=carry.cycle + 1,
        )
        
        return new_carry, logits, q_values, v_value
    
    def forward(
        self,
        inputs: torch.Tensor,
        given_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        puzzle_identifiers: Optional[torch.Tensor] = None,
        collect_trajectory: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with hypothesis mechanism.
        
        Args:
            inputs: Input tokens [B, L]
            given_mask: Which cells are given [B, L]
            labels: Ground truth (for training) [B, L]
            puzzle_identifiers: Puzzle IDs (optional)
            collect_trajectory: Whether to collect trajectory for RL
            
        Returns:
            Dict with logits, q_values, v_value, and optionally trajectory
        """
        batch_size = inputs.shape[0]
        device = inputs.device
        
        # Input embeddings
        input_embeddings = self._input_embeddings(inputs, puzzle_identifiers)
        cos_sin = self._get_cos_sin()
        
        # Initialize carry
        carry = self.initial_carry(batch_size, given_mask, device)
        
        # Copy given values to predictions
        carry.predictions = torch.where(given_mask, inputs, carry.predictions)
        
        # Trajectory collection
        trajectory = TRMHypothesisTrajectory() if collect_trajectory else None
        
        # Run H cycles
        for h in range(self.config.max_H_cycles):
            # Determine if this cycle should have gradients
            if self.config.detach_early_cycles and self.training:
                with_grad = (h == self.config.max_H_cycles - 1)
            else:
                with_grad = True
                
            # Forward one cycle
            carry, logits, q_values, v_value = self.forward_one_cycle(
                carry, input_embeddings, cos_sin, with_grad=with_grad
            )
            
            # Get predictions
            predictions = logits.argmax(dim=-1)
            
            # Collect trajectory
            if trajectory is not None:
                trajectory.z_H_history.append(carry.z_H.detach())
                trajectory.status_history.append(carry.status.clone())
                trajectory.prediction_history.append(carry.predictions.clone())
                trajectory.logits_history.append(logits)
                trajectory.q_history.append(q_values)
                trajectory.v_history.append(v_value)
            
            # Update predictions for unfilled cells
            is_unfilled = carry.status == CellStatus.UNFILLED
            is_hypothesis = carry.status == CellStatus.HYPOTHESIS
            can_update = is_unfilled | is_hypothesis
            
            # Always update predictions (before gating decision)
            carry.predictions = torch.where(can_update, predictions, carry.predictions)
            
            if self.training:
                # During TRAINING: Simple status update, NO rejection
                # This ensures supervised loss can learn
                carry.status = torch.where(
                    is_unfilled,
                    torch.full_like(carry.status, CellStatus.HYPOTHESIS),
                    carry.status
                )
                # Note: We track Q values for RL loss but don't gate during training
            else:
                # During INFERENCE: Apply full gating mechanism
                # Step 1: UNFILLED -> HYPOTHESIS
                carry.status = torch.where(
                    is_unfilled,
                    torch.full_like(carry.status, CellStatus.HYPOTHESIS),
                    carry.status
                )
                
                # Step 2: Apply gating (HYPOTHESIS -> LOCKED/REJECTED)
                new_status, new_predictions, gate_info = self.gating(
                    q_values=q_values,
                    status=carry.status,
                    predictions=carry.predictions,
                    training=False,
                )
                carry.status = new_status
                carry.predictions = new_predictions
            
            # Check termination
            done, _, _ = self.termination(carry.status, h + 1)
            if done.all():
                break
        
        # FORCE FINAL COMMITMENT: All HYPOTHESIS cells become LOCKED at the end
        # This prevents the model from gaming by rejecting everything
        is_hypothesis = carry.status == CellStatus.HYPOTHESIS
        carry.status = torch.where(
            is_hypothesis,
            torch.full_like(carry.status, CellStatus.LOCKED),
            carry.status
        )
                
        # Final trajectory status
        if trajectory is not None:
            trajectory.status_history.append(carry.status.clone())
            trajectory.prediction_history.append(carry.predictions.clone())
        
        # Prepare outputs
        outputs = {
            'logits': logits,
            'predictions': carry.predictions,
            'q_values': q_values,
            'v_value': v_value,
            'status': carry.status,
            'num_cycles': carry.cycle,
        }
        
        if trajectory is not None:
            outputs['trajectory'] = trajectory.to_dict()
            
        return outputs
    
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        mask: torch.Tensor,
        initial_entropy: Optional[torch.Tensor] = None,
        reward_computer: Optional[RewardComputer] = None,
        gae_computer: Optional[GAEComputer] = None,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined supervised + RL loss.
        
        Args:
            outputs: Forward pass outputs
            labels: Ground truth [B, L]
            mask: Valid cell mask [B, L]
            initial_entropy: Initial puzzle entropy [B]
            reward_computer: For computing rewards
            gae_computer: For computing advantages
            loss_weights: Weight for each loss term
            
        Returns:
            total_loss: Combined loss
            metrics: Dict with individual losses and stats
        """
        if loss_weights is None:
            loss_weights = {
                'supervised': 1.0,
                'ppo': 0.1,
                'q_head': 0.5,
                'v_head': 0.5,
                'entropy': 0.01,
            }
            
        metrics = {}
        total_loss = 0
        
        # ========== Supervised Loss ==========
        logits = outputs['logits']
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=IGNORE_LABEL_ID,
            reduction='none'
        ).view(labels.shape)
        
        # Masked mean
        ce_loss = (ce_loss * mask.float()).sum() / mask.sum().clamp(min=1)
        total_loss = total_loss + loss_weights['supervised'] * ce_loss
        metrics['loss_supervised'] = ce_loss.detach()
        
        # ========== RL Losses (if trajectory available) ==========
        if 'trajectory' in outputs and reward_computer is not None:
            trajectory = outputs['trajectory']
            
            # Compute rewards
            if initial_entropy is None:
                # Estimate entropy from mask
                initial_entropy = (~mask).sum(dim=-1).float() * math.log2(self.config.vocab_size)
                
            rewards, reward_info = reward_computer.compute_trajectory_rewards(
                trajectory=trajectory,
                labels=labels,
                initial_entropy=initial_entropy,
                max_cycles=self.config.max_H_cycles,
            )
            
            # Compute advantages
            if gae_computer is not None:
                q_history = trajectory['q_history']  # [B, T, L]
                T = q_history.shape[1]  # Number of timesteps
                
                # Aggregate rewards over cells
                rewards_agg = rewards.mean(dim=-1)  # [B, T]
                
                # Create done flags (all False except last)
                dones = torch.zeros_like(rewards_agg, dtype=torch.bool)
                dones[:, -1] = True
                
                # Create V estimates for each timestep
                # Use mean Q as proxy for V at each step
                v_estimates = q_history.mean(dim=-1)  # [B, T]
                
                # Pad V with final value for bootstrapping
                v_padded = torch.cat([v_estimates, outputs['v_value'].unsqueeze(-1)], dim=1)  # [B, T+1]
                
                advantages, returns = gae_computer.compute_gae(rewards_agg, v_padded, dones)
                advantages = gae_computer.normalize_advantages(advantages)
            else:
                # Simple advantage = reward - baseline
                advantages = rewards.mean(dim=-1) - outputs['v_value'].unsqueeze(-1)
                returns = rewards.mean(dim=-1)
            
            # ---------- PPO Loss ----------
            # Policy gradient with clipped objective
            logits_history = trajectory.get('logits_history')
            if logits_history is not None:
                # Get log probs for taken actions
                log_probs = F.log_softmax(logits_history, dim=-1)
                predictions = trajectory['prediction_history'][:, 1:]  # Skip initial state
                
                # Gather log probs for predictions
                # predictions: [B, T, L], log_probs: [B, T, L, V]
                action_log_probs = log_probs.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
                action_log_probs = (action_log_probs * mask.unsqueeze(1).float()).mean(dim=-1)  # [B, T]
                
                # PPO loss (simplified - without old log probs for first iteration)
                ppo_loss = -(action_log_probs * advantages.detach()).mean()
                total_loss = total_loss + loss_weights['ppo'] * ppo_loss
                metrics['loss_ppo'] = ppo_loss.detach()
            
            # ---------- Q-Head Loss ----------
            # Train Q to predict correctness directly (supervised)
            # Q should be high (+1) for correct predictions, low (-1) for wrong
            q_values = outputs['q_values']  # [B, L]
            predictions = outputs['predictions']
            is_correct = (predictions == labels).float()  # 1 if correct, 0 if wrong
            q_targets = is_correct * 2 - 1  # Map to [-1, 1]: correct=+1, wrong=-1
            
            # MSE loss: Q should match correctness
            q_loss = F.mse_loss(q_values[mask], q_targets[mask].detach())
            total_loss = total_loss + loss_weights['q_head'] * q_loss
            metrics['loss_q'] = q_loss.detach()
            
            # ---------- V-Head Loss ----------
            v_loss = F.mse_loss(outputs['v_value'], returns[:, -1].detach())
            total_loss = total_loss + loss_weights['v_head'] * v_loss
            metrics['loss_v'] = v_loss.detach()
            
            # ---------- Entropy Bonus ----------
            entropy = -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            entropy = (entropy * mask.float()).mean()
            total_loss = total_loss - loss_weights['entropy'] * entropy  # Negative for bonus
            metrics['entropy'] = entropy.detach()
            
            # Add reward info to metrics
            for k, v in reward_info.items():
                if isinstance(v, torch.Tensor):
                    metrics[f'reward_{k}'] = v.detach()
                else:
                    metrics[f'reward_{k}'] = v
        
        # Accuracy metrics
        with torch.no_grad():
            predictions = outputs['predictions']
            correct = (predictions == labels) & mask
            metrics['accuracy'] = correct.sum().float() / mask.sum().clamp(min=1)
            metrics['exact_accuracy'] = (correct.sum(dim=-1) == mask.sum(dim=-1)).float().mean()
            
        return total_loss, metrics

