"""
Mixture of Experts (MoE) Layer - DeepSeek Style

DeepSeek MoE architecture:
- Shared/Persistent Expert: Always activated for all tokens
- Routed Experts: Top-K experts selected by a learned router

Key design principle:
- Each expert (shared + routed) has size = original_FFN / (1 + top_k)
- Activated compute = (1 shared + top_k routed) × expert_size = 1x original
- More experts (N) = more total params, but SAME compute per token

Example with num_experts=8, top_k=2, shared=True:
- Each expert: original_inter / 3 (since 1+2=3 activated)
- Total params: (8 routed + 1 shared) × (original/3) = 3x original FFN
- Activated compute: 3 × (original/3) = 1x original ✓

Key advantages:
- Expert specialization without compute overhead
- Shared expert captures common patterns
- Load balancing loss prevents expert collapse
- Scalable: add more experts for more capacity at same compute cost
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from models.layers import CastedLinear, _find_multiple


class Expert(nn.Module):
    """Single SwiGLU expert with configurable intermediate size."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        # Use explicit intermediate_size (not expansion-based)
        self.gate_up_proj = CastedLinear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = CastedLinear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


class Router(nn.Module):
    """Top-K router with auxiliary load balancing loss."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        jitter_noise: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.jitter_noise = jitter_noise
        self.hidden_size = hidden_size
        
        # Router linear layer weight (will be cast to input dtype)
        self.gate_weight = nn.Parameter(
            torch.randn(num_experts, hidden_size) * (1.0 / hidden_size ** 0.5)
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states: [B, S, D]
            
        Returns:
            router_probs: [B, S, K] - weights for top-K experts
            selected_experts: [B, S, K] - indices of top-K experts
            router_logits: [B, S, num_experts] - for auxiliary loss
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Add jitter noise during training for exploration
        if self.training and self.jitter_noise > 0:
            hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        
        # Router logits - cast weight to input dtype (handles bfloat16)
        router_logits = F.linear(hidden_states, self.gate_weight.to(hidden_states.dtype))  # [B, S, num_experts]
        
        # Top-K selection with softmax normalization (compute in float32 for stability)
        router_probs = F.softmax(router_logits.float(), dim=-1).to(hidden_states.dtype)
        
        # Select top-K experts
        top_k_probs, selected_experts = torch.topk(
            router_probs, self.top_k, dim=-1
        )  # [B, S, K]
        
        # Renormalize top-K probabilities
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)
        
        return top_k_probs, selected_experts, router_logits


class MoELayer(nn.Module):
    """
    DeepSeek-style Mixture of Experts layer with PROPERLY SIZED experts.
    
    Design principle:
    - Each expert (shared + routed) has size = original_FFN / (1 + top_k)
    - Activated compute = (1 shared + top_k routed) × expert_size = 1x original
    - More experts (N) = more total params, but SAME compute
    
    Example with num_experts=8, top_k=2, shared=True:
    - Each expert: original_inter / 3 (since 1+2=3 activated)
    - Total params: (8 + 1) × (original/3) = 3x original FFN params
    - Activated compute: 3 × (original/3) = 1x original ✓
    
    This allows expert specialization while maintaining constant compute budget.
    
    Output = shared_expert(x) + sum_k(router_weight_k * expert_k(x))
    """
    
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        num_experts: int = 4,
        top_k: int = 2,
        shared_expert: bool = True,
        jitter_noise: float = 0.01,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert_enabled = shared_expert
        self.aux_loss_coef = aux_loss_coef
        
        # Compute original FFN intermediate size
        original_inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        
        # All experts have SAME size = original / (1 + top_k)
        # This ensures: activated = (1 shared + top_k routed) × expert_size = original
        num_activated = (1 if shared_expert else 0) + top_k
        expert_inter = max(16, original_inter // num_activated)  # Allow small experts for small models
        
        self.original_inter = original_inter
        self.expert_inter = expert_inter
        self.num_activated = num_activated
        
        # Shared/persistent expert (same size as routed experts)
        if shared_expert:
            self.shared_expert = Expert(hidden_size, expert_inter)
        
        # Routed experts (same size, N total, top_k activated)
        self.experts = nn.ModuleList([
            Expert(hidden_size, expert_inter) for _ in range(num_experts)
        ])
        
        # Router
        self.router = Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )
        
        # Store auxiliary loss for training
        self._aux_loss = None
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        """Return the auxiliary load balancing loss."""
        return self._aux_loss
    
    def _compute_aux_loss(
        self,
        router_logits: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute auxiliary load balancing loss.
        
        Encourages uniform routing across experts to prevent collapse.
        
        loss = num_experts * sum_i(f_i * p_i)
        where:
            f_i = fraction of tokens routed to expert i
            p_i = average router probability for expert i
        """
        batch_size, seq_len, _ = router_logits.shape
        num_tokens = batch_size * seq_len
        
        # Router probabilities (compute in float32 for stability)
        router_probs = F.softmax(router_logits.float(), dim=-1)  # [B, S, E]
        
        # Average probability per expert
        avg_probs = router_probs.mean(dim=(0, 1))  # [E]
        
        # Fraction of tokens assigned to each expert (from top-k selection)
        # Count how often each expert appears in selected_experts
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)  # [B, S, K, E]
        expert_counts = expert_mask.sum(dim=2).float()  # [B, S, E]
        expert_fractions = expert_counts.sum(dim=(0, 1)) / (num_tokens * self.top_k)  # [E]
        
        # Load balancing loss
        aux_loss = self.num_experts * (avg_probs * expert_fractions).sum()
        
        return aux_loss
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, S, D]
            
        Returns:
            output: [B, S, D]
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Initialize output
        final_output = torch.zeros_like(hidden_states)
        
        # 1. Shared expert (always active)
        if self.shared_expert_enabled:
            shared_output = self.shared_expert(hidden_states)
            final_output = final_output + shared_output
        
        # 2. Route to top-K experts
        router_probs, selected_experts, router_logits = self.router(hidden_states)
        # router_probs: [B, S, K]
        # selected_experts: [B, S, K]
        
        # Compute auxiliary loss during training
        if self.training:
            self._aux_loss = self._compute_aux_loss(router_logits, selected_experts)
        else:
            self._aux_loss = None
        
        # 3. Compute weighted expert outputs
        # Simple loop approach - trades some redundant compute for GPU efficiency
        # Each expert processes ALL tokens, then we mask by selection
        # This avoids expensive indexing operations that cause CPU-GPU sync
        
        for k in range(self.top_k):
            expert_indices = selected_experts[:, :, k]  # [B, S]
            expert_weights = router_probs[:, :, k:k+1]  # [B, S, 1]
            
            for e in range(self.num_experts):
                # Mask where this expert was selected for slot k
                mask = (expert_indices == e).unsqueeze(-1).to(hidden_states.dtype)  # [B, S, 1]
                
                # Skip if no tokens selected this expert (quick check)
                if mask.sum() == 0:
                    continue
                
                # Compute expert output for all tokens (simpler, more GPU-friendly)
                expert_output = self.experts[e](hidden_states)  # [B, S, D]
                
                # Add weighted contribution only where selected
                final_output = final_output + expert_output * expert_weights * mask
        
        return final_output


class MoELayerOptimized(nn.Module):
    """
    Optimized MoE implementation using batched operations.
    
    More efficient for GPU by avoiding loops over experts.
    
    Same design: each expert = original/(1+top_k), so activated compute = 1x original.
    """
    
    def __init__(
        self,
        hidden_size: int,
        expansion: float,
        num_experts: int = 4,
        top_k: int = 2,
        shared_expert: bool = True,
        jitter_noise: float = 0.01,
        aux_loss_coef: float = 0.01,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.shared_expert_enabled = shared_expert
        self.aux_loss_coef = aux_loss_coef
        
        # Compute original FFN intermediate size
        original_inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        
        # All experts same size = original / (1 + top_k)
        num_activated = (1 if shared_expert else 0) + top_k
        expert_inter = max(16, original_inter // num_activated)  # Allow small experts for small models
        
        self.intermediate_size = expert_inter
        
        # Shared expert (same size as routed)
        if shared_expert:
            self.shared_expert = Expert(hidden_size, expert_inter)
        
        # Batched expert weights (all same size)
        self.expert_gate_up = nn.Parameter(
            torch.randn(num_experts, hidden_size, expert_inter * 2) * (1.0 / hidden_size ** 0.5)
        )
        self.expert_down = nn.Parameter(
            torch.randn(num_experts, expert_inter, hidden_size) * (1.0 / expert_inter ** 0.5)
        )
        
        # Router
        self.router = Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            jitter_noise=jitter_noise,
        )
        
        self._aux_loss = None
    
    @property
    def aux_loss(self) -> Optional[torch.Tensor]:
        return self._aux_loss
    
    def _compute_aux_loss(self, router_logits, selected_experts):
        batch_size, seq_len, _ = router_logits.shape
        num_tokens = batch_size * seq_len
        
        router_probs = F.softmax(router_logits, dim=-1)
        avg_probs = router_probs.mean(dim=(0, 1))
        
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts)
        expert_counts = expert_mask.sum(dim=2).float()
        expert_fractions = expert_counts.sum(dim=(0, 1)) / (num_tokens * self.top_k)
        
        aux_loss = self.num_experts * (avg_probs * expert_fractions).sum()
        return aux_loss
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Shared expert
        if self.shared_expert_enabled:
            output = self.shared_expert(hidden_states)
        else:
            output = torch.zeros_like(hidden_states)
        
        # Routing
        router_probs, selected_experts, router_logits = self.router(hidden_states)
        
        if self.training:
            self._aux_loss = self._compute_aux_loss(router_logits, selected_experts)
        else:
            self._aux_loss = None
        
        # Flatten for batched expert computation
        x_flat = hidden_states.view(-1, hidden_dim)  # [B*S, D]
        output_flat = output.view(-1, hidden_dim)  # [B*S, D]
        
        selected_flat = selected_experts.view(-1, self.top_k)  # [B*S, K]
        weights_flat = router_probs.view(-1, self.top_k)  # [B*S, K]
        
        # Process each top-k slot
        for k in range(self.top_k):
            expert_indices = selected_flat[:, k]  # [B*S]
            weights = weights_flat[:, k:k+1]  # [B*S, 1]
            
            # Gather expert weights for selected experts (cast to input dtype)
            # expert_gate_up: [E, D, inter*2]
            gate_up_weights = self.expert_gate_up[expert_indices].to(x_flat.dtype)  # [B*S, D, inter*2]
            down_weights = self.expert_down[expert_indices].to(x_flat.dtype)  # [B*S, inter, D]
            
            # Expert computation: SwiGLU
            gate_up = torch.bmm(x_flat.unsqueeze(1), gate_up_weights).squeeze(1)  # [B*S, inter*2]
            gate, up = gate_up.chunk(2, dim=-1)
            expert_out = torch.bmm(
                (F.silu(gate) * up).unsqueeze(1), down_weights
            ).squeeze(1)  # [B*S, D]
            
            output_flat = output_flat + expert_out * weights
        
        return output_flat.view(batch_size, seq_len, hidden_dim)

