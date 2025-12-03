"""
Q-head and V-head implementations for hypothesis-driven self-correction.

Q(s,a): Per-cell action-value estimation
- Controls gating decisions (LOCK/HYPOTHESIS/REJECT)
- Trained on discounted returns with self-consistency rewards

V(s): State-value estimation  
- Provides baseline for advantage computation
- Aggregates over cells for variance reduction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class QHead(nn.Module):
    """
    Per-cell action-value head Q(s,a).
    
    Takes hidden state z_H and outputs Q values for each cell.
    Q is used for:
    1. Gating decisions (high Q -> LOCK, low Q -> REJECT)
    2. Computing advantages A = Q - V
    
    Output range: tanh activation -> [-1, 1]
    Interpretation:
    - Q ≈ +1: High confidence, correct prediction likely
    - Q ≈ 0: Uncertain, keep as hypothesis
    - Q ≈ -1: Low confidence, prediction likely wrong
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size // 2
            
        self.hidden_size = hidden_size
        
        # Two-layer MLP with tanh output
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(intermediate_size)
        
        # Initialize to output near-zero initially (uncertain)
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        # Standard init for fc1
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        
        # Small init for fc2 to start with near-zero Q values
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_H: Hidden state [batch_size, seq_len, hidden_size]
            
        Returns:
            Q values [batch_size, seq_len] in range [-1, 1]
        """
        # Cast to float32 for stable computation
        z_H = z_H.float()
        
        # MLP: hidden -> intermediate -> 1
        x = self.fc1(z_H)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Tanh to bound output to [-1, 1]
        q_values = torch.tanh(x.squeeze(-1))
        
        return q_values


class VHead(nn.Module):
    """
    State-value head V(s).
    
    Estimates expected return from current state, aggregated over all cells.
    Used as baseline in advantage computation: A = Q - V
    
    This reduces variance in policy gradient estimation.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        aggregation: str = 'mean',  # 'mean', 'attention', or 'cls'
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size // 2
            
        self.hidden_size = hidden_size
        self.aggregation = aggregation
        
        # Optional attention pooling
        if aggregation == 'attention':
            self.attention = nn.Linear(hidden_size, 1)
            
        # Value network
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(intermediate_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)
        
        if self.aggregation == 'attention':
            nn.init.xavier_uniform_(self.attention.weight)
            nn.init.zeros_(self.attention.bias)
        
    def forward(
        self, 
        z_H: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            z_H: Hidden state [batch_size, seq_len, hidden_size]
            mask: Optional mask for valid cells [batch_size, seq_len]
            
        Returns:
            V value [batch_size] - scalar per batch element
        """
        # Cast to float32 for stable computation
        z_H = z_H.float()
        
        batch_size, seq_len, _ = z_H.shape
        
        # Aggregate over sequence
        if self.aggregation == 'mean':
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1).float()
                pooled = (z_H * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = z_H.mean(dim=1)
                
        elif self.aggregation == 'attention':
            # Attention pooling
            attn_weights = self.attention(z_H).squeeze(-1)  # [B, L]
            if mask is not None:
                attn_weights = attn_weights.masked_fill(~mask, float('-inf'))
            attn_weights = F.softmax(attn_weights, dim=-1)  # [B, L]
            pooled = torch.einsum('bl,bld->bd', attn_weights, z_H)  # [B, D]
            
        elif self.aggregation == 'cls':
            # Use first token (like BERT [CLS])
            pooled = z_H[:, 0]
            
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
            
        # MLP to scalar value
        x = self.fc1(pooled)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        v_value = self.fc2(x).squeeze(-1)  # [B]
        
        return v_value


class DuelingHead(nn.Module):
    """
    Dueling architecture combining Q and V heads.
    
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    
    This decomposition helps learn state values and advantages separately,
    which can improve learning in cases where the value doesn't depend much
    on the action.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size // 2
            
        self.hidden_size = hidden_size
        
        # Shared feature layer
        self.shared_fc = nn.Linear(hidden_size, intermediate_size)
        self.shared_norm = nn.LayerNorm(intermediate_size)
        self.dropout = nn.Dropout(dropout)
        
        # Value stream (scalar per sequence)
        self.v_fc = nn.Linear(intermediate_size, 1)
        
        # Advantage stream (per cell)
        self.a_fc = nn.Linear(intermediate_size, 1)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training."""
        nn.init.xavier_uniform_(self.shared_fc.weight)
        nn.init.zeros_(self.shared_fc.bias)
        nn.init.xavier_uniform_(self.v_fc.weight, gain=0.1)
        nn.init.zeros_(self.v_fc.bias)
        nn.init.xavier_uniform_(self.a_fc.weight, gain=0.1)
        nn.init.zeros_(self.a_fc.bias)
        
    def forward(
        self, 
        z_H: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_H: Hidden state [batch_size, seq_len, hidden_size]
            mask: Optional mask for valid cells [batch_size, seq_len]
            
        Returns:
            Q: Action values [batch_size, seq_len]
            V: State value [batch_size]
            A: Advantages [batch_size, seq_len]
        """
        # Cast to float32 for stable computation
        z_H = z_H.float()
        
        batch_size, seq_len, _ = z_H.shape
        
        # Shared features
        shared = self.shared_fc(z_H)
        shared = self.shared_norm(shared)
        shared = F.gelu(shared)
        shared = self.dropout(shared)
        
        # Value stream: aggregate over sequence
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (shared * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        else:
            pooled = shared.mean(dim=1)
        V = self.v_fc(pooled).squeeze(-1)  # [B]
        
        # Advantage stream: per cell
        A = self.a_fc(shared).squeeze(-1)  # [B, L]
        
        # Center advantages (mean-subtraction for identifiability)
        if mask is not None:
            A_masked = A.masked_fill(~mask, 0)
            A_mean = A_masked.sum(dim=-1, keepdim=True) / mask.sum(dim=-1, keepdim=True).clamp(min=1)
        else:
            A_mean = A.mean(dim=-1, keepdim=True)
        A_centered = A - A_mean
        
        # Q = V + A (dueling combination)
        Q = V.unsqueeze(-1) + A_centered
        
        # Apply tanh to bound Q to [-1, 1]
        Q = torch.tanh(Q)
        
        return Q, V, A_centered


class ConfidenceHead(nn.Module):
    """
    Alternative confidence head that outputs calibrated probability.
    
    Instead of tanh, outputs sigmoid for P(correct | prediction).
    This is useful when we want Q to represent actual probability of correctness.
    """
    
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        if intermediate_size is None:
            intermediate_size = hidden_size // 2
            
        self.hidden_size = hidden_size
        
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(intermediate_size)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize to output ~0.5 initially (maximum uncertainty)."""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)
        
    def forward(self, z_H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_H: Hidden state [batch_size, seq_len, hidden_size]
            
        Returns:
            Confidence [batch_size, seq_len] in range [0, 1]
        """
        # Cast to float32 for stable computation
        z_H = z_H.float()
        
        x = self.fc1(z_H)
        x = self.layer_norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        # Sigmoid for probability output
        confidence = torch.sigmoid(x.squeeze(-1))
        
        return confidence

