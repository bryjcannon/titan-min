"""
Titan model implementation with DepthwiseSeparable1D, TitanBlock, MemoryPrefix, and TitanClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple
from .titan_memory import TitanLongTermMemory


class DepthwiseSeparable1D(nn.Module):
    """Depthwise separable 1D convolution."""
    
    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        
        # Depthwise convolution (groups=dim)
        self.depthwise = nn.Conv1d(
            dim, dim, 
            kernel_size=kernel_size, 
            padding=padding, 
            groups=dim
        )
        
        # Pointwise convolution (1x1)
        self.pointwise = nn.Conv1d(dim, dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: [B, L, C] -> [B, L, C]"""
        # Convert to conv1d format: [B, C, L]
        x = x.transpose(1, 2)
        
        # Apply depthwise then pointwise
        x = self.depthwise(x)
        x = self.pointwise(x)
        
        # Convert back to [B, L, C]
        x = x.transpose(1, 2)
        return x


class TitanBlock(nn.Module):
    """Titan transformer block with depthwise separable convolutions."""
    
    def __init__(self, dim: int, n_heads: int, conv_ks: int = 3, 
                 no_dsconv: bool = False, no_l2: bool = False, activation: str = 'silu'):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.no_dsconv = no_dsconv
        self.no_l2 = no_l2
        self.head_dim = dim // n_heads
        
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        # Store activation function
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'silu':
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Q, K, V projections
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        
        # Depthwise separable convolutions after Q, K, V (conditional)
        if not no_dsconv:
            self.q_conv = DepthwiseSeparable1D(dim, conv_ks)
            self.k_conv = DepthwiseSeparable1D(dim, conv_ks)
            self.v_conv = DepthwiseSeparable1D(dim, conv_ks)
        else:
            self.q_conv = nn.Identity()
            self.k_conv = nn.Identity()
            self.v_conv = nn.Identity()
        
        # Modern gating mechanism (Mehta et al. 2023 style)
        self.gate_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, dim * 2, bias=False)  # Projects to 2x for gating
        self.gate_activation = nn.SiLU()  # SwiGLU-style activation
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Temperature parameter for cosine attention
        self.temperature = nn.Parameter(torch.ones(1) * math.sqrt(self.head_dim))
        
        # Final layer norm for stability
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: [B, L, C]"""
        B, L, C = x.shape
        residual = x
        
        # Q, K, V projections with SiLU activation as specified in Titan paper
        q = self.activation(self.q_proj(x))  # [B, L, C] - SiLU activation
        k = self.activation(self.k_proj(x))  # [B, L, C] - SiLU activation
        v = self.activation(self.v_proj(x))  # [B, L, C] - SiLU activation
        
        # Apply depthwise separable convolutions (or identity if disabled)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        
        # Reshape for multi-head attention FIRST
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L, head_dim]
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L, head_dim]
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L, head_dim]
        
        # Per-head L2 normalization for cosine attention (conditional)
        if not self.no_l2:
            q = F.normalize(q, p=2, dim=-1)  # [B, n_heads, L, head_dim]
            k = F.normalize(k, p=2, dim=-1)  # [B, n_heads, L, head_dim]
        
        # Cosine attention scores with temperature scaling
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.temperature  # [B, n_heads, L, L]
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, L, head_dim]
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)  # [B, L, C]
        
        # Residual connection around attention
        attn_out = attn_out + residual
        
        # Modern gating mechanism: LayerNorm + SwiGLU-style gating before output projection
        normalized = self.gate_norm(attn_out)
        gate_proj = self.gate_proj(normalized)  # [B, L, 2*C]
        
        # Split into value and gate components
        value, gate = gate_proj.chunk(2, dim=-1)  # Each [B, L, C]
        
        # SwiGLU-style gating: value * SiLU(gate)
        gated = value * self.gate_activation(gate)
        
        # Final output projection
        out = self.out_proj(gated)
        
        # Final residual connection with LayerNorm for stability
        out = self.final_norm(out + attn_out)
        
        return out


# Old MemoryPrefix removed - replaced with TitanLongTermMemory


class TitanClassifier(nn.Module):
    """Titan classifier model."""
    
    def __init__(
        self, 
        vocab_size: int = 128, 
        dim: int = 256, 
        n_heads: int = 8, 
        n_layers: int = 2, 
        n_mem: int = 4,
        no_memory: bool = False,
        no_dsconv: bool = False,
        no_l2: bool = False,
        activation: str = 'silu',
        surprise_threshold: float = 0.8
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.n_mem = n_mem if not no_memory else 0
        self.no_memory = no_memory
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, dim)
        
        # Titan Long-Term Memory (conditional)
        if not no_memory:
            self.long_term_memory = TitanLongTermMemory(
                dim=dim,
                memory_dim=min(128, dim // 2),  # Adaptive memory dimension
                segment_size=64,  # Process in 64-token segments
                momentum=0.9,
                surprise_threshold=surprise_threshold
            )
        else:
            self.long_term_memory = None
        
        # Transformer blocks with ablation flags
        self.blocks = nn.ModuleList([
            TitanBlock(dim, n_heads, no_dsconv=no_dsconv, no_l2=no_l2, activation=activation) 
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with Titan Long-Term Memory.
        
        Args:
            x: Input token IDs [B, L]
            
        Returns:
            h_tokens_out: Memory-enhanced token states [B, L, C]
            rep: Summary vector [B, C] (global pooling of memory-enhanced states)
        """
        B, L = x.shape
        
        # Embedding
        h = self.embedding(x)  # [B, L, C]
        
        # Apply long-term memory if enabled
        if self.long_term_memory is not None:
            h = self.long_term_memory(h)  # [B, L, C] - memory-enhanced
        
        # Pass through transformer blocks
        for block in self.blocks:
            h = block(h)  # [B, L, C]
        
        # Final layer norm
        h = self.final_norm(h)  # [B, L, C]
        
        # Token representations are the full sequence
        h_tokens_out = h  # [B, L, C]
        
        # Summary representation: attention-weighted pooling
        # This gives more weight to important positions for needle detection
        attention_weights = torch.softmax(
            torch.norm(h, dim=-1), dim=-1
        ).unsqueeze(-1)  # [B, L, 1]
        
        rep = (h * attention_weights).sum(dim=1)  # [B, C]
        
        return h_tokens_out, rep
