"""
Titan model implementation with DepthwiseSeparable1D, TitanBlock, MemoryPrefix, and TitanClassifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


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
        
        # Gate mechanism
        self.gate_norm = nn.LayerNorm(dim)
        self.gate_linear = nn.Linear(dim, dim)
        
        # Output projection
        self.out_proj = nn.Linear(dim, dim)
        
        # Final layer norm for stability
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: [B, L, C]"""
        B, L, C = x.shape
        residual = x
        
        # Q, K, V projections with configurable activation
        q = self.activation(self.q_proj(x))  # [B, L, C]
        k = self.activation(self.k_proj(x))  # [B, L, C]
        v = self.activation(self.v_proj(x))  # [B, L, C]
        
        # Apply depthwise separable convolutions (or identity if disabled)
        q = self.q_conv(q)
        k = self.k_conv(k)
        v = self.v_conv(v)
        
        # L2 normalize Q and K over last dimension (conditional)
        if not self.no_l2:
            q = F.normalize(q, p=2, dim=-1)
            k = F.normalize(k, p=2, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L, head_dim]
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L, head_dim]
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)  # [B, n_heads, L, head_dim]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, n_heads, L, L]
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)  # [B, n_heads, L, head_dim]
        
        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, L, C)  # [B, L, C]
        
        # Residual connection around attention
        attn_out = attn_out + residual
        
        # Gate mechanism: LayerNorm + linear gate (sigmoid) * normalized activations
        normalized = self.gate_norm(attn_out)
        gate = torch.sigmoid(self.gate_linear(normalized))
        gated = gate * normalized
        
        # Final output projection
        out = self.out_proj(gated)
        
        # Final residual connection with LayerNorm for stability
        out = self.final_norm(out + attn_out)
        
        return out


class MemoryPrefix(nn.Module):
    """Memory prefix module with EMA writeback."""
    
    def __init__(self, dim: int, n_slots: int = 4, beta: float = 0.1):
        super().__init__()
        self.dim = dim
        self.n_slots = n_slots
        self.beta = beta
        
        # Learnable memory parameters [1, n_slots, dim]
        self.mem = nn.Parameter(torch.randn(1, n_slots, dim) * 0.02)
        
        # MLP for EMA writeback
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.ReLU(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with EMA writeback. x: [B, L, C] -> [B, n_slots + L, C]"""
        B, L, C = x.shape
        
        # Expand memory to batch size
        mem_expanded = self.mem.expand(B, -1, -1)  # [B, n_slots, C]
        
        # Concatenate memory in front of tokens
        x_with_mem = torch.cat([mem_expanded, x], dim=1)  # [B, n_slots + L, C]
        
        # EMA writeback (no grad)
        if self.training:
            with torch.no_grad():
                # Compute mean over sequence dimension (excluding memory slots)
                x_mean = x.mean(dim=1)  # [B, C]
                
                # Apply MLP and take mean over batch
                mlp_out = self.mlp(x_mean)  # [B, C]
                batch_mean = mlp_out.mean(dim=0, keepdim=True).unsqueeze(0)  # [1, 1, C]
                
                # EMA update
                self.mem.data = (1 - self.beta) * self.mem.data + self.beta * batch_mean
        
        return x_with_mem


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
        activation: str = 'silu'
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
        
        # Memory prefix (conditional)
        if not no_memory:
            self.memory_prefix = MemoryPrefix(dim, n_mem)
        else:
            self.memory_prefix = None
        
        # Transformer blocks with ablation flags
        self.blocks = nn.ModuleList([
            TitanBlock(dim, n_heads, no_dsconv=no_dsconv, no_l2=no_l2, activation=activation) 
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input token IDs [B, L]
            
        Returns:
            h_tokens_out: States for original tokens only [B, L, C] (strip memory)
            rep: Summary vector [B, C] (use last position after memory as query-like summary)
        """
        B, L = x.shape
        
        # Embedding
        h = self.embedding(x)  # [B, L, C]
        
        # Add memory prefix (conditional)
        if not self.no_memory:
            h = self.memory_prefix(h)  # [B, n_mem + L, C]
        
        # Apply transformer blocks
        for block in self.blocks:
            h = block(h)  # [B, n_mem + L, C] or [B, L, C] if no memory
        
        # Final layer norm
        h = self.final_norm(h)  # [B, n_mem + L, C] or [B, L, C] if no memory
        
        # Strip memory slots to get original token states (or use all if no memory)
        if not self.no_memory:
            h_tokens_out = h[:, self.n_mem:]  # [B, L, C]
        else:
            h_tokens_out = h  # [B, L, C]
        
        # Use last position after memory as summary (query-like summary)
        rep = h[:, -1]  # [B, C] - last position
        
        return h_tokens_out, rep
