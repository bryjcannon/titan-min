"""
Model heads for various tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def position_logits(rep: torch.Tensor, token_states: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Compute position logits for needle position prediction.
    
    Args:
        rep: Summary representation [B, C]
        token_states: Token states [B, L, C]
        lengths: Sequence lengths [B] - positions >= lengths will be masked
        
    Returns:
        logits: Position logits [B, L] with positions >= lengths masked to -inf
    """
    # Compute logits using einsum: [B, C] x [B, L, C] -> [B, L]
    logits = torch.einsum('bc,blc->bl', rep, token_states)
    
    # Create mask for positions >= lengths
    B, L = logits.shape
    positions = torch.arange(L, device=logits.device).unsqueeze(0).expand(B, -1)  # [B, L]
    mask = positions >= lengths.unsqueeze(1)  # [B, L]
    
    # Mask invalid positions with -inf
    logits = logits.masked_fill(mask, float('-inf'))
    
    return logits


class LanguageModelingHead(nn.Module):
    """Language modeling head for next token prediction."""
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class ClassificationHead(nn.Module):
    """Classification head for sequence classification."""
    
    def __init__(self, d_model: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, hidden_states: torch.Tensor, pooling: str = 'mean') -> torch.Tensor:
        # Pool sequence representations
        if pooling == 'mean':
            pooled = hidden_states.mean(dim=1)
        elif pooling == 'max':
            pooled = hidden_states.max(dim=1)[0]
        elif pooling == 'cls':
            pooled = hidden_states[:, 0]  # Use first token
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


class NIAHHead(nn.Module):
    """Needle in a Haystack detection head."""
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.position_head = nn.Linear(d_model, 1)  # Predict needle position
        self.detection_head = nn.Linear(d_model, 2)  # Binary classification: needle present/absent
        
    def forward(self, hidden_states: torch.Tensor) -> dict:
        """
        Forward pass for NIAH task.
        
        Args:
            hidden_states: [batch_size, seq_len, d_model]
            
        Returns:
            Dictionary with position logits and detection logits
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Apply dropout
        hidden_states = self.dropout(hidden_states)
        
        # Position prediction: score each position
        position_logits = self.position_head(hidden_states).squeeze(-1)  # [batch_size, seq_len]
        
        # Detection: pool and classify
        pooled = hidden_states.mean(dim=1)  # [batch_size, d_model]
        detection_logits = self.detection_head(pooled)  # [batch_size, 2]
        
        return {
            'position_logits': position_logits,
            'detection_logits': detection_logits
        }


class RegressionHead(nn.Module):
    """Regression head for continuous value prediction."""
    
    def __init__(self, d_model: int, output_dim: int = 1, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(d_model, output_dim)
        
    def forward(self, hidden_states: torch.Tensor, pooling: str = 'mean') -> torch.Tensor:
        # Pool sequence representations
        if pooling == 'mean':
            pooled = hidden_states.mean(dim=1)
        elif pooling == 'max':
            pooled = hidden_states.max(dim=1)[0]
        elif pooling == 'cls':
            pooled = hidden_states[:, 0]
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
            
        pooled = self.dropout(pooled)
        return self.regressor(pooled)


class MultiTaskHead(nn.Module):
    """Multi-task head combining multiple objectives."""
    
    def __init__(
        self, 
        d_model: int, 
        vocab_size: Optional[int] = None,
        num_classes: Optional[int] = None,
        output_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.heads = nn.ModuleDict()
        
        if vocab_size is not None:
            self.heads['lm'] = LanguageModelingHead(d_model, vocab_size)
            
        if num_classes is not None:
            self.heads['classification'] = ClassificationHead(d_model, num_classes, dropout)
            
        if output_dim is not None:
            self.heads['regression'] = RegressionHead(d_model, output_dim, dropout)
            
        # Always include NIAH head
        self.heads['niah'] = NIAHHead(d_model, dropout)
        
    def forward(self, hidden_states: torch.Tensor, task: str = 'niah', **kwargs) -> torch.Tensor:
        """
        Forward pass for specified task.
        
        Args:
            hidden_states: Model hidden states
            task: Task name ('lm', 'classification', 'regression', 'niah')
            **kwargs: Additional arguments for specific heads
            
        Returns:
            Task-specific output
        """
        if task not in self.heads:
            raise ValueError(f"Task '{task}' not available. Available tasks: {list(self.heads.keys())}")
            
        return self.heads[task](hidden_states, **kwargs)
