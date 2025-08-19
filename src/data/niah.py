"""
Needle in a Haystack (NIAH) dataset implementation.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Union


class NIAHDataset(Dataset):
    """
    Needle in a Haystack dataset for testing long-context capabilities.
    """
    
    def __init__(
        self,
        n_samples: int = 10000,
        seq_lens: Tuple[int, ...] = (64, 128, 256),
        vocab_size: int = 128,
        seed: int = 0
    ):
        """
        Initialize NIAH dataset.
        
        Args:
            n_samples: Number of samples in dataset
            seq_lens: Tuple of possible sequence lengths
            vocab_size: Vocabulary size (tokens are in [1, vocab_size-2], 
                       vocab_size-1 is NEEDLE, 0 is PAD)
            seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.seq_lens = seq_lens
        self.vocab_size = vocab_size
        self.PAD = 0
        self.NEEDLE = vocab_size - 1
        
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Generate all samples
        self.samples = self._generate_samples()
    
    def _generate_samples(self) -> List[Tuple[torch.LongTensor, int]]:
        """Generate all dataset samples."""
        samples = []
        
        for _ in range(self.n_samples):
            # Choose random sequence length
            L = np.random.choice(self.seq_lens)
            
            # Generate sequence of random tokens in [1, vocab_size-2]
            sequence = torch.randint(1, self.vocab_size - 1, (L,))
            
            # Choose random needle position
            needle_pos = np.random.randint(0, L)
            
            # Place NEEDLE token at needle_pos
            sequence[needle_pos] = self.NEEDLE
            
            samples.append((sequence, needle_pos))
        
        return samples
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            x: LongTensor[L] - sequence with needle
            y: LongTensor[] - needle position (scalar)
        """
        x, y = self.samples[idx]
        return x, torch.tensor(y, dtype=torch.long)


def collate(batch: List[Tuple[torch.LongTensor, torch.LongTensor]]) -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
    """
    Collate function for NIAH dataset.
    Left-pads sequences to max length in batch with PAD=0.
    
    Args:
        batch: List of (x, y) tuples
        
    Returns:
        X: LongTensor[B, L] - padded sequences
        Y: LongTensor[B] - needle positions
        lengths: LongTensor[B] - original sequence lengths
    """
    sequences, targets = zip(*batch)
    
    # Get original lengths
    lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)
    
    # Find max length in batch
    max_len = max(len(seq) for seq in sequences)
    
    # Left-pad sequences
    padded_sequences = []
    for seq in sequences:
        pad_len = max_len - len(seq)
        if pad_len > 0:
            # Left-pad with PAD=0
            padded_seq = torch.cat([torch.zeros(pad_len, dtype=torch.long), seq])
        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)
    
    # Stack into batch tensors
    X = torch.stack(padded_sequences)
    Y = torch.stack(targets)
    
    return X, Y, lengths
