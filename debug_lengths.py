#!/usr/bin/env python3
"""
Debug script to check sequence length distribution in test set.
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter

# Import the dataset
import sys
sys.path.append('src')
from data.niah import NIAHDataset, collate


def create_test_dataset(seed=42):
    """Create test split of NIAH dataset (same as training script)."""
    # Create full dataset with same parameters as training
    dataset = NIAHDataset(n_samples=10000, seed=seed)
    
    # Create splits with fixed generator seed (same as training)
    generator = torch.Generator()
    generator.manual_seed(42)
    
    # Split indices: 8000/1000/1000 (same as training)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    test_indices = indices[9000:]  # Last 1000 samples
    
    # Create test dataset
    test_dataset = Subset(dataset, test_indices)
    
    return test_dataset


def analyze_test_lengths():
    """Analyze sequence length distribution in test set."""
    print("Creating test dataset...")
    test_dataset = create_test_dataset()
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)
    
    # Collect all lengths
    all_lengths = []
    length_counts = Counter()
    
    print("Analyzing lengths...")
    for batch in test_loader:
        X, Y, lengths = batch
        batch_lengths = lengths.tolist()
        all_lengths.extend(batch_lengths)
        
        for length in batch_lengths:
            length_counts[length] += 1
    
    print(f"\nTotal samples analyzed: {len(all_lengths)}")
    print(f"Unique lengths: {sorted(set(all_lengths))}")
    print(f"\nLength distribution:")
    for length in sorted(length_counts.keys()):
        count = length_counts[length]
        percentage = (count / len(all_lengths)) * 100
        print(f"  Length {length}: {count} samples ({percentage:.1f}%)")
    
    # Check for potential issues
    if 256 in length_counts:
        count_256 = length_counts[256]
        print(f"\n🔍 Length 256 analysis:")
        print(f"  Count: {count_256}")
        print(f"  Percentage: {(count_256 / len(all_lengths)) * 100:.2f}%")
        
        if count_256 == 0:
            print("  ❌ ISSUE: No length 256 sequences found!")
        elif count_256 < 10:
            print("  ⚠️  WARNING: Very few length 256 sequences - might cause statistical issues")
    else:
        print(f"\n❌ CRITICAL: Length 256 not found in test set!")
    
    return length_counts


if __name__ == "__main__":
    analyze_test_lengths()
