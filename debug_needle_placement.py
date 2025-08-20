#!/usr/bin/env python3
"""
Debug script to examine needle placement patterns in length 256 sequences.
"""

import torch
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import Counter

# Import modules
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


def debug_needle_placement():
    """Debug needle placement patterns in length 256 sequences."""
    print("Creating test dataset...")
    test_dataset = create_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)
    
    # Collect length 256 samples
    length_256_data = []
    
    print("Analyzing needle placement in length 256 sequences...")
    for batch_idx, batch in enumerate(test_loader):
        X, Y, lengths = batch
        
        # Find length 256 samples in this batch
        for i in range(X.size(0)):
            if lengths[i] == 256:
                sequence = X[i].tolist()
                needle_pos = Y[i].item()
                
                # Find needle token (127) in sequence
                needle_token = 127
                actual_needle_positions = [j for j, token in enumerate(sequence) if token == needle_token]
                
                length_256_data.append({
                    'batch_idx': batch_idx,
                    'sample_idx': i,
                    'target_pos': needle_pos,
                    'actual_needle_positions': actual_needle_positions,
                    'sequence_length': len(sequence),
                    'needle_count': len(actual_needle_positions),
                    'first_10_tokens': sequence[:10],
                    'last_10_tokens': sequence[-10:],
                    'token_at_target': sequence[needle_pos] if needle_pos < len(sequence) else 'OUT_OF_RANGE'
                })
    
    print(f"\nFound {len(length_256_data)} length-256 samples")
    
    # Analyze needle placement patterns
    target_positions = [d['target_pos'] for d in length_256_data]
    needle_counts = [d['needle_count'] for d in length_256_data]
    
    print(f"\nNeedle position statistics:")
    print(f"  Min target position: {min(target_positions)}")
    print(f"  Max target position: {max(target_positions)}")
    print(f"  Mean target position: {np.mean(target_positions):.1f}")
    print(f"  Unique target positions: {len(set(target_positions))}")
    
    print(f"\nNeedle count statistics:")
    print(f"  Min needle count: {min(needle_counts)}")
    print(f"  Max needle count: {max(needle_counts)}")
    print(f"  Needle count distribution: {Counter(needle_counts)}")
    
    # Check for suspicious patterns
    print(f"\nFirst 10 samples:")
    for i, data in enumerate(length_256_data[:10]):
        print(f"  Sample {i+1}:")
        print(f"    Target pos: {data['target_pos']}")
        print(f"    Actual needle positions: {data['actual_needle_positions']}")
        print(f"    Needle count: {data['needle_count']}")
        print(f"    Token at target: {data['token_at_target']}")
        print(f"    First 10 tokens: {data['first_10_tokens']}")
        print()
    
    # Check for data leakage patterns
    issues = []
    
    # Check if needle is always at the same position
    if len(set(target_positions)) == 1:
        issues.append(f"All needles at same position: {target_positions[0]}")
    
    # Check if there are multiple needles per sequence
    multi_needle_count = sum(1 for count in needle_counts if count > 1)
    if multi_needle_count > 0:
        issues.append(f"{multi_needle_count} sequences have multiple needles")
    
    # Check if needle is always at target position
    correct_placements = sum(1 for d in length_256_data 
                           if d['target_pos'] in d['actual_needle_positions'])
    if correct_placements == len(length_256_data):
        issues.append("All needles are correctly placed at target positions")
    
    # Check for position patterns
    position_counter = Counter(target_positions)
    most_common_pos = position_counter.most_common(5)
    print(f"\nMost common target positions:")
    for pos, count in most_common_pos:
        percentage = (count / len(target_positions)) * 100
        print(f"  Position {pos}: {count} times ({percentage:.1f}%)")
    
    if issues:
        print(f"\n🚨 POTENTIAL ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print(f"\n✅ No obvious data leakage patterns detected")
    
    return length_256_data


if __name__ == "__main__":
    debug_needle_placement()
