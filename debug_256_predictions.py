#!/usr/bin/env python3
"""
Debug script to examine length 256 predictions in detail.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path

# Import modules
import sys
sys.path.append('src')
from data.niah import NIAHDataset, collate
from models.titan_min import TitanClassifier
from models.heads import position_logits


def load_model_from_checkpoint(ckpt_path):
    """Load model from checkpoint file."""
    ckpt_path = Path(ckpt_path)
    
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    
    print(f"Loading model from checkpoint: {ckpt_path}")
    
    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = checkpoint['config']
    
    # Reconstruct model from config
    model = TitanClassifier(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        n_mem=config['n_mem'],
        no_memory=config.get('no_memory', False),
        no_dsconv=config.get('no_dsconv', False),
        no_l2=config.get('no_l2', False),
        activation=config.get('activation', 'silu')
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


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


def debug_256_predictions():
    """Debug length 256 predictions in detail."""
    print("Loading model...")
    model, config = load_model_from_checkpoint("artifacts/run1/best.ckpt")
    
    print("Creating test dataset...")
    test_dataset = create_test_dataset()
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)
    
    device = torch.device('cpu')
    model.to(device)
    
    # Collect length 256 samples
    length_256_samples = []
    
    print("Collecting length 256 samples...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            X, Y, lengths = batch
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            
            # Find length 256 samples in this batch
            length_256_mask = (lengths == 256)
            if length_256_mask.any():
                # Forward pass
                h_tokens_out, rep = model(X)
                
                # Compute position logits
                logits = position_logits(rep, h_tokens_out, lengths)
                
                # Get predictions
                predictions = logits.argmax(dim=-1)
                
                # Store length 256 samples
                for i in range(X.size(0)):
                    if lengths[i] == 256:
                        pred = predictions[i].item()
                        target = Y[i].item()
                        seq_len = lengths[i].item()
                        
                        # Get logits for this sample
                        sample_logits = logits[i].cpu().numpy()
                        
                        length_256_samples.append({
                            'batch_idx': batch_idx,
                            'sample_idx': i,
                            'prediction': pred,
                            'target': target,
                            'seq_len': seq_len,
                            'is_correct': pred == target,
                            'logits_shape': sample_logits.shape,
                            'max_logit_pos': np.argmax(sample_logits),
                            'logits_range': (np.min(sample_logits), np.max(sample_logits)),
                            'target_logit': sample_logits[target] if target < len(sample_logits) else 'OUT_OF_RANGE'
                        })
    
    print(f"\nFound {len(length_256_samples)} length-256 samples")
    
    # Analyze results
    correct_count = sum(1 for s in length_256_samples if s['is_correct'])
    accuracy = correct_count / len(length_256_samples) if length_256_samples else 0
    
    print(f"Length 256 accuracy: {correct_count}/{len(length_256_samples)} = {accuracy:.4f} ({accuracy*100:.1f}%)")
    
    # Show first few samples
    print(f"\nFirst 10 length-256 samples:")
    for i, sample in enumerate(length_256_samples[:10]):
        print(f"  Sample {i+1}:")
        print(f"    Prediction: {sample['prediction']}, Target: {sample['target']}, Correct: {sample['is_correct']}")
        print(f"    Logits shape: {sample['logits_shape']}")
        print(f"    Max logit pos: {sample['max_logit_pos']}")
        print(f"    Target logit: {sample['target_logit']}")
        print(f"    Logits range: {sample['logits_range']}")
        print()
    
    # Check for suspicious patterns
    all_predictions = [s['prediction'] for s in length_256_samples]
    all_targets = [s['target'] for s in length_256_samples]
    
    print(f"Prediction statistics:")
    print(f"  Min prediction: {min(all_predictions)}")
    print(f"  Max prediction: {max(all_predictions)}")
    print(f"  Unique predictions: {len(set(all_predictions))}")
    
    print(f"Target statistics:")
    print(f"  Min target: {min(all_targets)}")
    print(f"  Max target: {max(all_targets)}")
    print(f"  Unique targets: {len(set(all_targets))}")
    
    # Check if all predictions are the same
    if len(set(all_predictions)) == 1:
        print(f"🚨 ISSUE: All predictions are the same value: {all_predictions[0]}")
    
    # Check if predictions are always equal to targets (which would give 100%)
    if all(s['is_correct'] for s in length_256_samples):
        print(f"🚨 ISSUE: All length-256 predictions are correct - this suggests a bug!")
        
        # Check if targets are all the same
        if len(set(all_targets)) == 1:
            print(f"   All targets are the same: {all_targets[0]}")
        
        # Check if predictions are all the same
        if len(set(all_predictions)) == 1:
            print(f"   All predictions are the same: {all_predictions[0]}")


if __name__ == "__main__":
    debug_256_predictions()
