"""
Evaluation script for TitanClassifier on NIAH test set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import argparse
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from .data.niah import NIAHDataset, collate
from .models.titan_min import TitanClassifier
from .models.heads import position_logits


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate TitanClassifier on NIAH test set")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    return parser.parse_args()


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
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, config


def get_position_bin(needle_pos, seq_len):
    """Get position bin (0-20%, 20-40%, etc.) for a needle position."""
    percentage = (needle_pos / seq_len) * 100
    
    if percentage < 20:
        return "0-20%"
    elif percentage < 40:
        return "20-40%"
    elif percentage < 60:
        return "40-60%"
    elif percentage < 80:
        return "60-80%"
    else:
        return "80-100%"


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    
    # Overall metrics
    total_correct = 0
    total_samples = 0
    
    # Metrics by sequence length
    acc_by_length = {64: {'correct': 0, 'total': 0},
                     128: {'correct': 0, 'total': 0},
                     256: {'correct': 0, 'total': 0}}
    
    # Metrics by position bin
    acc_by_pos_bin = {"0-20%": {'correct': 0, 'total': 0},
                      "20-40%": {'correct': 0, 'total': 0},
                      "40-60%": {'correct': 0, 'total': 0},
                      "60-80%": {'correct': 0, 'total': 0},
                      "80-100%": {'correct': 0, 'total': 0}}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            X, Y, lengths = batch
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            
            # Forward pass
            h_tokens_out, rep = model(X)
            
            # Compute position logits
            logits = position_logits(rep, h_tokens_out, lengths)
            
            # Get predictions
            predictions = logits.argmax(dim=-1)
            
            # Update metrics for each sample in batch
            for i in range(X.size(0)):
                pred = predictions[i].item()
                target = Y[i].item()
                seq_len = lengths[i].item()
                
                # Overall accuracy
                is_correct = (pred == target)
                total_correct += is_correct
                total_samples += 1
                
                # Accuracy by sequence length
                if seq_len in acc_by_length:
                    acc_by_length[seq_len]['correct'] += is_correct
                    acc_by_length[seq_len]['total'] += 1
                
                # Accuracy by position bin
                pos_bin = get_position_bin(target, seq_len)
                acc_by_pos_bin[pos_bin]['correct'] += is_correct
                acc_by_pos_bin[pos_bin]['total'] += 1
    
    # Calculate final accuracies
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
    
    # Calculate accuracy by length
    acc_by_L = {}
    for length, stats in acc_by_length.items():
        if stats['total'] > 0:
            acc_by_L[length] = stats['correct'] / stats['total']
        else:
            acc_by_L[length] = 0.0
    
    # Calculate accuracy by position bin
    final_acc_by_pos_bin = {}
    for bin_name, stats in acc_by_pos_bin.items():
        if stats['total'] > 0:
            final_acc_by_pos_bin[bin_name] = stats['correct'] / stats['total']
        else:
            final_acc_by_pos_bin[bin_name] = 0.0
    
    return {
        'overall_acc': overall_acc,
        'acc_by_L': acc_by_L,
        'acc_by_pos_bin': final_acc_by_pos_bin,
        'total_samples': total_samples
    }


def save_report(results, checkpoint_path, output_path):
    """Save evaluation report to JSON file."""
    report = {
        'checkpoint_path': str(checkpoint_path),
        'overall_acc': results['overall_acc'],
        'acc_by_L': {
            '64': results['acc_by_L'][64],
            '128': results['acc_by_L'][128],
            '256': results['acc_by_L'][256]
        },
        'acc_by_pos_bin': results['acc_by_pos_bin'],
        'total_samples': results['total_samples']
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Report saved to: {output_path}")


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model
        model, config = load_model_from_checkpoint(args.ckpt)
        model.to(device)
        
        # Create test dataset (same splits as training)
        print("Creating test dataset...")
        test_dataset = create_test_dataset(config['seed'])
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,  # Use reasonable batch size for evaluation
            shuffle=False,
            collate_fn=collate
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
        # Evaluate model
        print("\nEvaluating model on test set...")
        results = evaluate_model(model, test_loader, device)
        
        # Print results
        print(f"\n=== Evaluation Results ===")
        print(f"Overall Accuracy: {results['overall_acc']:.4f}")
        print(f"\nAccuracy by Sequence Length:")
        for length, acc in results['acc_by_L'].items():
            print(f"  Length {length}: {acc:.4f}")
        print(f"\nAccuracy by Position Bin:")
        for bin_name, acc in results['acc_by_pos_bin'].items():
            print(f"  {bin_name}: {acc:.4f}")
        print(f"\nTotal samples evaluated: {results['total_samples']}")
        
        # Save report
        checkpoint_path = Path(args.ckpt)
        report_path = checkpoint_path.parent / "report.json"
        save_report(results, checkpoint_path, report_path)
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
