"""
Script to dump test predictions for detailed analysis.
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

from .data.niah import NIAHDataset, collate
from .models.titan_min import TitanClassifier
from .models.heads import position_logits


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dump test predictions to CSV")
    parser.add_argument("--ckpt", type=str, default=os.path.join("artifacts", "run1", "best.ckpt"), help="Path to checkpoint file")
    return parser.parse_args()


def create_test_dataset(seed=42):
    """Create test split of NIAH dataset (same as training and eval scripts)."""
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
    
    return test_dataset, test_indices


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


def dump_predictions(model, test_loader, test_indices, device, output_path):
    """Run inference on test set and dump predictions to CSV."""
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        batch_idx = 0
        for batch in tqdm(test_loader, desc="Generating predictions"):
            X, Y, lengths = batch
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            
            # Forward pass
            h_tokens_out, rep = model(X)
            
            # Compute position logits
            logits = position_logits(rep, h_tokens_out, lengths)
            
            # Get predictions
            preds = logits.argmax(dim=-1)
            
            # Store predictions for each sample in batch
            for i in range(X.size(0)):
                sample_idx = batch_idx * test_loader.batch_size + i
                idx_in_split = test_indices[sample_idx]  # Original dataset index
                
                predictions.append({
                    'idx_in_split': idx_in_split,
                    'length': lengths[i].item(),
                    'needle_pos_true': Y[i].item(),
                    'needle_pos_pred': preds[i].item()
                })
            
            batch_idx += 1
    
    # Write to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        fieldnames = ['idx_in_split', 'length', 'needle_pos_true', 'needle_pos_pred']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write predictions
        for pred in predictions:
            writer.writerow(pred)
    
    print(f"Predictions saved to: {output_path}")
    print(f"Total rows written: {len(predictions) + 1} (header + {len(predictions)} samples)")
    
    return len(predictions)


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Load model
        model, config = load_model_from_checkpoint(args.ckpt)
        model.to(device)
        
        # Create test dataset (same splits as training and eval)
        print("Creating test dataset...")
        test_dataset, test_indices = create_test_dataset(config['seed'])
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,  # Use reasonable batch size
            shuffle=False,  # Important: don't shuffle to maintain order
            collate_fn=collate
        )
        
        print(f"Test samples: {len(test_dataset)}")
        
        # Determine output path
        checkpoint_path = Path(args.ckpt)
        output_path = checkpoint_path.parent / "predictions.csv"
        
        # Dump predictions
        print("\nGenerating predictions...")
        num_samples = dump_predictions(model, test_loader, test_indices, device, output_path)
        
        # Verify output
        if output_path.exists():
            with open(output_path, 'r') as f:
                row_count = sum(1 for _ in f)
            
            print(f"\n=== Verification ===")
            print(f"CSV file exists: {output_path}")
            print(f"Total rows in CSV: {row_count} (expected: 1001)")
            print(f"Samples processed: {num_samples} (expected: 1000)")
            
            if row_count == 1001:
                print("✓ CSV format is correct (1001 rows = header + 1000 samples)")
            else:
                print(f"✗ CSV format is incorrect (expected 1001 rows, got {row_count})")
        else:
            print(f"✗ CSV file was not created at {output_path}")
        
    except Exception as e:
        print(f"Prediction dump failed: {e}")
        raise


if __name__ == "__main__":
    main()
