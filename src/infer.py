"""
Inference script for TitanClassifier needle finding.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import csv

from .models.titan_min import TitanClassifier
from .models.heads import position_logits
from .utils.checkpoint import load_checkpoint


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run inference on TitanClassifier for needle finding")
    
    # Model loading arguments (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--ckpt", type=str, help="Path to checkpoint file")
    model_group.add_argument("--script", type=str, help="Path to TorchScript model file")
    
    # Input arguments
    parser.add_argument("--seq", type=str, required=True, help="Comma-separated sequence of integers")
    parser.add_argument("--save_csv", type=str, help="Optional path to save CSV results")
    
    return parser.parse_args()


def parse_sequence(seq_str):
    """Parse comma-separated sequence string into tensor.
    
    Args:
        seq_str: Comma-separated string of integers
        
    Returns:
        tuple: (X tensor [1, L], lengths tensor [1])
    """
    try:
        # Parse comma-separated integers
        tokens = [int(x.strip()) for x in seq_str.split(',')]
        
        # Convert to tensor and add batch dimension
        X = torch.tensor(tokens).unsqueeze(0)  # [1, L]
        lengths = torch.tensor([len(tokens)])  # [1]
        
        return X, lengths
    except ValueError as e:
        raise ValueError(f"Invalid sequence format. Expected comma-separated integers, got: {seq_str}") from e


def load_checkpoint_model(ckpt_path):
    """Load model from checkpoint file.
    
    Args:
        ckpt_path: Path to checkpoint file
        
    Returns:
        Model ready for inference
    """
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
    
    return model


def load_torchscript_model(script_path):
    """Load TorchScript model.
    
    Args:
        script_path: Path to TorchScript model file
        
    Returns:
        TorchScript model ready for inference
    """
    script_path = Path(script_path)
    
    if not script_path.exists():
        raise FileNotFoundError(f"TorchScript model not found: {script_path}")
    
    print(f"Loading TorchScript model: {script_path}")
    
    model = torch.jit.load(str(script_path), map_location='cpu')
    model.eval()
    
    print("TorchScript model loaded successfully")
    
    return model


def run_inference(model, X, lengths, is_torchscript=False):
    """Run inference on the model.
    
    Args:
        model: Model (either TitanClassifier or TorchScript)
        X: Input tensor [1, L]
        lengths: Lengths tensor [1]
        is_torchscript: Whether model is TorchScript
        
    Returns:
        Predicted needle index (int)
    """
    with torch.no_grad():
        if is_torchscript:
            # TorchScript model (NeedleFinder) directly returns logits
            logits = model(X, lengths)
        else:
            # Regular TitanClassifier model
            h_tokens_out, rep = model(X)
            logits = position_logits(rep, h_tokens_out, lengths)
        
        # Get predicted index via argmax
        predicted_index = logits.argmax(dim=-1).item()
        
    return predicted_index


def save_to_csv(csv_path, sequence, predicted_index):
    """Save results to CSV file.
    
    Args:
        csv_path: Path to save CSV file
        sequence: Original sequence string
        predicted_index: Predicted needle index
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'predicted_index'])
        writer.writerow([sequence, predicted_index])
    
    print(f"Results saved to CSV: {csv_path}")


def main():
    args = parse_args()
    
    # Validate arguments
    if not args.ckpt and not args.script:
        raise ValueError("Must specify either --ckpt or --script")
    
    try:
        # Parse input sequence
        X, lengths = parse_sequence(args.seq)
        print(f"Input sequence length: {lengths.item()}")
        print(f"Input tokens: {X.squeeze().tolist()}")
        
        # Load model
        if args.ckpt:
            model = load_checkpoint_model(args.ckpt)
            is_torchscript = False
        else:
            model = load_torchscript_model(args.script)
            is_torchscript = True
        
        # Run inference
        predicted_index = run_inference(model, X, lengths, is_torchscript)
        
        # Print result
        print(f"Predicted needle index: {predicted_index}")
        
        # Save to CSV if requested
        if args.save_csv:
            save_to_csv(args.save_csv, args.seq, predicted_index)
        
    except Exception as e:
        print(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()
