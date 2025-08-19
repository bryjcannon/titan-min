"""
Model export utilities for TitanClassifier deployment.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import json

from .models.titan_min import TitanClassifier
from .models.heads import position_logits
from .utils.checkpoint import load_checkpoint


class NeedleFinder(nn.Module):
    """Wrapper module for TitanClassifier that returns position logits."""
    
    def __init__(self, titan_classifier):
        super().__init__()
        self.titan_classifier = titan_classifier
    
    def forward(self, x, lengths):
        """Forward pass returning position logits.
        
        Args:
            x: Input token sequences [B, L]
            lengths: Sequence lengths [B]
            
        Returns:
            logits: Position logits [B, L]
        """
        h_tokens_out, rep = self.titan_classifier(x)
        logits = position_logits(rep, h_tokens_out, lengths)
        return logits


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Export TitanClassifier to TorchScript")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint file")
    return parser.parse_args()


def export_model(checkpoint_path):
    """Export model from checkpoint to TorchScript.
    
    Args:
        checkpoint_path: Path to the checkpoint file
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load checkpoint without model (we'll reconstruct from config)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    config = checkpoint['config']
    
    print(f"Reconstructing TitanClassifier from config: {config}")
    
    # Reconstruct TitanClassifier from config
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
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Wrap in NeedleFinder
    needle_finder = NeedleFinder(model)
    needle_finder.eval()
    
    print("Creating example inputs for tracing...")
    
    # Create example inputs (batch_size=2, seq_len=128)
    example_x = torch.randint(1, config['vocab_size']-1, (2, 128))  # Random tokens
    example_lengths = torch.tensor([64, 128])  # Example lengths
    
    print("Tracing model with torch.jit.trace...")
    
    # Trace the model
    with torch.no_grad():
        traced_model = torch.jit.trace(needle_finder, (example_x, example_lengths))
    
    # Determine output directory from checkpoint path
    output_dir = checkpoint_path.parent
    output_path = output_dir / "model_scripted.pt"
    
    print(f"Saving TorchScript model to: {output_path}")
    
    # Save traced model
    traced_model.save(str(output_path))
    
    # Create export manifest
    manifest = {
        "source_checkpoint": str(checkpoint_path),
        "model_config": config,
        "export_format": "torchscript",
        "model_class": "NeedleFinder",
        "input_shape": {
            "x": ["batch_size", "sequence_length"],
            "lengths": ["batch_size"]
        },
        "output_shape": ["batch_size", "sequence_length"],
        "example_input_shapes": {
            "x": list(example_x.shape),
            "lengths": list(example_lengths.shape)
        }
    }
    
    manifest_path = output_dir / "export_manifest.json"
    
    print(f"Saving export manifest to: {manifest_path}")
    
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print("\nExport completed successfully!")
    print(f"TorchScript model: {output_path}")
    print(f"Export manifest: {manifest_path}")
    
    # Test the exported model
    print("\nTesting exported model...")
    loaded_model = torch.jit.load(str(output_path))
    
    with torch.no_grad():
        test_output = loaded_model(example_x, example_lengths)
        print(f"Test output shape: {test_output.shape}")
        print(f"Test output range: [{test_output.min():.4f}, {test_output.max():.4f}]")
    
    return output_path, manifest_path


def main():
    args = parse_args()
    
    try:
        export_model(args.ckpt)
    except Exception as e:
        print(f"Export failed: {e}")
        raise


if __name__ == "__main__":
    main()
