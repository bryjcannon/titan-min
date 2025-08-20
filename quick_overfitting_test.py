#!/usr/bin/env python3
"""
Quick test to validate overfitting hypothesis using existing trained model.
Tests the model on sequences with lengths around 256 to see if the pattern holds.
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict

# Import modules
import sys
sys.path.append('src')
from data.niah import NIAHDataset, collate
from models.titan_min import TitanClassifier
from models.heads import position_logits


def load_trained_model(ckpt_path="artifacts/run1/best.ckpt"):
    """Load the existing trained model that shows the bug."""
    print(f"Loading model from: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = checkpoint['config']
    
    # Create model with original complex memory (to reproduce bug)
    # We need to temporarily use the old model architecture
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, config


def create_test_sequences_around_256():
    """Create test sequences with lengths around 256 to test divisibility hypothesis."""
    
    # Test lengths around 256 (64*4)
    test_lengths = [252, 253, 254, 255, 256, 257, 258, 259, 260]
    
    # Also test other multiples of 64
    multiples_64 = [64, 128, 192, 256, 320]
    
    # And test non-multiples
    non_multiples = [63, 65, 127, 129, 191, 193, 255, 257]
    
    all_test_lengths = sorted(set(test_lengths + multiples_64 + non_multiples))
    
    print(f"Testing lengths: {all_test_lengths}")
    
    # Create datasets for each length
    test_data = {}
    
    for length in all_test_lengths:
        print(f"Creating test data for length {length}...")
        
        # Create small dataset with only this length
        samples = []
        np.random.seed(42)  # Fixed seed for reproducibility
        
        for _ in range(50):  # 50 samples per length
            # Generate sequence
            sequence = torch.randint(1, 127, (length,))  # Tokens 1-126
            
            # Choose random needle position
            needle_pos = np.random.randint(0, length)
            
            # Place needle token (127)
            sequence[needle_pos] = 127
            
            samples.append((sequence, needle_pos))
        
        test_data[length] = samples
    
    return test_data


def test_model_on_length(model, samples, length):
    """Test model accuracy on samples of a specific length."""
    
    # Create simple batch
    sequences = []
    targets = []
    
    for seq, target in samples:
        sequences.append(seq)
        targets.append(target)
    
    # Pad sequences to same length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    lengths = []
    
    for seq in sequences:
        padded = torch.zeros(max_len, dtype=torch.long)
        padded[:len(seq)] = seq
        padded_sequences.append(padded)
        lengths.append(len(seq))
    
    # Create batch tensors
    X = torch.stack(padded_sequences)  # [B, L]
    Y = torch.tensor(targets)  # [B]
    lengths_tensor = torch.tensor(lengths)  # [B]
    
    # Evaluate
    with torch.no_grad():
        h_tokens_out, rep = model(X)
        logits = position_logits(rep, h_tokens_out, lengths_tensor)
        predictions = logits.argmax(dim=-1)
        
        # Calculate accuracy
        correct = (predictions == Y).sum().item()
        total = len(Y)
        accuracy = correct / total
    
    return accuracy, correct, total


def main():
    """Run quick overfitting validation test."""
    print("🔬 Quick Overfitting Hypothesis Validation")
    print("=" * 50)
    
    try:
        # Load trained model
        model, config = load_trained_model()
        print(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Create test data
        print("\nCreating test sequences...")
        test_data = create_test_sequences_around_256()
        
        # Test each length
        print("\nTesting model on different lengths...")
        results = {}
        
        for length in sorted(test_data.keys()):
            samples = test_data[length]
            accuracy, correct, total = test_model_on_length(model, samples, length)
            
            results[length] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': total,
                'is_multiple_of_64': length % 64 == 0
            }
            
            status = "🚨" if accuracy > 0.8 else "✅" if accuracy > 0.3 else "❌"
            divisor_status = "(64×)" if length % 64 == 0 else "     "
            
            print(f"  {status} Length {length:3d} {divisor_status}: {accuracy:.3f} ({correct}/{total})")
        
        # Analyze results
        print(f"\n📊 ANALYSIS:")
        
        multiples_64 = [r for l, r in results.items() if r['is_multiple_of_64']]
        non_multiples = [r for l, r in results.items() if not r['is_multiple_of_64']]
        
        if multiples_64:
            avg_multiple_acc = np.mean([r['accuracy'] for r in multiples_64])
            print(f"  Average accuracy on multiples of 64: {avg_multiple_acc:.3f}")
        
        if non_multiples:
            avg_non_multiple_acc = np.mean([r['accuracy'] for r in non_multiples])
            print(f"  Average accuracy on non-multiples of 64: {avg_non_multiple_acc:.3f}")
        
        if multiples_64 and non_multiples:
            difference = avg_multiple_acc - avg_non_multiple_acc
            print(f"  Difference: {difference:.3f}")
            
            if difference > 0.2:  # 20% threshold
                print(f"  🚨 OVERFITTING DETECTED: Multiples of 64 perform {difference:.1%} better!")
                print(f"  💡 This confirms the segmentation-based overfitting hypothesis.")
            else:
                print(f"  ✅ No significant overfitting pattern detected.")
        
        # Check for suspicious high accuracies
        suspicious = [(l, r) for l, r in results.items() if r['accuracy'] > 0.8]
        if suspicious:
            print(f"\n🚨 SUSPICIOUS HIGH ACCURACIES:")
            for length, result in suspicious:
                divisor_note = " (multiple of 64)" if result['is_multiple_of_64'] else ""
                print(f"    Length {length}: {result['accuracy']:.3f}{divisor_note}")
        
        print(f"\n🎯 CONCLUSION:")
        if any(r['accuracy'] > 0.8 for r in results.values()):
            print(f"  The model shows suspiciously high accuracy on some lengths.")
            print(f"  This supports the architectural overfitting hypothesis.")
        else:
            print(f"  No obvious overfitting patterns detected in this quick test.")
            print(f"  The fix may have already resolved the issue.")
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        print(f"This might be because the model architecture has been fixed.")
        print(f"Try running with the original complex memory model to reproduce the bug.")


if __name__ == "__main__":
    main()
