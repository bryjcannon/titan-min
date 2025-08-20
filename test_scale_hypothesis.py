#!/usr/bin/env python3
"""
Test if the overfitting issue is related to small input scale.
Tests model performance on very long sequences to see if overfitting disappears.
"""

import torch
import numpy as np
from collections import defaultdict

# Import modules
import sys
sys.path.append('src')
from models.titan_min import TitanClassifier
from models.heads import position_logits


def load_model(ckpt_path="artifacts/run1/best.ckpt"):
    """Load the trained model."""
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    config = checkpoint['config']
    
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


def create_long_sequence(length, needle_pos=None, seed=42):
    """Create a very long sequence with needle."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate sequence
    sequence = torch.randint(1, 127, (length,))
    
    # Place needle
    if needle_pos is None:
        needle_pos = np.random.randint(0, length)
    
    sequence[needle_pos] = 127  # Needle token
    
    return sequence, needle_pos


def test_scale_effect(model):
    """Test model performance on sequences of increasing length."""
    print("🔬 Testing Scale Effect on Overfitting")
    print("=" * 50)
    
    # Test different scales
    test_lengths = [
        64,      # 1 segment
        128,     # 2 segments  
        256,     # 4 segments (known overfitting)
        512,     # 8 segments
        1024,    # 16 segments
        2048,    # 32 segments
        4096,    # 64 segments (memory buffer overflow)
        8192,    # 128 segments (severe overflow)
    ]
    
    results = {}
    
    for length in test_lengths:
        print(f"\n📏 Testing length {length} ({length//64} segments)...")
        
        # Test multiple samples
        accuracies = []
        
        for sample in range(5):  # 5 samples per length
            try:
                # Create sequence
                sequence, needle_pos = create_long_sequence(length, seed=42+sample)
                
                # Prepare input
                X = sequence.unsqueeze(0)  # [1, L]
                lengths_tensor = torch.tensor([length])
                
                # Forward pass
                with torch.no_grad():
                    h_tokens_out, rep = model(X)
                    logits = position_logits(rep, h_tokens_out, lengths_tensor)
                    prediction = logits.argmax(dim=-1).item()
                
                # Check accuracy
                is_correct = (prediction == needle_pos)
                accuracies.append(is_correct)
                
                print(f"  Sample {sample+1}: Needle at {needle_pos}, Predicted {prediction} {'✓' if is_correct else '✗'}")
                
            except Exception as e:
                print(f"  Sample {sample+1}: ERROR - {e}")
                accuracies.append(False)
        
        # Calculate accuracy for this length
        accuracy = sum(accuracies) / len(accuracies)
        num_segments = length // 64
        buffer_overflow = num_segments > 10  # Memory buffer maxlen=10
        
        results[length] = {
            'accuracy': accuracy,
            'num_segments': num_segments,
            'buffer_overflow': buffer_overflow,
            'samples': len(accuracies)
        }
        
        overflow_note = " (BUFFER OVERFLOW)" if buffer_overflow else ""
        print(f"  📊 Length {length}: {accuracy:.3f} accuracy ({num_segments} segments){overflow_note}")
    
    return results


def analyze_scale_results(results):
    """Analyze results to see if overfitting decreases with scale."""
    print(f"\n📊 SCALE ANALYSIS:")
    print(f"{'Length':<8} {'Segments':<10} {'Accuracy':<10} {'Overflow':<10} {'Status'}")
    print("-" * 50)
    
    overfitting_lengths = []
    normal_lengths = []
    
    for length, result in results.items():
        accuracy = result['accuracy']
        num_segments = result['num_segments']
        overflow = result['buffer_overflow']
        
        # Classify performance
        if accuracy > 0.8:
            status = "🚨 OVERFITTING"
            overfitting_lengths.append(length)
        elif accuracy > 0.3:
            status = "✅ NORMAL"
            normal_lengths.append(length)
        else:
            status = "❌ POOR"
            normal_lengths.append(length)
        
        overflow_str = "YES" if overflow else "NO"
        
        print(f"{length:<8} {num_segments:<10} {accuracy:<10.3f} {overflow_str:<10} {status}")
    
    # Analysis
    print(f"\n🎯 HYPOTHESIS TESTING:")
    
    # Test 1: Does overfitting decrease with scale?
    small_scale = [r['accuracy'] for l, r in results.items() if r['num_segments'] <= 4]
    large_scale = [r['accuracy'] for l, r in results.items() if r['num_segments'] > 10]
    
    if small_scale and large_scale:
        avg_small = np.mean(small_scale)
        avg_large = np.mean(large_scale)
        
        print(f"  Small scale (≤4 segments): {avg_small:.3f} average accuracy")
        print(f"  Large scale (>10 segments): {avg_large:.3f} average accuracy")
        print(f"  Difference: {avg_small - avg_large:.3f}")
        
        if avg_small > avg_large + 0.2:
            print(f"  ✅ HYPOTHESIS CONFIRMED: Overfitting decreases with scale!")
        else:
            print(f"  ❌ HYPOTHESIS REJECTED: No clear scale effect")
    
    # Test 2: Does buffer overflow prevent overfitting?
    no_overflow = [r['accuracy'] for r in results.values() if not r['buffer_overflow']]
    with_overflow = [r['accuracy'] for r in results.values() if r['buffer_overflow']]
    
    if no_overflow and with_overflow:
        avg_no_overflow = np.mean(no_overflow)
        avg_with_overflow = np.mean(with_overflow)
        
        print(f"\n  No buffer overflow: {avg_no_overflow:.3f} average accuracy")
        print(f"  With buffer overflow: {avg_with_overflow:.3f} average accuracy")
        
        if avg_no_overflow > avg_with_overflow + 0.2:
            print(f"  ✅ BUFFER HYPOTHESIS CONFIRMED: Overflow prevents overfitting!")
        else:
            print(f"  ❌ BUFFER HYPOTHESIS REJECTED: No clear buffer effect")
    
    # Test 3: Specific length 256 analysis
    if 256 in results:
        length_256_acc = results[256]['accuracy']
        other_accs = [r['accuracy'] for l, r in results.items() if l != 256]
        
        if other_accs:
            avg_others = np.mean(other_accs)
            print(f"\n  Length 256 accuracy: {length_256_acc:.3f}")
            print(f"  Other lengths average: {avg_others:.3f}")
            
            if length_256_acc > avg_others + 0.3:
                print(f"  🚨 LENGTH 256 ANOMALY CONFIRMED!")
            else:
                print(f"  ✅ Length 256 behaves normally")


def main():
    """Run scale hypothesis test."""
    try:
        # Load model
        model, config = load_model()
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test scale effect
        results = test_scale_effect(model)
        
        # Analyze results
        analyze_scale_results(results)
        
        print(f"\n💡 CONCLUSION:")
        print(f"  If overfitting decreases with scale, it confirms the hypothesis that")
        print(f"  the issue is related to small input size allowing memory exploitation.")
        print(f"  Large sequences should overwhelm the memory buffer and prevent cheating.")
        
    except Exception as e:
        print(f"❌ Error in scale test: {e}")


if __name__ == "__main__":
    main()
