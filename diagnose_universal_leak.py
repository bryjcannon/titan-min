#!/usr/bin/env python3
"""
Diagnostic test to understand why the model achieves 100% accuracy on all lengths.
This investigates potential sources of universal data leakage.
"""

import torch
import torch.nn as nn
import numpy as np

# Import modules
import sys
sys.path.append('src')
from data.niah import NIAHDataset, collate
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


def test_needle_token_detection(model):
    """Test if the model can detect needle tokens without position information."""
    print("🔍 Testing needle token detection...")
    
    # Create a simple sequence with needle at known position
    seq_len = 10
    sequence = torch.randint(1, 127, (1, seq_len))  # [1, L]
    needle_pos = 5
    sequence[0, needle_pos] = 127  # Place needle
    
    print(f"  Sequence: {sequence[0].tolist()}")
    print(f"  Needle at position: {needle_pos}")
    
    # Forward pass
    with torch.no_grad():
        h_tokens_out, rep = model(sequence)
        
        # Check token embeddings
        embeddings = model.embedding(sequence)  # [1, L, C]
        needle_embedding = embeddings[0, needle_pos]  # [C]
        other_embeddings = embeddings[0, [i for i in range(seq_len) if i != needle_pos]]  # [L-1, C]
        
        print(f"  Needle embedding norm: {torch.norm(needle_embedding).item():.4f}")
        print(f"  Other embeddings norm (avg): {torch.norm(other_embeddings, dim=-1).mean().item():.4f}")
        
        # Check if needle embedding is obviously different
        needle_norm = torch.norm(needle_embedding)
        other_norms = torch.norm(other_embeddings, dim=-1)
        
        if needle_norm > other_norms.max() * 2:
            print(f"  🚨 LEAK: Needle embedding has suspiciously high norm!")
        
        # Test position logits
        lengths = torch.tensor([seq_len])
        logits = position_logits(rep, h_tokens_out, lengths)
        prediction = logits.argmax(dim=-1).item()
        
        print(f"  Predicted position: {prediction}")
        print(f"  Correct: {prediction == needle_pos}")
        
        return prediction == needle_pos


def test_position_independence(model):
    """Test if model prediction depends on actual needle position or just sequence content."""
    print("\n🔍 Testing position independence...")
    
    seq_len = 8
    base_sequence = torch.randint(1, 127, (seq_len,))
    
    results = []
    
    for needle_pos in range(seq_len):
        # Create sequence with needle at different positions
        sequence = base_sequence.clone()
        sequence[needle_pos] = 127
        
        # Forward pass
        with torch.no_grad():
            X = sequence.unsqueeze(0)  # [1, L]
            lengths = torch.tensor([seq_len])
            
            h_tokens_out, rep = model(X)
            logits = position_logits(rep, h_tokens_out, lengths)
            prediction = logits.argmax(dim=-1).item()
            
            results.append({
                'true_pos': needle_pos,
                'predicted_pos': prediction,
                'correct': prediction == needle_pos
            })
    
    print(f"  Results:")
    for r in results:
        status = "✓" if r['correct'] else "✗"
        print(f"    Needle at {r['true_pos']} → Predicted {r['predicted_pos']} {status}")
    
    accuracy = sum(r['correct'] for r in results) / len(results)
    print(f"  Overall accuracy: {accuracy:.3f}")
    
    return accuracy


def test_needle_token_uniqueness(model):
    """Test what happens when we use different 'needle' tokens."""
    print("\n🔍 Testing needle token uniqueness...")
    
    seq_len = 6
    needle_pos = 3
    
    # Test different tokens as 'needles'
    test_tokens = [63, 64, 65, 126, 127]  # Including the real needle (127)
    
    for token in test_tokens:
        sequence = torch.randint(1, 127, (1, seq_len))
        sequence[0, needle_pos] = token
        
        with torch.no_grad():
            lengths = torch.tensor([seq_len])
            h_tokens_out, rep = model(sequence)
            logits = position_logits(rep, h_tokens_out, lengths)
            prediction = logits.argmax(dim=-1).item()
            
            correct = prediction == needle_pos
            status = "✓" if correct else "✗"
            special = " (REAL NEEDLE)" if token == 127 else ""
            
            print(f"  Token {token:3d} at pos {needle_pos} → Predicted {prediction} {status}{special}")


def test_random_sequences(model, n_tests=20):
    """Test model on completely random sequences to see if it still cheats."""
    print(f"\n🔍 Testing {n_tests} random sequences...")
    
    correct_count = 0
    
    for i in range(n_tests):
        # Random length between 5 and 20
        seq_len = np.random.randint(5, 21)
        
        # Random sequence
        sequence = torch.randint(1, 127, (seq_len,))
        
        # Random needle position
        needle_pos = np.random.randint(0, seq_len)
        sequence[needle_pos] = 127
        
        # Test
        with torch.no_grad():
            X = sequence.unsqueeze(0)
            lengths = torch.tensor([seq_len])
            
            h_tokens_out, rep = model(X)
            logits = position_logits(rep, h_tokens_out, lengths)
            prediction = logits.argmax(dim=-1).item()
            
            if prediction == needle_pos:
                correct_count += 1
    
    accuracy = correct_count / n_tests
    print(f"  Random sequence accuracy: {accuracy:.3f} ({correct_count}/{n_tests})")
    
    if accuracy > 0.8:
        print(f"  🚨 UNIVERSAL LEAK: Model cheats even on random sequences!")
    
    return accuracy


def analyze_model_internals(model):
    """Analyze model internals for potential leakage sources."""
    print(f"\n🔍 Analyzing model internals...")
    
    # Check embedding weights
    embedding_weights = model.embedding.weight  # [vocab_size, dim]
    needle_embedding = embedding_weights[127]  # Needle token embedding
    other_embeddings = embedding_weights[:127]  # Other token embeddings
    
    needle_norm = torch.norm(needle_embedding)
    other_norms = torch.norm(other_embeddings, dim=-1)
    
    print(f"  Needle token (127) embedding norm: {needle_norm:.4f}")
    print(f"  Other tokens embedding norm (avg): {other_norms.mean():.4f}")
    print(f"  Other tokens embedding norm (max): {other_norms.max():.4f}")
    
    if needle_norm > other_norms.max() * 2:
        print(f"  🚨 EMBEDDING LEAK: Needle token has suspiciously high norm!")
    
    # Check if needle embedding is an outlier
    z_score = (needle_norm - other_norms.mean()) / other_norms.std()
    print(f"  Needle embedding z-score: {z_score:.2f}")
    
    if abs(z_score) > 3:
        print(f"  🚨 STATISTICAL ANOMALY: Needle embedding is a significant outlier!")


def main():
    """Run comprehensive diagnostic tests."""
    print("🔬 Diagnosing Universal Data Leakage")
    print("=" * 50)
    
    try:
        # Load model
        model, config = load_model()
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Run diagnostic tests
        test1 = test_needle_token_detection(model)
        test2 = test_position_independence(model)
        test_needle_token_uniqueness(model)
        test3 = test_random_sequences(model)
        analyze_model_internals(model)
        
        # Summary
        print(f"\n🎯 DIAGNOSTIC SUMMARY:")
        print(f"  Needle detection test: {'PASS' if test1 else 'FAIL'}")
        print(f"  Position independence: {test2:.3f}")
        print(f"  Random sequence accuracy: {test3:.3f}")
        
        if test3 > 0.8:
            print(f"\n🚨 CONCLUSION: Universal data leakage confirmed!")
            print(f"  The model has found a way to cheat on ALL sequences.")
            print(f"  This is likely due to:")
            print(f"    1. Needle token embedding anomaly")
            print(f"    2. Architecture allowing direct position access")
            print(f"    3. Memory system storing absolute positions")
        else:
            print(f"\n✅ CONCLUSION: No universal leakage detected in random tests.")
            print(f"  The 100% accuracy might be specific to test setup.")
        
    except Exception as e:
        print(f"❌ Error in diagnostic: {e}")


if __name__ == "__main__":
    main()
