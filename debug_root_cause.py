#!/usr/bin/env python3
"""
Comprehensive diagnostic script to identify the true root cause of length 256 overfitting.
Tests multiple hypotheses beyond segmentation.
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.models.titan_min import TitanClassifier
from src.models.heads import position_logits
from src.data.niah import NIAHDataset
from src.utils.checkpoint import load_checkpoint
import json

def test_without_memory():
    """Test if the bug exists when memory is completely disabled."""
    print("=== Testing Model WITHOUT Memory ===")
    
    # Load the fixed_run model but disable memory
    ckpt_path = "artifacts/fixed_run/best.ckpt"
    checkpoint = load_checkpoint(ckpt_path)
    config = checkpoint['config']
    
    # Create model WITHOUT memory
    model = TitanClassifier(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        n_mem=0,  # NO MEMORY
        no_memory=True,  # Explicitly disable
        no_dsconv=config.get('no_dsconv', False),
        no_l2=config.get('no_l2', False),
        activation=config.get('activation', 'silu')
    )
    
    # Load weights (memory parts will be ignored)
    try:
        model.load_state_dict(checkpoint['model'], strict=False)
    except:
        print("Warning: Could not load all weights (expected due to memory removal)")
    
    model.eval()
    
    # Test on length 256 sequences
    test_dataset = NIAHDataset(
        vocab_size=config['vocab_size'],
        seq_lengths=(256,),
        num_samples=100,
        seed=42
    )
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(min(100, len(test_dataset))):
            tokens, target_pos = test_dataset[i]
            tokens = tokens.unsqueeze(0)  # Add batch dim
            
            h_out, rep = model(tokens)
            logits = position_logits(h_out, rep)
            
            # Mask positions beyond sequence length
            seq_len = tokens.size(1)
            mask = torch.arange(logits.size(-1)) >= seq_len
            logits = logits.masked_fill(mask, float('-inf'))
            
            pred_pos = logits.argmax(dim=-1).item()
            
            if pred_pos == target_pos:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Length 256 accuracy WITHOUT memory: {accuracy:.4f} ({correct}/{total})")
    return accuracy

def test_position_encoding_bias():
    """Test if there's a position encoding bias for length 256."""
    print("\n=== Testing Position Encoding Bias ===")
    
    # Create sequences of different lengths and check if position 256 has special properties
    vocab_size = 1000
    
    for seq_len in [64, 128, 256, 512]:
        # Create dummy sequence
        tokens = torch.randint(1, vocab_size, (1, seq_len))
        
        # Create simple model to test embeddings
        embedding = torch.nn.Embedding(vocab_size, 256)
        h = embedding(tokens)  # [1, seq_len, 256]
        
        # Check if position 256 has unusual properties
        if seq_len >= 256:
            pos_255_norm = torch.norm(h[0, 255]).item()  # 0-indexed, so pos 255 = 256th token
            avg_norm = torch.norm(h[0]).item() / seq_len
            
            print(f"Seq len {seq_len}: Position 256 norm: {pos_255_norm:.4f}, Avg norm: {avg_norm:.4f}")

def test_attention_patterns():
    """Test if attention patterns are unusual for length 256."""
    print("\n=== Testing Attention Patterns ===")
    
    # Load model
    ckpt_path = "artifacts/fixed_run/best.ckpt"
    checkpoint = load_checkpoint(ckpt_path)
    config = checkpoint['config']
    
    model = TitanClassifier(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        n_mem=config['n_mem'],
        no_dsconv=config.get('no_dsconv', False),
        no_l2=config.get('no_l2', False),
        activation=config.get('activation', 'silu')
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test attention patterns for different lengths
    for seq_len in [64, 128, 256]:
        tokens = torch.randint(1, config['vocab_size'], (1, seq_len))
        
        with torch.no_grad():
            h_out, rep = model(tokens)
            
            # Check if there are unusual patterns in the representation
            rep_norm = torch.norm(rep).item()
            h_out_norm = torch.norm(h_out).item()
            
            print(f"Seq len {seq_len}: Rep norm: {rep_norm:.4f}, H_out norm: {h_out_norm:.4f}")

def test_loss_landscape():
    """Test if the loss landscape is different for length 256."""
    print("\n=== Testing Loss Landscape ===")
    
    # Load model
    ckpt_path = "artifacts/fixed_run/best.ckpt"
    checkpoint = load_checkpoint(ckpt_path)
    config = checkpoint['config']
    
    model = TitanClassifier(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        n_mem=config['n_mem'],
        no_dsconv=config.get('no_dsconv', False),
        no_l2=config.get('no_l2', False),
        activation=config.get('activation', 'silu')
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Test loss for different lengths
    for seq_len in [64, 128, 256]:
        test_dataset = NIAHDataset(
            vocab_size=config['vocab_size'],
            seq_lengths=(seq_len,),
            num_samples=50,
            seed=42
        )
        
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for i in range(min(50, len(test_dataset))):
                tokens, target_pos = test_dataset[i]
                tokens = tokens.unsqueeze(0)
                
                h_out, rep = model(tokens)
                logits = position_logits(h_out, rep)
                
                # Mask positions beyond sequence length
                mask = torch.arange(logits.size(-1)) >= tokens.size(1)
                logits = logits.masked_fill(mask, float('-inf'))
                
                target = torch.tensor([target_pos])
                loss = F.cross_entropy(logits, target)
                
                total_loss += loss.item()
                total_samples += 1
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        print(f"Length {seq_len}: Average loss: {avg_loss:.4f}")

def test_gradient_magnitudes():
    """Test if gradients behave differently for length 256."""
    print("\n=== Testing Gradient Magnitudes ===")
    
    # Load model
    ckpt_path = "artifacts/fixed_run/best.ckpt"
    checkpoint = load_checkpoint(ckpt_path)
    config = checkpoint['config']
    
    model = TitanClassifier(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        n_mem=config['n_mem'],
        no_dsconv=config.get('no_dsconv', False),
        no_l2=config.get('no_l2', False),
        activation=config.get('activation', 'silu')
    )
    model.load_state_dict(checkpoint['model'])
    model.train()  # Enable gradients
    
    # Test gradient magnitudes for different lengths
    for seq_len in [64, 128, 256]:
        tokens = torch.randint(1, config['vocab_size'], (1, seq_len))
        target_pos = torch.randint(0, seq_len, (1,))
        
        model.zero_grad()
        
        h_out, rep = model(tokens)
        logits = position_logits(h_out, rep)
        
        # Mask positions beyond sequence length
        mask = torch.arange(logits.size(-1)) >= seq_len
        logits = logits.masked_fill(mask, float('-inf'))
        
        loss = F.cross_entropy(logits, target_pos)
        loss.backward()
        
        # Calculate total gradient magnitude
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        print(f"Length {seq_len}: Total gradient norm: {total_grad_norm:.4f}")

def test_embedding_analysis():
    """Analyze if certain embeddings have special properties."""
    print("\n=== Testing Embedding Analysis ===")
    
    # Load model
    ckpt_path = "artifacts/fixed_run/best.ckpt"
    checkpoint = load_checkpoint(ckpt_path)
    config = checkpoint['config']
    
    model = TitanClassifier(
        vocab_size=config['vocab_size'],
        dim=config['dim'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        n_mem=config['n_mem'],
        no_dsconv=config.get('no_dsconv', False),
        no_l2=config.get('no_l2', False),
        activation=config.get('activation', 'silu')
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Analyze embedding weights
    embedding_weights = model.embedding.weight.data  # [vocab_size, dim]
    
    # Check if needle token (999) has unusual properties
    needle_token = 999
    needle_embedding = embedding_weights[needle_token]
    
    # Compare with other tokens
    other_embeddings = embedding_weights[:needle_token]  # All tokens before needle
    
    needle_norm = torch.norm(needle_embedding).item()
    avg_norm = torch.norm(other_embeddings, dim=1).mean().item()
    
    print(f"Needle token ({needle_token}) embedding norm: {needle_norm:.4f}")
    print(f"Average other token embedding norm: {avg_norm:.4f}")
    print(f"Needle norm ratio: {needle_norm / avg_norm:.4f}")

if __name__ == "__main__":
    print("🔍 Comprehensive Root Cause Analysis for Length 256 Overfitting")
    print("=" * 70)
    
    # Test 1: Model without memory
    no_memory_acc = test_without_memory()
    
    # Test 2: Position encoding bias
    test_position_encoding_bias()
    
    # Test 3: Attention patterns
    test_attention_patterns()
    
    # Test 4: Loss landscape
    test_loss_landscape()
    
    # Test 5: Gradient magnitudes
    test_gradient_magnitudes()
    
    # Test 6: Embedding analysis
    test_embedding_analysis()
    
    print("\n" + "=" * 70)
    print("🎯 SUMMARY OF FINDINGS:")
    print(f"- Length 256 accuracy WITHOUT memory: {no_memory_acc:.4f}")
    
    if no_memory_acc > 0.9:
        print("❌ BUG PERSISTS without memory - NOT a segmentation/memory issue!")
        print("🔍 Look at: embeddings, position encoding, attention, or evaluation logic")
    elif no_memory_acc < 0.1:
        print("✅ Bug disappears without memory - segmentation/memory IS the issue")
        print("🔍 Need stronger memory disruption techniques")
    else:
        print("⚠️  Partial improvement without memory - mixed causes")
        print("🔍 Both memory AND other components may be involved")
