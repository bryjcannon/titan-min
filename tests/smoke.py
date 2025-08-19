"""
Smoke tests for Titan-Min repository.

This script performs quick sanity checks to ensure all core components work correctly:
1. Dataset construction and validation
2. Model construction and forward pass
3. Single training step validation

Run with: python tests/smoke.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

# Import our modules
from data.niah import NIAHDataset, collate
from models.titan_min import TitanClassifier
from models.heads import position_logits


def test_dataset_construction():
    """Test dataset construction and validate NEEDLE placement."""
    print("🔍 Testing dataset construction...")
    
    # Create small dataset for testing
    dataset = NIAHDataset(n_samples=100, seed=42)
    
    print(f"  ✓ Dataset created with {len(dataset)} samples")
    
    # Test multiple samples
    for i in range(10):
        tokens, needle_pos, length = dataset[i]
        
        # Assert tokens is a tensor
        assert isinstance(tokens, torch.Tensor), f"Sample {i}: tokens should be tensor"
        
        # Assert needle_pos is in valid range [0, length-1]
        assert 0 <= needle_pos < length, f"Sample {i}: needle_pos {needle_pos} not in [0, {length-1}]"
        
        # Assert exactly one NEEDLE token (127) in the sequence
        needle_count = (tokens[:length] == 127).sum().item()
        assert needle_count == 1, f"Sample {i}: found {needle_count} needles, expected 1"
        
        # Assert NEEDLE is at the correct position
        actual_needle_pos = torch.where(tokens[:length] == 127)[0].item()
        assert actual_needle_pos == needle_pos, f"Sample {i}: needle at pos {actual_needle_pos}, expected {needle_pos}"
        
        # Assert sequence length is valid
        assert length in [64, 128, 256], f"Sample {i}: invalid length {length}"
        
        # Assert no PAD tokens within the sequence
        pad_count = (tokens[:length] == 0).sum().item()
        assert pad_count == 0, f"Sample {i}: found {pad_count} PAD tokens within sequence"
    
    print("  ✓ All dataset samples validated")
    
    # Test collate function
    batch_samples = [dataset[i] for i in range(5)]
    batch_tokens, batch_needle_pos, batch_lengths = collate(batch_samples)
    
    assert batch_tokens.shape[0] == 5, "Batch size should be 5"
    assert len(batch_needle_pos) == 5, "Needle positions batch size should be 5"
    assert len(batch_lengths) == 5, "Lengths batch size should be 5"
    
    # Check left-padding
    for i in range(5):
        length = batch_lengths[i]
        max_len = batch_tokens.shape[1]
        
        # Check that padding is on the left
        if length < max_len:
            pad_tokens = batch_tokens[i, :max_len-length]
            assert (pad_tokens == 0).all(), f"Left padding should be all zeros"
        
        # Check that sequence content is preserved
        sequence_tokens = batch_tokens[i, max_len-length:]
        original_tokens = batch_samples[i][0][:length]
        assert torch.equal(sequence_tokens, original_tokens), f"Sequence content not preserved in batch"
    
    print("  ✓ Collate function validated")
    print("✅ Dataset construction tests passed\n")


def test_model_construction():
    """Test model construction and forward pass shapes."""
    print("🔍 Testing model construction...")
    
    # Test default model
    model = TitanClassifier(
        vocab_size=128,
        dim=64,  # Small for testing
        n_heads=4,
        n_layers=2,
        n_mem=2
    )
    
    print(f"  ✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass with random input
    batch_size, seq_len = 2, 32
    x = torch.randint(0, 128, (batch_size, seq_len))
    
    model.eval()
    with torch.no_grad():
        h_tokens, rep = model(x)
    
    # Assert output shapes
    expected_tokens_shape = (batch_size, seq_len, 64)  # [B, L, C]
    expected_rep_shape = (batch_size, 64)  # [B, C]
    
    assert h_tokens.shape == expected_tokens_shape, f"Token states shape {h_tokens.shape}, expected {expected_tokens_shape}"
    assert rep.shape == expected_rep_shape, f"Rep shape {rep.shape}, expected {expected_rep_shape}"
    
    print(f"  ✓ Forward pass shapes: tokens {h_tokens.shape}, rep {rep.shape}")
    
    # Test position_logits function
    lengths = torch.tensor([20, 25])  # Different lengths for each sample
    logits = position_logits(rep, h_tokens, lengths)
    
    expected_logits_shape = (batch_size, seq_len)  # [B, L]
    assert logits.shape == expected_logits_shape, f"Logits shape {logits.shape}, expected {expected_logits_shape}"
    
    # Check masking: positions >= lengths should be -inf
    for i in range(batch_size):
        length = lengths[i]
        masked_positions = logits[i, length:]
        assert torch.all(torch.isinf(masked_positions) & (masked_positions < 0)), f"Positions >= length should be -inf"
        
        # Valid positions should be finite
        valid_positions = logits[i, :length]
        assert torch.all(torch.isfinite(valid_positions)), f"Valid positions should be finite"
    
    print("  ✓ Position logits and masking validated")
    
    # Test ablation models
    ablation_configs = [
        {"no_memory": True},
        {"no_dsconv": True},
        {"no_l2": True},
        {"activation": "relu"},
        {"no_memory": True, "no_dsconv": True, "no_l2": True, "activation": "relu"}
    ]
    
    for i, config in enumerate(ablation_configs):
        model_ablated = TitanClassifier(
            vocab_size=128, dim=64, n_heads=4, n_layers=2, n_mem=2, **config
        )
        
        model_ablated.eval()
        with torch.no_grad():
            h_tokens_abl, rep_abl = model_ablated(x)
        
        # Shapes should be the same regardless of ablations
        assert h_tokens_abl.shape == expected_tokens_shape, f"Ablation {i}: wrong token shape"
        assert rep_abl.shape == expected_rep_shape, f"Ablation {i}: wrong rep shape"
    
    print(f"  ✓ All {len(ablation_configs)} ablation configurations validated")
    print("✅ Model construction tests passed\n")


def test_training_step():
    """Test single training step to ensure loss computation and backprop work."""
    print("🔍 Testing training step...")
    
    # Create model and data
    model = TitanClassifier(vocab_size=128, dim=64, n_heads=4, n_layers=2, n_mem=2)
    dataset = NIAHDataset(n_samples=50, seed=42)
    dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    # Get a batch
    batch_tokens, batch_needle_pos, batch_lengths = next(iter(dataloader))
    
    print(f"  ✓ Batch loaded: {batch_tokens.shape}")
    
    # Store initial losses for comparison
    losses = []
    
    # Run a few training steps
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass
        h_tokens, rep = model(batch_tokens)
        
        # Compute position logits
        logits = position_logits(rep, h_tokens, batch_lengths)
        
        # Compute loss
        loss = criterion(logits, batch_needle_pos)
        
        # Assertions
        assert torch.isfinite(loss), f"Step {step}: loss is not finite: {loss}"
        assert loss.item() > 0, f"Step {step}: loss should be positive: {loss.item()}"
        
        losses.append(loss.item())
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist and are finite
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert torch.isfinite(torch.tensor(grad_norm)), f"Step {step}: infinite gradient in {name}"
                grad_norms.append(grad_norm)
        
        assert len(grad_norms) > 0, f"Step {step}: no gradients computed"
        
        optimizer.step()
        
        print(f"  ✓ Step {step}: loss = {loss.item():.4f}, grad_norm = {np.mean(grad_norms):.4f}")
    
    # Check that loss generally decreases (not strict requirement)
    if len(losses) >= 3:
        # Use moving average to smooth out noise
        early_avg = np.mean(losses[:2])
        late_avg = np.mean(losses[-2:])
        
        if late_avg >= early_avg:
            print(f"  ⚠️  Warning: Loss not decreasing (early: {early_avg:.4f}, late: {late_avg:.4f})")
        else:
            print(f"  ✓ Loss decreasing: {early_avg:.4f} → {late_avg:.4f}")
    
    print("✅ Training step tests passed\n")


def test_end_to_end():
    """Test end-to-end workflow with small dataset."""
    print("🔍 Testing end-to-end workflow...")
    
    # Create small dataset
    dataset = NIAHDataset(n_samples=32, seed=42)
    train_size = 24
    val_size = 8
    
    train_dataset = torch.utils.data.Subset(dataset, range(train_size))
    val_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    
    train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=collate)
    val_loader = DataLoader(val_dataset, batch_size=4, collate_fn=collate)
    
    # Create model
    model = TitanClassifier(vocab_size=128, dim=32, n_heads=2, n_layers=1, n_mem=1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print(f"  ✓ Setup: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Training epoch
    model.train()
    train_losses = []
    
    for batch_tokens, batch_needle_pos, batch_lengths in train_loader:
        optimizer.zero_grad()
        h_tokens, rep = model(batch_tokens)
        logits = position_logits(rep, h_tokens, batch_lengths)
        loss = criterion(logits, batch_needle_pos)
        
        assert torch.isfinite(loss), "Training loss not finite"
        train_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
    
    avg_train_loss = np.mean(train_losses)
    print(f"  ✓ Training: avg loss = {avg_train_loss:.4f}")
    
    # Validation
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_tokens, batch_needle_pos, batch_lengths in val_loader:
            h_tokens, rep = model(batch_tokens)
            logits = position_logits(rep, h_tokens, batch_lengths)
            loss = criterion(logits, batch_needle_pos)
            
            assert torch.isfinite(loss), "Validation loss not finite"
            val_losses.append(loss.item())
            
            # Compute accuracy
            predictions = logits.argmax(dim=1)
            correct += (predictions == batch_needle_pos).sum().item()
            total += batch_needle_pos.size(0)
    
    avg_val_loss = np.mean(val_losses)
    accuracy = correct / total if total > 0 else 0.0
    
    print(f"  ✓ Validation: avg loss = {avg_val_loss:.4f}, accuracy = {accuracy:.4f}")
    
    # Sanity checks
    assert 0.0 <= accuracy <= 1.0, f"Accuracy should be in [0,1], got {accuracy}"
    assert avg_val_loss > 0, f"Validation loss should be positive, got {avg_val_loss}"
    
    print("✅ End-to-end workflow tests passed\n")


def main():
    """Run all smoke tests."""
    print("🚀 Starting Titan-Min Smoke Tests\n")
    print("=" * 50)
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Run all tests
        test_dataset_construction()
        test_model_construction()
        test_training_step()
        test_end_to_end()
        
        print("=" * 50)
        print("🎉 ALL SMOKE TESTS PASSED!")
        print("The Titan-Min repository is ready for use.")
        
    except Exception as e:
        print("=" * 50)
        print(f"❌ SMOKE TEST FAILED: {e}")
        print("Please check the implementation and try again.")
        raise


if __name__ == "__main__":
    main()
