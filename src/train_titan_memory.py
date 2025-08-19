"""
Enhanced training script for TitanClassifier with true Titan Long-Term Memory.
Includes memory reset between epochs and proper memory state management.
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from .data.niah import NIAHDataset, collate
from .models.titan_min import TitanClassifier
from .models.heads import position_logits
from .utils.checkpoint import save_checkpoint, dump_json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train TitanClassifier with true Titan Memory on NIAH task")
    
    # Training arguments
    parser.add_argument("--out_dir", type=str, default=os.path.join("artifacts", "titan_memory_run"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (smaller for memory efficiency)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (higher for memory learning)")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # Model arguments
    parser.add_argument("--dim", type=int, default=256, help="Model dimension")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--n_mem", type=int, default=4, help="Number of memory slots (for compatibility)")
    
    # Memory-specific arguments
    parser.add_argument("--memory_dim", type=int, default=128, help="Memory module dimension")
    parser.add_argument("--segment_size", type=int, default=64, help="Memory segment size")
    parser.add_argument("--memory_lr", type=float, default=0.01, help="Online memory learning rate")
    parser.add_argument("--n_persistent", type=int, default=4, help="Number of persistent memory slots")
    parser.add_argument("--reset_memory_each_epoch", action="store_true", help="Reset memory state between epochs")
    
    # Ablation arguments
    parser.add_argument("--no_memory", action="store_true", help="Bypass Titan memory completely")
    parser.add_argument("--no_dsconv", action="store_true", help="Bypass depthwise-separable convolutions")
    parser.add_argument("--no_l2", action="store_true", help="Skip L2-normalization of Q/K")
    parser.add_argument("--act", type=str, choices=["relu", "silu"], default="silu", help="Activation function")
    
    return parser.parse_args()


def create_datasets(seed=42):
    """Create train/val/test splits of NIAH dataset."""
    # Create full dataset
    dataset = NIAHDataset(n_samples=10000, seed=seed)
    
    # Create splits with fixed generator seed
    generator = torch.Generator()
    generator.manual_seed(42)
    
    # Split indices: 8000/1000/1000
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    train_indices = indices[:8000]
    val_indices = indices[8000:9000]
    test_indices = indices[9000:]
    
    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def evaluate(model, dataloader, device, reset_memory=False):
    """Evaluate model on validation/test set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    # Reset memory state for clean evaluation if requested
    if reset_memory and hasattr(model, 'long_term_memory') and model.long_term_memory is not None:
        model.long_term_memory.reset_memory()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            X, Y, lengths = batch
            X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
            
            # Forward pass
            h_tokens_out, rep = model(X)
            
            # Compute position logits
            logits = position_logits(rep, h_tokens_out, lengths)
            
            # Compute loss
            loss = criterion(logits, Y)
            total_loss += loss.item()
            
            # Compute accuracy
            predictions = logits.argmax(dim=-1)
            correct += (predictions == Y).sum().item()
            total += Y.size(0)
    
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0.0
    
    return accuracy, avg_loss


def train_epoch(model, train_loader, optimizer, criterion, device, memory_lr=0.01):
    """Train for one epoch with memory updates."""
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, batch in enumerate(pbar):
        X, Y, lengths = batch
        X, Y, lengths = X.to(device), Y.to(device), lengths.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        h_tokens_out, rep = model(X)
        
        # Compute position logits
        logits = position_logits(rep, h_tokens_out, lengths)
        
        # Compute loss
        loss = criterion(logits, Y)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update memory learning rate in the memory module if it exists
        if hasattr(model, 'long_term_memory') and model.long_term_memory is not None:
            # Set memory learning rate for online updates
            if hasattr(model.long_term_memory.memory_module, 'update_memory_online'):
                # The online update happens during forward pass, but we can adjust the rate
                pass
        
        # Update metrics
        train_loss += loss.item()
        predictions = logits.argmax(dim=-1)
        train_correct += (predictions == Y).sum().item()
        train_total += Y.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'acc': f"{train_correct / train_total:.4f}"
        })
    
    train_acc = train_correct / train_total
    train_loss = train_loss / len(train_loader)
    
    return train_acc, train_loss


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets and dataloaders
    print("Creating datasets...")
    train_dataset, val_dataset, test_dataset = create_datasets(args.seed)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model with Titan memory
    print("Creating TitanClassifier with Long-Term Memory...")
    model = TitanClassifier(
        vocab_size=128,
        dim=args.dim,
        n_heads=args.heads,
        n_layers=args.layers,
        n_mem=args.n_mem,  # For compatibility, but not used with new memory
        no_memory=args.no_memory,
        no_dsconv=args.no_dsconv,
        no_l2=args.no_l2,
        activation=args.act
    ).to(device)
    
    # Override memory configuration if using memory
    if not args.no_memory and hasattr(model, 'long_term_memory') and model.long_term_memory is not None:
        # Update memory configuration
        model.long_term_memory.memory_dim = args.memory_dim
        model.long_term_memory.segment_size = args.segment_size
        model.long_term_memory.n_persistent = args.n_persistent
        print(f"Memory configured: dim={args.memory_dim}, segment_size={args.segment_size}, persistent_slots={args.n_persistent}")
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Save config
    config = {
        'vocab_size': 128,
        'dim': args.dim,
        'n_heads': args.heads,
        'n_layers': args.layers,
        'n_mem': args.n_mem,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'seed': args.seed,
        # Memory-specific config
        'memory_dim': args.memory_dim,
        'segment_size': args.segment_size,
        'memory_lr': args.memory_lr,
        'n_persistent': args.n_persistent,
        'reset_memory_each_epoch': args.reset_memory_each_epoch,
        # Ablation flags
        'no_memory': args.no_memory,
        'no_dsconv': args.no_dsconv,
        'no_l2': args.no_l2,
        'activation': args.act
    }
    dump_json(config, out_dir / "config.json")
    print(f"Config saved to {out_dir / 'config.json'}")
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Reset memory state between epochs if requested
        if args.reset_memory_each_epoch and hasattr(model, 'long_term_memory') and model.long_term_memory is not None:
            model.long_term_memory.reset_memory()
            print("Memory state reset for new epoch")
        
        # Training
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.memory_lr)
        
        # Validation (with fresh memory state)
        val_acc, val_loss = evaluate(model, val_loader, device, reset_memory=True)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save checkpoints
        metrics = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        
        # Always save last checkpoint
        save_checkpoint(
            out_dir / "last.ckpt",
            model,
            optimizer,
            epoch + 1,
            metrics,
            config
        )
        
        # Save best checkpoint if improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            save_checkpoint(
                out_dir / "best.ckpt",
                model,
                optimizer,
                epoch + 1,
                metrics,
                config
            )
            
            # Save best metrics
            best_metrics = {
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'best_train_acc': train_acc,
                'best_val_loss': val_loss,
                # Include all config in metrics
                **config
            }
            dump_json(best_metrics, out_dir / "metrics.json")
            
            print(f"New best validation accuracy: {best_val_acc:.4f}")
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Artifacts saved to: {out_dir}")
    
    # Final test evaluation
    if len(test_dataset) > 0:
        test_loader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            collate_fn=collate
        )
        
        # Load best model for test evaluation
        best_ckpt = torch.load(out_dir / "best.ckpt", map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        
        test_acc, test_loss = evaluate(model, test_loader, device, reset_memory=True)
        print(f"\nFinal Test Results:")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        # Save test results
        test_results = {
            'test_acc': test_acc,
            'test_loss': test_loss,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch
        }
        dump_json(test_results, out_dir / "test_results.json")


if __name__ == "__main__":
    main()
