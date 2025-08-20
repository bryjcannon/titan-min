#!/usr/bin/env python3
"""
Script to test the architectural overfitting hypothesis.
Tests different sequence lengths and segment sizes to validate if the model
exploits segmentation arithmetic rather than learning genuine needle detection.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
import argparse

# Import modules
import sys
sys.path.append('src')
from data.niah import NIAHDataset, collate
from models.titan_min import TitanClassifier
from models.heads import position_logits


class OverfittingTester:
    """Test different configurations to validate architectural overfitting hypothesis."""
    
    def __init__(self, base_config: Dict[str, Any]):
        self.base_config = base_config
        self.results = {}
        
    def create_dataset_with_lengths(self, seq_lens: Tuple[int, ...], n_samples: int = 1000, seed: int = 42):
        """Create a dataset with specific sequence lengths."""
        # Temporarily modify NIAHDataset to use custom lengths
        original_init = NIAHDataset.__init__
        
        def custom_init(self, n_samples_inner=n_samples, seq_lens_inner=seq_lens, vocab_size=128, seed_inner=seed):
            original_init(self, n_samples_inner, seq_lens_inner, vocab_size, seed_inner)
        
        # Monkey patch temporarily
        NIAHDataset.__init__ = custom_init
        dataset = NIAHDataset()
        NIAHDataset.__init__ = original_init  # Restore original
        
        return dataset
    
    def quick_train_and_evaluate(self, dataset, config_name: str, epochs: int = 3) -> Dict[str, float]:
        """Quickly train and evaluate a model on the given dataset."""
        print(f"\n=== Testing Configuration: {config_name} ===")
        
        # Create data loader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)
        
        # Create model
        model = TitanClassifier(
            vocab_size=self.base_config['vocab_size'],
            dim=self.base_config['dim'],
            n_heads=self.base_config['n_heads'],
            n_layers=self.base_config['n_layers'],
            n_mem=self.base_config['n_mem'],
            no_memory=self.base_config.get('no_memory', False)
        )
        
        # Quick training
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            total_correct = 0
            total_samples = 0
            
            for batch in dataloader:
                X, Y, lengths = batch
                
                optimizer.zero_grad()
                
                # Forward pass
                h_tokens_out, rep = model(X)
                logits = position_logits(rep, h_tokens_out, lengths)
                
                # Compute loss
                loss = criterion(logits, Y)
                loss.backward()
                optimizer.step()
                
                # Track metrics
                predictions = logits.argmax(dim=-1)
                correct = (predictions == Y).sum().item()
                
                total_loss += loss.item()
                total_correct += correct
                total_samples += X.size(0)
            
            accuracy = total_correct / total_samples
            print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(dataloader):.4f}, Acc={accuracy:.4f}")
        
        # Evaluate by length
        return self.evaluate_by_length(model, dataset)
    
    def evaluate_by_length(self, model, dataset) -> Dict[str, float]:
        """Evaluate model accuracy by sequence length."""
        model.eval()
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate)
        
        # Track accuracy by length
        length_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        with torch.no_grad():
            for batch in dataloader:
                X, Y, lengths = batch
                
                # Forward pass
                h_tokens_out, rep = model(X)
                logits = position_logits(rep, h_tokens_out, lengths)
                predictions = logits.argmax(dim=-1)
                
                # Update stats by length
                for i in range(X.size(0)):
                    length = lengths[i].item()
                    is_correct = (predictions[i] == Y[i]).item()
                    
                    length_stats[length]['correct'] += is_correct
                    length_stats[length]['total'] += 1
        
        # Calculate accuracies
        accuracies = {}
        for length, stats in length_stats.items():
            if stats['total'] > 0:
                accuracies[length] = stats['correct'] / stats['total']
            else:
                accuracies[length] = 0.0
        
        # Print results
        print(f"  Results by length:")
        for length in sorted(accuracies.keys()):
            acc = accuracies[length]
            count = length_stats[length]['total']
            print(f"    Length {length}: {acc:.4f} ({count} samples)")
        
        return accuracies
    
    def test_configuration(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Test a specific configuration."""
        config_name = config['name']
        seq_lens = config['seq_lens']
        
        print(f"\nTesting {config_name} with lengths {seq_lens}")
        
        # Create dataset
        dataset = self.create_dataset_with_lengths(seq_lens, n_samples=1000)
        
        # Train and evaluate
        accuracies = self.quick_train_and_evaluate(dataset, config_name, epochs=3)
        
        # Analyze results
        analysis = self.analyze_results(accuracies, seq_lens, config.get('segment_size', 64))
        
        return {
            'config': config,
            'accuracies': accuracies,
            'analysis': analysis
        }
    
    def analyze_results(self, accuracies: Dict[int, float], seq_lens: Tuple[int, ...], segment_size: int) -> Dict[str, Any]:
        """Analyze results for overfitting patterns."""
        analysis = {
            'perfect_divisors': [],
            'non_divisors': [],
            'suspicious_accuracies': [],
            'overfitting_detected': False
        }
        
        for length in seq_lens:
            acc = accuracies.get(length, 0.0)
            
            # Check if length is perfectly divisible by segment size
            if length % segment_size == 0:
                analysis['perfect_divisors'].append({
                    'length': length,
                    'accuracy': acc,
                    'segments': length // segment_size
                })
            else:
                analysis['non_divisors'].append({
                    'length': length,
                    'accuracy': acc
                })
            
            # Flag suspiciously high accuracies (>80%)
            if acc > 0.8:
                analysis['suspicious_accuracies'].append({
                    'length': length,
                    'accuracy': acc,
                    'is_divisor': length % segment_size == 0
                })
        
        # Detect overfitting pattern
        divisor_accs = [item['accuracy'] for item in analysis['perfect_divisors']]
        non_divisor_accs = [item['accuracy'] for item in analysis['non_divisors']]
        
        if divisor_accs and non_divisor_accs:
            avg_divisor_acc = np.mean(divisor_accs)
            avg_non_divisor_acc = np.mean(non_divisor_accs)
            
            # If divisors perform significantly better, flag as overfitting
            if avg_divisor_acc > avg_non_divisor_acc + 0.2:  # 20% threshold
                analysis['overfitting_detected'] = True
                analysis['divisor_advantage'] = avg_divisor_acc - avg_non_divisor_acc
        
        return analysis
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all overfitting hypothesis tests."""
        print("🔬 Testing Architectural Overfitting Hypothesis")
        print("=" * 60)
        
        # Define test configurations
        test_configs = [
            {
                'name': 'original_problem',
                'seq_lens': (64, 128, 256),
                'segment_size': 64,
                'description': 'Original configuration that showed length 256 overfitting'
            },
            {
                'name': 'asymmetric_lengths',
                'seq_lens': (63, 64, 65, 127, 128, 129, 255, 256, 257),
                'segment_size': 64,
                'description': 'Test lengths around perfect divisors'
            },
            {
                'name': 'smaller_segments',
                'seq_lens': (32, 64, 96, 128),
                'segment_size': 32,
                'description': 'Smaller segment size (32 instead of 64)'
            },
            {
                'name': 'prime_segments',
                'seq_lens': (37, 74, 111, 148),
                'segment_size': 37,
                'description': 'Prime segment size to eliminate divisibility patterns'
            },
            {
                'name': 'multiple_divisors',
                'seq_lens': (64, 128, 192, 256, 320),
                'segment_size': 64,
                'description': 'Multiple perfect divisors of 64'
            }
        ]
        
        all_results = {}
        
        for config in test_configs:
            try:
                result = self.test_configuration(config)
                all_results[config['name']] = result
                
                # Print analysis
                analysis = result['analysis']
                print(f"\n📊 Analysis for {config['name']}:")
                print(f"  Overfitting detected: {analysis['overfitting_detected']}")
                if analysis['overfitting_detected']:
                    print(f"  Divisor advantage: {analysis.get('divisor_advantage', 0):.3f}")
                if analysis['suspicious_accuracies']:
                    print(f"  Suspicious high accuracies: {len(analysis['suspicious_accuracies'])}")
                    for item in analysis['suspicious_accuracies']:
                        print(f"    Length {item['length']}: {item['accuracy']:.3f} (divisor: {item['is_divisor']})")
                
            except Exception as e:
                print(f"❌ Error testing {config['name']}: {e}")
                all_results[config['name']] = {'error': str(e)}
        
        return all_results
    
    def generate_report(self, results: Dict[str, Any], output_file: str = 'overfitting_test_results.json'):
        """Generate a comprehensive report of the overfitting tests."""
        
        # Save detailed results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate summary
        print(f"\n📋 OVERFITTING HYPOTHESIS TEST SUMMARY")
        print("=" * 60)
        
        overfitting_count = 0
        total_tests = 0
        
        for config_name, result in results.items():
            if 'error' in result:
                print(f"❌ {config_name}: ERROR - {result['error']}")
                continue
                
            total_tests += 1
            analysis = result['analysis']
            
            if analysis['overfitting_detected']:
                overfitting_count += 1
                print(f"🚨 {config_name}: OVERFITTING DETECTED")
            else:
                print(f"✅ {config_name}: No overfitting detected")
        
        print(f"\n🎯 CONCLUSION:")
        print(f"  Overfitting detected in {overfitting_count}/{total_tests} configurations")
        
        if overfitting_count > 0:
            print(f"  ⚠️  HYPOTHESIS CONFIRMED: Model exploits segmentation patterns")
            print(f"  💡 Recommendation: Use simpler memory mechanism or no memory")
        else:
            print(f"  ✅ HYPOTHESIS REJECTED: No systematic overfitting detected")
        
        print(f"\n📄 Detailed results saved to: {output_file}")


def main():
    """Main function to run overfitting hypothesis tests."""
    parser = argparse.ArgumentParser(description="Test architectural overfitting hypothesis")
    parser.add_argument("--output", type=str, default="overfitting_test_results.json",
                       help="Output file for results")
    args = parser.parse_args()
    
    # Base model configuration
    base_config = {
        'vocab_size': 128,
        'dim': 256,
        'n_heads': 8,
        'n_layers': 2,
        'n_mem': 4,
        'no_memory': False  # Test with memory to reproduce the bug
    }
    
    # Create tester and run all tests
    tester = OverfittingTester(base_config)
    results = tester.run_all_tests()
    tester.generate_report(results, args.output)


if __name__ == "__main__":
    main()
