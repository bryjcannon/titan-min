"""
Visualization script for TitanClassifier evaluation results.
"""

import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize TitanClassifier evaluation results")
    parser.add_argument("--report", type=str, required=True, help="Path to report.json file")
    return parser.parse_args()


def load_report(report_path):
    """Load evaluation report from JSON file."""
    report_path = Path(report_path)
    
    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    print(f"Loaded report from: {report_path}")
    print(f"Overall accuracy: {report['overall_acc']:.4f}")
    
    return report


def plot_accuracy_by_length(report):
    """Plot accuracy vs sequence length (line with markers)."""
    # Extract data
    lengths = [64, 128, 256]
    accuracies = [report['acc_by_L']['64'], report['acc_by_L']['128'], report['acc_by_L']['256']]
    
    # Create figure
    plt.figure(figsize=(8, 6))
    
    # Plot line with markers
    plt.plot(lengths, accuracies, marker='o', markersize=8, linewidth=2, color='blue')
    
    # Add value labels on points
    for length, acc in zip(lengths, accuracies):
        plt.annotate(f'{acc:.3f}', (length, acc), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Formatting
    plt.xlabel('Sequence Length', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Sequence Length', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)
    
    # Set x-axis ticks to show all lengths
    plt.xticks(lengths)
    
    # Add overall accuracy as horizontal line for reference
    overall_acc = report['overall_acc']
    plt.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'Overall: {overall_acc:.3f}')
    plt.legend()
    
    plt.tight_layout()
    print("Showing Plot 1: Accuracy vs Sequence Length")
    plt.show()


def plot_accuracy_by_position(report):
    """Plot accuracy vs position bins (bar chart)."""
    # Extract data
    position_bins = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    accuracies = [report['acc_by_pos_bin'][bin_name] for bin_name in position_bins]
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Create bar chart
    bars = plt.bar(position_bins, accuracies, color='skyblue', edgecolor='navy', alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.annotate(f'{acc:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom',
                    fontsize=10)
    
    # Formatting
    plt.xlabel('Position Bin (% of sequence length)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy vs Needle Position Bins', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim(0, 1.0)
    
    # Add overall accuracy as horizontal line for reference
    overall_acc = report['overall_acc']
    plt.axhline(y=overall_acc, color='red', linestyle='--', alpha=0.7, 
                label=f'Overall: {overall_acc:.3f}')
    plt.legend()
    
    plt.tight_layout()
    print("Showing Plot 2: Accuracy vs Position Bins")
    plt.show()


def main():
    args = parse_args()
    
    try:
        # Load report
        report = load_report(args.report)
        
        print(f"\nGenerating visualizations...")
        print(f"Total samples: {report['total_samples']}")
        
        # Plot 1: Accuracy vs sequence length
        plot_accuracy_by_length(report)
        
        # Plot 2: Accuracy vs position bins
        plot_accuracy_by_position(report)
        
        print("\nVisualization complete!")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        raise


if __name__ == "__main__":
    main()
