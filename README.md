# Titan-Min

A minimal implementation of a Titan-style transformer model for the Needle-in-a-Haystack (NIAH) task. This repository provides a complete machine learning pipeline from training to deployment, featuring a novel architecture with memory prefix, depthwise separable convolutions, and advanced attention mechanisms.

## Project Summary

**Titan-Min** implements a compact transformer architecture with the following key components:

- **TitanBlock**: Core transformer block with:
  - **SiLU activation** on Q/K/V projections
  - **L2 normalization** of Q/K vectors for stable attention
  - **Depthwise separable convolutions** after Q/K/V for enhanced feature processing
  - **Gating mechanism** with LayerNorm and sigmoid gating
  - **Multi-head attention** with residual connections

- **MemoryPrefix**: Learnable memory slots with EMA (Exponential Moving Average) writeback
- **Position prediction**: Specialized head for needle position detection
- **Ablation support**: Comprehensive toggles for architecture components

## Dataset Specification

**Needle-in-a-Haystack (NIAH) Dataset**:
- **Task**: Locate a special "needle" token (127) within random token sequences
- **Vocabulary**: 128 tokens (0-127), where 0 is PAD and 127 is NEEDLE
- **Sequence lengths**: Variable (64, 128, 256 tokens)
- **Total samples**: 10,000
- **Splits**: 
  - **Training**: 8,000 samples
  - **Validation**: 1,000 samples  
  - **Test**: 1,000 samples
- **Reproducibility**: Fixed random seed (42) for consistent splits

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd titan-min
python -m pip install -r requirements.txt
```

## Project Structure

```
titan-min/
├── src/
│   ├── data/
│   │   └── niah.py          # Data loading and processing
│   ├── models/
│   │   ├── titan_min.py     # Main Titan model implementation
│   │   └── heads.py         # Model heads (classification, regression, etc.)
│   ├── utils/
│   │   └── checkpoint.py    # Model checkpointing utilities
│   ├── train.py             # Training script
│   ├── export.py            # Model export utilities
│   ├── infer.py             # Inference script
│   ├── eval.py              # Evaluation script
│   ├── dump_test_preds.py   # Test prediction dumping
│   └── viz.py               # Visualization utilities
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore patterns
└── README.md               # This file
```

## Usage

### Training
```bash
python src/train.py
```

### Inference
```bash
python src/infer.py
```

### Evaluation
```bash
python src/eval.py
```

## Dependencies

- torch>=2.2
- einops>=0.7
- numpy
- matplotlib
- tqdm
