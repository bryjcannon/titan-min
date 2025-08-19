# Titan-Min

A minimal implementation of a Titan-style transformer model for the Needle-in-a-Haystack (NIAH) task. This repository provides a complete machine learning pipeline from training to deployment, featuring a novel architecture with memory prefix, depthwise separable convolutions, and advanced attention mechanisms.

## Project Summary

**Titan-Min** implements a compact transformer architecture with the following key components:

- **TitanBlock**: Core transformer block with:
  - **SiLU activation** on Q/K/V projections (paper-faithful)
  - **Per-head L2 normalization** of Q/K vectors for cosine attention
  - **Depthwise separable convolutions** after Q/K/V for enhanced feature processing
  - **Modern SwiGLU-style gating** with LayerNorm before output projection
  - **Temperature-scaled attention** with learnable temperature parameter
  - **Multi-head attention** with residual connections

- **TitanLongTermMemory**: Paper-faithful long-term memory module with:
  - **Surprise-driven updates** using gradient-based metrics
  - **Adaptive forgetting mechanism** with learnable gating parameter α_t
  - **Key-value mapping loss** for meta-learning (Equation 12)
  - **Direct query-based retrieval** (y_t = M*(q_t))
  - **Persistent memory** slots prepended to sequences
  - **Memory as a Context** architecture
  - **Online weight updates** during training and inference

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
│   │   └── niah.py              # Data loading and processing
│   ├── models/
│   │   ├── titan_min.py         # Main Titan model implementation
│   │   ├── titan_memory.py      # TitanLongTermMemory module (paper-faithful)
│   │   └── heads.py             # Model heads (classification, regression, etc.)
│   ├── utils/
│   │   └── checkpoint.py        # Model checkpointing utilities
│   ├── train.py                 # Basic training script
│   ├── train_titan_memory.py    # Enhanced training with TitanLongTermMemory
│   ├── export.py                # Model export utilities
│   ├── infer.py                 # Inference script
│   ├── eval.py                  # Evaluation script
│   ├── dump_test_preds.py       # Test prediction dumping
│   └── viz.py                   # Visualization utilities
├── run_pipeline.sh              # Complete ML pipeline runner
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore patterns
└── README.md                   # This file
```

## Quick Start

### Run Complete Pipeline

The easiest way to train and evaluate the Titan model is using the automated pipeline:

```bash
# Make the script executable
chmod +x run_pipeline.sh

# Run the complete pipeline
./run_pipeline.sh
```

This will:
1. **Train** the model with TitanLongTermMemory (4 epochs)
2. **Export** to TorchScript format
3. **Evaluate** on test set
4. **Generate** prediction CSV and visualizations
5. **Save** all artifacts to `artifacts/run1/`

### Pipeline Configuration

The pipeline uses these default parameters:
- **Dimensions**: 256
- **Attention heads**: 8
- **Layers**: 2
- **Memory slots**: 4
- **Memory dimension**: 128
- **Segment size**: 64
- **Persistent memory**: 8 slots
- **Surprise threshold**: 0.8

To modify parameters, edit the configuration section in `run_pipeline.sh`.

## Manual Usage

### Training with TitanLongTermMemory
```bash
python -m src.train_titan_memory \
    --out_dir artifacts/custom \
    --epochs 10 \
    --dim 512 \
    --memory_dim 256 \
    --segment_size 128
```

### Basic Training (without long-term memory)
```bash
python -m src.train \
    --out_dir artifacts/basic \
    --no_memory
```

### Inference
```bash
python -m src.infer --ckpt artifacts/run1/best.ckpt
```

### Evaluation
```bash
python -m src.eval --ckpt artifacts/run1/best.ckpt
```

## Dependencies

- torch>=2.2
- einops>=0.7
- numpy
- matplotlib
- tqdm
