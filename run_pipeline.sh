#!/bin/bash

# Titan-Min ML Pipeline Runner
# This script runs the complete titan-min machine learning pipeline
# from training through inference.

set -e  # Exit on any error

echo "🚀 Starting Titan-Min ML Pipeline"
echo "================================="

# Configuration
EPOCHS=4
DIM=256
HEADS=8
LAYERS=2
N_MEM=4
OUT_DIR="artifacts/run1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}📋 Step: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    print_error "Virtual environment .venv not found!"
    echo "Please create it first with: python -m venv .venv"
    exit 1
fi

# Activate virtual environment
print_step "Activating virtual environment"
source .venv/bin/activate
print_success "Virtual environment activated"

# Verify dependencies
print_step "Checking dependencies"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')" || {
    print_error "PyTorch not found. Please install dependencies:"
    echo "pip install -r requirements.txt"
    exit 1
}
print_success "Dependencies verified"

# Verify project structure
print_step "Verifying project structure"
python -c "
import os
import sys

# Check if all required files exist
required_files = [
    'requirements.txt',
    'README.md',
    '.gitignore',
    'src/__init__.py',
    'src/data/__init__.py',
    'src/data/niah.py',
    'src/models/__init__.py',
    'src/models/titan_min.py',
    'src/models/heads.py',
    'src/utils/__init__.py',
    'src/utils/checkpoint.py',
    'src/train.py',
    'src/export.py',
    'src/infer.py',
    'src/eval.py',
    'src/dump_test_preds.py',
    'src/viz.py'
]

missing_files = []
for file in required_files:
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    print(f'✗ Missing files: {missing_files}')
    sys.exit(1)
else:
    print('✓ All required files present')
"
print_success "Project structure verified"

# Step 1: Training
print_step "Training model (${EPOCHS} epochs)"
echo "Configuration:"
echo "  - Dimensions: ${DIM}"
echo "  - Attention heads: ${HEADS}"
echo "  - Layers: ${LAYERS}"
echo "  - Memory slots: ${N_MEM}"
echo "  - Output directory: ${OUT_DIR}"
echo ""

python -m src.train \
    --out_dir ${OUT_DIR} \
    --epochs ${EPOCHS} \
    --dim ${DIM} \
    --heads ${HEADS} \
    --layers ${LAYERS} \
    --n_mem ${N_MEM}

print_success "Training completed"

# Step 2: Export to TorchScript
print_step "Exporting model to TorchScript"
python -m src.export --ckpt ${OUT_DIR}/best.ckpt
print_success "Model exported to ${OUT_DIR}/model_scripted.pt"

# Step 3: Evaluation
print_step "Evaluating model on test set"
python -m src.eval --ckpt ${OUT_DIR}/best.ckpt
print_success "Evaluation completed - results saved to ${OUT_DIR}/report.json"

# Step 4: Dump test predictions
print_step "Generating test predictions CSV"
python -m src.dump_test_preds --ckpt ${OUT_DIR}/best.ckpt
print_success "Predictions saved to ${OUT_DIR}/predictions.csv"

# Step 5: Visualization
print_step "Generating visualizations"
python -m src.viz --report ${OUT_DIR}/report.json
print_success "Visualizations generated"

# Step 6: Sample inference
print_step "Running sample inference"
echo "Testing with sequence: [1,2,3,127,4,5,6] (needle token 127 at position 3)"
python -m src.infer --ckpt ${OUT_DIR}/best.ckpt --seq "1,2,3,127,4,5,6"
print_success "Inference completed"

# Summary
echo ""
echo "🎉 Pipeline completed successfully!"
echo "=================================="
echo "Generated artifacts in ${OUT_DIR}:"
echo "  📁 Checkpoints:"
echo "    - best.ckpt (best validation model)"
echo "    - last.ckpt (final epoch model)"
echo "  📊 Configuration & Metrics:"
echo "    - config.json (training configuration)"
echo "    - metrics.json (training metrics)"
echo "  🚀 Deployment:"
echo "    - model_scripted.pt (TorchScript model)"
echo "    - export_manifest.json (export metadata)"
echo "  📈 Evaluation:"
echo "    - report.json (test set evaluation)"
echo "    - predictions.csv (test predictions)"
echo ""
echo "🔧 Usage examples:"
echo "  # Run with different configuration:"
echo "  ./run_pipeline.sh"
echo ""
echo "  # Run individual components:"
echo "  source .venv/bin/activate"
echo "  python -m src.train --help"
echo "  python -m src.infer --ckpt ${OUT_DIR}/best.ckpt --seq \"1,2,3,127,4,5,6\""
echo ""
echo "  # Test with ablations:"
echo "  python -m src.train --out_dir artifacts/no_memory --no_memory --epochs 2"
echo "  python -m src.train --out_dir artifacts/no_dsconv --no_dsconv --epochs 2"
echo "  python -m src.train --out_dir artifacts/relu --act relu --epochs 2"
