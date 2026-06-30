#!/bin/bash
# Setup for the deepseek_v4_mini project (dual-stream thought memory).
# Creates the `diffusion-thought` conda env and installs the runtime deps.

set -e

echo "🚀 Setting up environment for deepseek_v4_mini"
echo "=============================================="

if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Install Anaconda/Miniconda first."
    exit 1
fi

# Conda env (name kept as 'diffusion-thought' for continuity with existing tooling)
echo "📦 Creating conda env 'diffusion-thought' (python 3.10)..."
conda create -n diffusion-thought python=3.10 -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate diffusion-thought

# PyTorch + CUDA (cu118 is the tested build for the RTX 3090 / WSL2 setup)
echo "🔥 Installing PyTorch (CUDA 11.8)..."
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Project dependencies
echo "📚 Installing project dependencies..."
pip install -r requirements.txt

# Verify
echo "🧪 Verifying installation..."
python - <<'PY'
import torch, transformers, datasets
print(f"PyTorch       {torch.__version__}")
print(f"Transformers  {transformers.__version__}")
print(f"Datasets      {datasets.__version__}")
if torch.cuda.is_available():
    print(f"✅ CUDA {torch.version.cuda} — {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    print("⚠️  CUDA not available — training will run on CPU (slow).")
from torch.utils.tensorboard import SummaryWriter  # train.py needs this
print("✅ tensorboard backend OK")
PY

echo ""
echo "🎉 Setup complete."
echo ""
echo "📋 Quick start:"
echo "  conda activate diffusion-thought"
echo "  export PYTHONPATH=\$PWD"
echo "  python -m deepseek_v4_mini.train deepseek_v4_mini/configs/tiny.yaml"
echo ""
echo "  Memory experiments:"
echo "    configs/code.yaml          # bank reset per sequence (baseline)"
echo "    configs/code_persist.yaml  # bank persists across sequences"
echo "    configs/synth_recall.yaml  # addressable-recall diagnostic"
echo "    configs/gist.yaml          # latent-context (gist) diagnostic"
