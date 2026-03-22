#!/bin/bash
# setup_env.sh — Chameleon node environment setup for CRAG project
# Run once per node: bash setup_env.sh
# Save this script to ~/ so it persists across sessions.

set -e

CONDA_DIR="$HOME/miniconda"
ENV_NAME="crag"

# ── Miniconda ────────────────────────────────────────────────────────────────

if [ -d "$CONDA_DIR" ]; then
    echo "[INFO] Miniconda already installed at $CONDA_DIR — skipping install."
else
    echo "[INFO] Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm /tmp/miniconda.sh
    echo "[INFO] Miniconda installed."
fi

source "$CONDA_DIR/bin/activate"

# ── Conda environment ────────────────────────────────────────────────────────

if conda env list | grep -q "^$ENV_NAME "; then
    echo "[INFO] Conda env '$ENV_NAME' already exists — skipping creation."
else
    echo "[INFO] Creating conda env '$ENV_NAME' (Python 3.10)..."
    conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"

# ── Python packages ──────────────────────────────────────────────────────────

echo "[INFO] Installing Python packages..."
pip install --quiet \
    torch \
    faiss-gpu \
    sentence-transformers \
    langgraph \
    datasets \
    ragas \
    transformers \
    accelerate

# ── Verify GPU ───────────────────────────────────────────────────────────────

echo ""
echo "[INFO] GPU check:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
    || echo "[WARN] nvidia-smi not available."

echo ""
echo "[INFO] Python package versions:"
python -c "
import torch, faiss, sentence_transformers, langgraph, datasets
print(f'  torch              {torch.__version__}  (CUDA: {torch.cuda.is_available()})')
print(f'  faiss              {faiss.__version__}')
print(f'  sentence-transformers {sentence_transformers.__version__}')
print(f'  datasets           {datasets.__version__}')
"

# ── Done ─────────────────────────────────────────────────────────────────────

echo ""
echo "[DONE] Environment ready."
echo "       Activate with: conda activate $ENV_NAME"
