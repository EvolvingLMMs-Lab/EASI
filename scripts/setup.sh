#!/usr/bin/env bash
# Setup script for EASI evaluation environment.
# Run from the repo root: bash scripts/setup.sh
#
# Prerequisites:
#   - Python 3.11
#   - uv (pip install uv, or: curl -LsSf https://astral.sh/uv/install.sh | sh)
#   - CUDA toolkit (for flash-attn and torch)
#
# What this does:
#   1. Initializes the VLMEvalKit submodule
#   2. Creates a Python 3.11 venv
#   3. Installs VLMEvalKit + pinned dependencies
#   4. Installs flash-attn (optional but recommended)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "=== EASI Evaluation Setup ==="
echo "Repo root: $REPO_ROOT"
echo ""

# ---- Step 1: Initialize submodules ----
echo "[1/4] Initializing submodules..."
git submodule update --init --recursive
echo ""

# ---- Step 2: Create venv ----
echo "[2/4] Creating Python 3.11 virtual environment..."
if [ -d ".venv" ]; then
    echo "  .venv already exists, skipping creation"
else
    uv venv -p 3.11
fi
echo ""

# Activate venv for the rest of the script
# shellcheck disable=SC1091
source .venv/bin/activate

# ---- Step 3: Install VLMEvalKit ----
echo "[3/4] Installing VLMEvalKit and dependencies..."
cd VLMEvalKit
uv pip install .
cd "$REPO_ROOT"

# Pin specific versions (overrides VLMEvalKit's unpinned defaults)
uv pip install "torch==2.7.1" "torchvision==0.22.1" "setuptools<81" "transformers>=4.45,<5"

# Additional dependencies for the eval/submission scripts
uv pip install requests
echo ""

# ---- Step 4: Install flash-attn (optional) ----
echo "[4/4] Installing flash-attn (this may take a few minutes)..."
if uv pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "  flash-attn installed successfully"
else
    echo "  WARNING: flash-attn installation failed (CUDA toolkit may be missing)"
    echo "  Evaluation will still work but may be slower for some models"
fi
echo ""

# ---- Done ----
echo "=== Setup complete ==="
echo ""
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "To run EASI-8 evaluation:"
echo "  bash scripts/vlmevalkit_submit.sh"
