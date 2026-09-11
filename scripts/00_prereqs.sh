#!/usr/bin/env bash
# Toolchain and Python packages. CUDA 12.8 is a hard requirement (README section 3).
set -euo pipefail
VENV="${VENV:-$HOME/ubr-trt/.venv}"
sudo apt-get update -qq
sudo apt-get install -y -qq cmake ninja-build build-essential git
sudo apt-get install -y -qq cuda-toolkit-12-8      # needs NVIDIAs CUDA apt repo configured
# python3-venv is often absent on WSL images; uv sidesteps it and is what this was proven with
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv "$VENV" --python 3.12
PIP_INDEX_URL=https://pypi.org/simple PIP_EXTRA_INDEX_URL=https://pypi.nvidia.com \
UV_INDEX_STRATEGY=unsafe-best-match \
uv pip install --python "$VENV/bin/python" \
    "tensorrt-cu12==11.2.1.2" "nvidia-cutlass-dsl==4.7.0" "cupy-cuda12x==12.3.0" onnx numpy
echo "prereqs done"
