#!/usr/bin/env bash
# Build both engines. The decoder alone is not enough for a vision-language model.
set -euo pipefail
CKPT="${1:?usage: 40_build_engines.sh <checkpoint-dir> <engine-dir>}"
ENG="${2:?usage: 40_build_engines.sh <checkpoint-dir> <engine-dir>}"
SRC="${SRC:-$HOME/src/tensorrt-edgellm}"; B="$SRC/build"
VENV="${VENV:-$HOME/ubr-trt/.venv}"; CUDA="${CUDA:-/usr/local/cuda-12.8}"
SITE="$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")"
export LD_LIBRARY_PATH="$SITE/tensorrt_libs:$B:${LD_LIBRARY_PATH:-}"
export EDGELLM_NVRTC_INCLUDE="$CUDA/include"
export EDGELLM_PLUGIN_PATH="$B/libNvInfer_edgellm_plugin.so"
mkdir -p "$ENG"
# --onnxDir takes a COMPONENT directory (the one holding config.json), not the onnx/ root
"$B/examples/llm/llm_build" --onnxDir "$CKPT/onnx/llm" --engineDir "$ENG" \
    --maxInputLen "${MAX_INPUT_LEN:-1024}" --maxKVCacheCapacity "${MAX_KV:-4096}" --maxBatchSize "${BS:-1}"
"$B/examples/multimodal/visual_build" --onnxDir "$CKPT/onnx/visual" --engineDir "$ENG"
du -sh "$ENG/llm.engine" "$ENG/visual/visual.engine"
