#!/usr/bin/env bash
# One image, one question. Run from the Edge-LLM source root if you use its bundled test images.
set -euo pipefail
ENG="${1:?usage: 50_infer.sh <engine-dir> <image> [prompt]}"
IMG="${2:?usage: 50_infer.sh <engine-dir> <image> [prompt]}"
PROMPT="${3:-Describe this image.}"
SRC="${SRC:-$HOME/src/tensorrt-edgellm}"; B="$SRC/build"
VENV="${VENV:-$HOME/ubr-trt/.venv}"; CUDA="${CUDA:-/usr/local/cuda-12.8}"
SITE="$("$VENV/bin/python" -c "import sysconfig; print(sysconfig.get_paths()[\"purelib\"])")"
export LD_LIBRARY_PATH="$SITE/tensorrt_libs:$B:${LD_LIBRARY_PATH:-}"
export EDGELLM_NVRTC_INCLUDE="$CUDA/include"
export EDGELLM_PLUGIN_PATH="$B/libNvInfer_edgellm_plugin.so"
IN=$(mktemp); OUT=$(mktemp)
cat > "$IN" <<JSON
{"batch_size":1,"temperature":0.0,"max_generate_length":${MAX_TOKENS:-64},
 "requests":[{"messages":[{"role":"user","content":[
   {"type":"image","image":"$IMG"},{"type":"text","text":"$PROMPT"}]}]}]}
JSON
"$B/examples/llm/llm_inference" --engineDir "$ENG" --multimodalEngineDir "$ENG" \
    --inputFile "$IN" --outputFile "$OUT" --dumpProfile
"$VENV/bin/python" -c "import json,sys; d=json.load(open(\"$OUT\")); r=(d[\"responses\"] if isinstance(d,dict) and \"responses\" in d else d)[0]; print(); print(r[\"output_text\"])"
