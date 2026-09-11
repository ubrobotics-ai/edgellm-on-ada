#!/usr/bin/env bash
# Edge-LLM at the tag the checkpoint was exported with. Submodules are NOT optional: a plain
# --depth 1 clone fails later during metadata generation.
set -euo pipefail
SRC="${SRC:-$HOME/src/tensorrt-edgellm}"
TAG="${TAG:-v0.10.1}"
mkdir -p "$(dirname "$SRC")"
[ -d "$SRC/.git" ] || git clone -q --branch "$TAG" --depth 1 https://github.com/NVIDIA/TensorRT-Edge-LLM.git "$SRC"
git -C "$SRC" submodule update --init --recursive
git -C "$SRC" apply --check "$(dirname "$0")/../patches/0001-nvrtc-cuda-include-path.patch" \
  && git -C "$SRC" apply "$(dirname "$0")/../patches/0001-nvrtc-cuda-include-path.patch" \
  && echo "patch applied" || echo "patch already applied (or does not fit this tag)"
