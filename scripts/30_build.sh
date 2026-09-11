#!/usr/bin/env bash
# Build the CuTe DSL kernels for this GPU, then Edge-LLM itself.
#
# Both kernel groups are required, for reasons that only surface later if you skip them:
#   fmha            — the vision encoder's ViTAttentionPlugin has ONLY CuTe DSL backends, and
#                     without them it refuses its configuration in a message that never says so
#   int4_fp16_gemm  — int4GroupwiseGemmPluginV2 likewise, and there the failure arrives at
#                     INFERENCE time as a bare plugin assertion, long after everything built
#
# The decoder engine alone builds and runs without either, because its attention path is the
# XQA JIT. That is a trap: it looks like success.
set -euo pipefail

SRC="${SRC:-$HOME/src/tensorrt-edgellm}"
TRT_PKG="${TRT_PKG:-$HOME/src/trt-pkg}"
VENV="${VENV:-$HOME/ubr-trt/.venv}"
CUDA="${CUDA:-/usr/local/cuda-12.8}"
ARCH_TAG="${ARCH_TAG:-sm_89}"
JOBS="${JOBS:-$(nproc)}"

[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA — CUDA 12.8+ is required, see README §3" >&2; exit 1; }

echo "== CuTe DSL kernels for $ARCH_TAG (fmha + int4_fp16_gemm)"
( cd "$SRC"
  PATH="$CUDA/bin:/usr/bin:/bin" CUDA_HOME="$CUDA" \
    "$VENV/bin/python" kernelSrcs/build_cutedsl.py \
      --gpu_arch "$ARCH_TAG" --arch x86_64 --kernels fmha,int4_fp16_gemm --clean )

echo "== configure"
rm -rf "$SRC/build"; mkdir -p "$SRC/build"
( cd "$SRC/build"
  # NOTE the semicolon: ENABLE_CUTE_DSL is a CMake list. A comma-separated value is accepted
  # and then "matches no groups", which reads like the artifact is missing.
  cmake .. -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DTRT_PACKAGE_DIR="$TRT_PKG" \
    -DCUDA_CTK_VERSION="$(basename "$CUDA" | sed 's/cuda-//')" \
    -DCMAKE_CUDA_COMPILER="$CUDA/bin/nvcc" \
    "-DENABLE_CUTE_DSL=fmha;int4_fp16_gemm" \
    -DCUTE_DSL_ARTIFACT_TAG="$ARCH_TAG" )

echo "== build"
ninja -C "$SRC/build" -j "$JOBS"
ls -la "$SRC/build/libNvInfer_edgellm_plugin.so"
echo "done. Export EDGELLM_PLUGIN_PATH=$SRC/build/libNvInfer_edgellm_plugin.so before running anything."
