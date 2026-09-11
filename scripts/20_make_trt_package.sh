#!/usr/bin/env bash
# Assemble a TRT_PACKAGE_DIR that CMake's FindTensorRT will accept, without a system install.
#
# The finder wants NvInfer.h under include/ and libnvinfer + libnvonnxparser under lib/. The pip
# wheels give you the libraries and no headers; the TensorRT open-source repo gives you the
# headers and no libraries. Put the two together and point the build at the result.
#
# MATCH THE VERSIONS. Headers from OSS tag vX.Y against libraries from a different minor is
# asking for a link-time surprise, so this pins both.
#
#   ./20_make_trt_package.sh [trt-version] [venv] [outdir]
set -euo pipefail

TRT_TAG="${1:-v11.2}"
VENV="${2:-$HOME/ubr-trt/.venv}"
OUT="${3:-$HOME/src/trt-pkg}"
OSS_DIR="${OSS_DIR:-$HOME/src/trt-oss}"

SITE="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
LIBS="$SITE/tensorrt_libs"
[ -d "$LIBS" ] || { echo "no tensorrt_libs under $SITE — install tensorrt-cu12 first" >&2; exit 1; }

if [ ! -f "$OSS_DIR/include/NvInfer.h" ]; then
  echo "== fetching TensorRT headers at $TRT_TAG"
  rm -rf "$OSS_DIR"
  git clone -q --depth 1 --branch "$TRT_TAG" --filter=blob:none --sparse \
      https://github.com/NVIDIA/TensorRT.git "$OSS_DIR"
  git -C "$OSS_DIR" sparse-checkout set include
fi
# A header of zero bytes is what a full disk leaves behind, and it fails much later as a
# hundred "'nvinfer1' has not been declared" errors. Check here instead.
[ -s "$OSS_DIR/include/NvInfer.h" ] || { echo "NvInfer.h is empty — refetch" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/include" "$OUT/lib"
cp -r "$OSS_DIR/include/." "$OUT/include/"
n=0
for so in "$LIBS"/*.so.*; do
  case "$so" in *builder_resource*) continue;; esac      # data blobs, not link targets
  b="$(basename "$so")"
  ln -sf "$so" "$OUT/lib/$b"
  ln -sf "$so" "$OUT/lib/${b%.*}"                        # plain .so name for the linker
  n=$((n + 1))
done
echo "linked $n libraries, $(ls "$OUT"/include/*.h | wc -l) headers -> $OUT"

cat > /tmp/trt_probe.cpp <<'EOF'
#include <NvInfer.h>
#include <NvOnnxParser.h>
int main() { return (int) (sizeof(nvinfer1::ILogger) > 0); }
EOF
CUDA_INC="${CUDA_INC:-/usr/local/cuda-12.8/include}"
g++ -std=gnu++17 -isystem "$OUT/include" -isystem "$CUDA_INC" -c /tmp/trt_probe.cpp -o /tmp/trt_probe.o
echo "verified: the headers compile and declare nvinfer1"
