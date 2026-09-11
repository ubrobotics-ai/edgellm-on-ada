# Cosmos3-Edge INT4 on an Ada desktop GPU

Running NVIDIA's [TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) and a
Cosmos3-Edge INT4-AWQ checkpoint on a **GeForce RTX 4060 Ti (Ada, SM89)** under WSL2 — a
configuration the published support matrix does not list.

It works. Measured 2026-09-11 on that card:

| | |
|---|---|
| decoder engine | 840 MiB, built in 23 s |
| vision encoder engine | 942 MiB |
| generation | **94.4 tokens/s**, 68.9 end to end including prefill |
| image tokens per frame | 494 |
| peak GPU memory | 14.6 GiB, sharing the card with other workloads |

Output on the project's own test image, unedited:

> In this captivating photograph, a red panda is seen lounging on a wooden platform. The
> platform is constructed from two light-colored wooden planks, with one plank positioned
> horizontally and the other placed vertically atop it.

## Why this repo exists

The Edge-LLM support matrix lists x86-64 as a developer tier covering SM80, SM86, SM100 and
SM120. **SM89 is absent**, which reads as "unsupported". It isn't: Ada is already in the
project's own default CUDA architecture list (`CMAKE_CUDA_ARCHITECTURES 80;86;89;90`), the
attention plugin compiles SM89 kernels by name, and TensorRT ships an SM89 builder resource.
The gap is that nobody documented the path, and there are four places where an x86 Ada build
stops for reasons whose error messages point somewhere else.

This repo is that path: one patch, the flags, and the order to do things in. It carries **no
NVIDIA source** — clone theirs, apply the patch. Both are Apache 2.0.

## The four things that stop you

**1. No TensorRT headers.** The build wants a TensorRT install with headers and the ONNX
parser. The pip wheels ship libraries but no headers, and the WSL CUDA apt repo carries no
TensorRT at all. You do not need a system install: take the headers from the
[TensorRT open-source repo](https://github.com/NVIDIA/TensorRT) at the tag matching your
wheels and point `TRT_PACKAGE_DIR` at a directory holding those plus symlinks to the wheel's
libraries. `scripts/make_trt_package.sh` does it.

**2. The system CUDA compiler is probably wrong.** CMake picks `nvcc` from `PATH`. On this
machine that was 11.5, which cannot target SM89 at all, while `CUDA_DIR` pointed at a newer
toolkit — a mismatch that configures cleanly and fails much later. Pass
`-DCMAKE_CUDA_COMPILER` explicitly.

**3. CUDA 12.8 is a real requirement, not a default.** The generated CuTe DSL headers use
`cudaLibrary_t`, which does not exist before 12.8. With 12.5 everything builds, both engines
build, and the failure arrives as `'cudaLibrary_t' does not name a type` only once you enable
the kernel group the vision tower needs.

**4. The runtime kernel compile has no include path.** This is the one real code change. The
attention plugin JIT-compiles its kernel through NVRTC and deliberately passes no `-I`, on the
grounds that every header is supplied as a virtual include. That holds for the project's own
headers; it does not hold for `cuda_fp16.h`, which includes `vector_types.h` from the toolkit,
and NVRTC on CUDA 12.x for x86 does not carry it as a builtin. The result is:

```
cuda_fp16.h(129): catastrophic error: cannot open source file "vector_types.h"
```

`patches/0001-nvrtc-cuda-include-path.patch` adds one include path, taken from
`EDGELLM_NVRTC_INCLUDE` when set. Ten lines.

## Which kernels you need

`ENABLE_CUTE_DSL=OFF` builds, and the **decoder engine builds and runs** without CuTe DSL,
because its attention path uses the XQA JIT. Two things then fail:

- the **vision encoder** will not build: `ViTAttentionPlugin` has only CuTe DSL backends, so
  with them off it refuses its configuration and the error says nothing about CuTe DSL;
- **inference** fails at execution inside a plugin, because `int4GroupwiseGemmPluginV2` also
  has CuTe DSL backends.

So build both groups: `fmha` and `int4_fp16_gemm`. There is no prebuilt x86 artifact; generate
it with the project's own script. Note the CMake list separator is a semicolon, and a
comma-separated list is silently rejected with "matched no groups".

## Recipe

```bash
scripts/00_prereqs.sh          # cmake, ninja, CUDA 12.8, the TensorRT wheels
scripts/10_fetch_sources.sh    # Edge-LLM at v0.10.1 + submodules, TensorRT OSS headers
scripts/20_make_trt_package.sh # assemble TRT_PACKAGE_DIR from headers + wheel libs
scripts/30_build.sh            # CuTe DSL artifact for sm_89, then the C++ build
scripts/40_build_engines.sh <checkpoint-dir> <engine-dir>
scripts/50_infer.sh <engine-dir> <image> "Describe this image."
```

Every script is short and does one thing; read them rather than trusting this list.

## Environment variables the runtime actually needs

| variable | why |
|---|---|
| `EDGELLM_PLUGIN_PATH` | the tools look for `build/libNvInfer_edgellm_plugin.so` relative to the working directory and say so only in an INFO line |
| `EDGELLM_NVRTC_INCLUDE` | the include path added by the patch |
| `LD_LIBRARY_PATH` | the TensorRT libraries, if they come from a wheel rather than `/usr` |

## What this does not tell you

Nothing here is an accuracy claim. The checkpoint's own card says its INT4 accuracy is
unvalidated, and running it on a desktop GPU does not change that. Nor are these numbers a
guide to Jetson performance: an Orin shares LPDDR5 between CPU and GPU and has no equivalent of
this card's memory bandwidth. What this establishes is narrower and still useful — that the
artefact can be executed, and therefore evaluated, without an Orin on the bench.

## Versions this was proven on

| | |
|---|---|
| GPU | GeForce RTX 4060 Ti 16 GB, SM89, driver 591.86 |
| OS | Ubuntu 22.04 on WSL2, Windows 11 |
| Edge-LLM | v0.10.1, commit `e8b2952` |
| TensorRT | 11.2.1.2, `tensorrt-cu12` wheels, headers from OSS tag v11.2 |
| CUDA | 12.8 toolkit |
| CuTe DSL | `nvidia-cutlass-dsl` 4.7.0, `cupy-cuda12x` 12.3.0 |
| checkpoint | `ubr-physical-ai/Cosmos3-Edge-INT4-AWQ`, W4A16 AWQ, group 128 |

## Licence

Apache 2.0, matching TensorRT Edge-LLM. The patch is a derivative of their file and carries
their copyright header; everything else here is ours.
