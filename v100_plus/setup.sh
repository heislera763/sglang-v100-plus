#!/usr/bin/env bash
# One-time, project-local build. The serving launcher performs no installations.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export SGLANG_BUILD_RUST_EXTS=none CUDA_VISIBLE_DEVICES=0,1,2,3
export XDG_CACHE_HOME="$PWD/.cache/runtime"
export SGLANG_CACHE_DIR="$PWD/.cache/sglang"
export TORCH_EXTENSIONS_DIR="$PWD/.cache/torch"
export SGLANG_JIT_CACHE_DIR="$PWD/.cache/jit"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.9}
export PATH="$PWD/.venv/bin:$CUDA_HOME/bin:$PATH"
export TRITON_PTXAS_PATH="$CUDA_HOME/bin/ptxas"
export TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 CMAKE_BUILD_PARALLEL_LEVEL=4 NVCC_THREADS=1
export UV_LOCK_TIMEOUT=7200
uv venv --python 3.12 .venv --allow-existing
uv pip install --python .venv/bin/python setuptools setuptools-scm setuptools-rust wheel
uv sync --locked --no-install-package sglang-kernel
torch_cmake_dir=$(.venv/bin/python -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")')
# Reconfigure against this environment rather than a stale Torch probe path.
rm -f "$PWD/artifacts/aot-build/CMakeCache.txt"
uv sync --locked --no-build-isolation \
  --config-settings-package "sglang-kernel:build-dir=$PWD/artifacts/aot-build" \
  --config-settings-package "sglang-kernel:cmake.define.Torch_DIR=$torch_cmake_dir"
marlin="$PWD/artifacts/marlin-v100"
marlin_revision=95b1e62e309fb6f58ea4d43391b539ed9f28a29f
if [[ -d $marlin/.git && $(git -C "$marlin" rev-parse HEAD) != "$marlin_revision" ]]; then
  # Preserve earlier builds and local edits; only the pinned checkout is active.
  previous="$PWD/artifacts/marlin-v100-previous-$(date -u +%Y%m%dT%H%M%S)"
  mv "$marlin" "$previous"
  echo "Archived previous Marlin checkout at $previous"
fi
if [[ ! -d $marlin/.git ]]; then
  git clone https://github.com/heislera763/marlin-v100.git "$marlin"
  git -C "$marlin" checkout --detach "$marlin_revision"
fi
[[ $(git -C "$marlin" rev-parse HEAD) == "$marlin_revision" ]]
for patch in "$PWD"/v100_plus/patches/marlin-v100-*.patch; do
  if ! git -C "$marlin" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$marlin" apply "$patch"
  fi
done
export CUTLASS_DIR=${CUTLASS_DIR:-$PWD/artifacts/cutlass}
if [[ ! -d $CUTLASS_DIR ]]; then
  git clone --depth 1 --branch v4.2.1 https://github.com/NVIDIA/cutlass.git "$CUTLASS_DIR"
fi
[[ -f $CUTLASS_DIR/include/cute/tensor.hpp && -f $CUTLASS_DIR/include/cutlass/cutlass.h ]]
# ERNIE stores routed experts as MXFP4 with E8M0 scales. The NVFP4-only
# profile omits that operator even when its FP8 option is enabled.
export CMAKE_ARGS="-DMARLIN_V100_NVFP4_ONLY=OFF -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_CUDA_FLAGS=-gencode=arch=compute_70,code=sm_70"
build_python="$PWD/.venv/bin/python"
(cd "$marlin" && "$build_python" setup.py build_ext --inplace)

# Canonical SM70 attention package; no vLLM runtime or dense compilation grid.
fa="$PWD/artifacts/flash-attention-v100"
fa_revision=531e92bd705b034705318a4641c9d6130cdc52a4
if [[ -d $fa/.git && $(git -C "$fa" rev-parse HEAD) != "$fa_revision" ]]; then
  previous="$PWD/artifacts/flash-attention-v100-previous-$(date -u +%Y%m%dT%H%M%S)"
  mv "$fa" "$previous"
  echo "Archived previous FA checkout at $previous"
fi
if [[ ! -d $fa/.git ]]; then
  git clone https://github.com/heislera763/flash-attention-v100.git "$fa"
  git -C "$fa" checkout --detach "$fa_revision"
fi
[[ $(git -C "$fa" rev-parse HEAD) == "$fa_revision" ]]
git -C "$fa" submodule update --init csrc/cutlass
CUDACXX="$CUDA_HOME/bin/nvcc" uv pip install --python .venv/bin/python \
  --no-build-isolation --config-settings="build-dir=$PWD/artifacts/fa-sparse-build" \
  "$fa/csrc/sparse_mla"
