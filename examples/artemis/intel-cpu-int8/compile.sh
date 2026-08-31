#!/usr/bin/env bash

# Build or reuse the candidate checkout's Intel CPU kernel wheel.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

require_command ccache
require_command flock
require_command sha256sum
require_workspace
activate_workspace

mkdir -p "$ARTEMIS_WHEEL_CACHE" "$ARTEMIS_CCACHE_DIR"
exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

AOT_DIR="$ARTEMIS_REPO_ROOT/python/sglang/kernels/aot"
test -f "$AOT_DIR/pyproject_cpu.toml" ||
  fail "CPU kernel project is missing: $AOT_DIR/pyproject_cpu.toml"

content_hash="$(
  (
    cd "$ARTEMIS_REPO_ROOT"
    find python/sglang/kernels/aot/csrc -type f \
      \( -name '*.cpp' -o -name '*.h' -o -name '*.txt' \) -print0 |
      sort -z |
      xargs -0 sha256sum
    sha256sum python/sglang/kernels/aot/pyproject_cpu.toml
  ) | sha256sum | cut -d' ' -f1
)"
cache_dir="$ARTEMIS_WHEEL_CACHE/$content_hash"

shopt -s nullglob
cached_wheels=("$cache_dir"/*.whl)
shopt -u nullglob
if test "${#cached_wheels[@]}" -eq 0; then
  echo "CPU kernel cache miss: $content_hash"
  build_root="$(mktemp -d "${TMPDIR:-/tmp}/sglang-artemis-build.XXXXXX")"
  cleanup() {
    rm -rf -- "$build_root"
  }
  trap cleanup EXIT

  mkdir -p "$build_root/source" "$build_root/dist"
  cp -a "$AOT_DIR/." "$build_root/source/"
  cp "$AOT_DIR/pyproject_cpu.toml" "$build_root/source/pyproject.toml"

  export CMAKE_ARGS="-DCMAKE_CXX_COMPILER_LAUNCHER=ccache ${CMAKE_ARGS:-}"
  export CCACHE_BASEDIR="$ARTEMIS_REPO_ROOT"
  export CCACHE_NOHASHDIR=1
  (
    cd "$build_root/source"
    python -m pip wheel . --no-build-isolation --no-deps --wheel-dir "$build_root/dist"
  )

  mkdir -p "$cache_dir.tmp.$$"
  cp "$build_root/dist"/*.whl "$cache_dir.tmp.$$/"
  if ! mv "$cache_dir.tmp.$$" "$cache_dir" 2>/dev/null; then
    rm -rf -- "$cache_dir.tmp.$$"
  fi
  trap - EXIT
  cleanup
else
  echo "CPU kernel cache hit: $content_hash"
fi

shopt -s nullglob
cached_wheels=("$cache_dir"/*.whl)
shopt -u nullglob
test "${#cached_wheels[@]}" -eq 1 ||
  fail "expected exactly one cached wheel in $cache_dir"

kernel_dir="$ARTEMIS_REPO_ROOT/.artemis-kernel"
rm -rf -- "$kernel_dir"
mkdir -p "$kernel_dir"
python -m zipfile -e "${cached_wheels[0]}" "$kernel_dir"

python - "$kernel_dir" <<'PY'
import pathlib
import sys

kernel_dir = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(kernel_dir))
import sgl_kernel

loaded = pathlib.Path(sgl_kernel.__file__).resolve()
if kernel_dir not in loaded.parents:
    raise SystemExit(f"loaded {loaded}, expected a kernel under {kernel_dir}")
print(f"Candidate kernel ready: {loaded}")
PY
