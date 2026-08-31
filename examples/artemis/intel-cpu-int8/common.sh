#!/usr/bin/env bash

# Shared, machine-portable paths for the Intel CPU INT8 Artemis example.

set -euo pipefail

ARTEMIS_HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ARTEMIS_REPO_ROOT="$(cd -- "$ARTEMIS_HELPER_DIR/../../.." && pwd)"

ARTEMIS_CACHE_ROOT="${ARTEMIS_CACHE_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/artemis/sglang-intel-cpu-int8}"
ARTEMIS_VENV_MARKER="${ARTEMIS_VENV_MARKER:-$ARTEMIS_CACHE_ROOT/venv-path}"
if test -z "${ARTEMIS_VENV_DIR+x}" && test -s "$ARTEMIS_VENV_MARKER"; then
  ARTEMIS_VENV_DIR="$(<"$ARTEMIS_VENV_MARKER")"
fi
ARTEMIS_VENV_DIR="${ARTEMIS_VENV_DIR:-$ARTEMIS_CACHE_ROOT/venv}"
ARTEMIS_WHEEL_CACHE="${ARTEMIS_WHEEL_CACHE:-$ARTEMIS_CACHE_ROOT/wheels}"
ARTEMIS_CCACHE_DIR="${ARTEMIS_CCACHE_DIR:-$ARTEMIS_CACHE_ROOT/ccache}"
ARTEMIS_LOCK_FILE="${ARTEMIS_LOCK_FILE:-$ARTEMIS_CACHE_ROOT/workspace.lock}"
ARTEMIS_SEED_FILE="${ARTEMIS_SEED_FILE:-$ARTEMIS_CACHE_ROOT/seed-commit}"
ARTEMIS_BUILD_JOBS="${ARTEMIS_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "required command not found: $1"
}

require_workspace() {
  test -x "$ARTEMIS_VENV_DIR/bin/python" ||
    fail "workspace is not initialized; run $ARTEMIS_HELPER_DIR/setup-workspace.sh"
  test -f "$ARTEMIS_SEED_FILE" ||
    fail "workspace seed marker is missing: $ARTEMIS_SEED_FILE"
}

activate_workspace() {
  require_workspace
  # shellcheck source=/dev/null
  source "$ARTEMIS_VENV_DIR/bin/activate"
  export CCACHE_DIR="$ARTEMIS_CCACHE_DIR"
  export PYTHONPATH="$ARTEMIS_REPO_ROOT/.artemis-kernel:$ARTEMIS_REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
}

require_amx() {
  test "$(uname -m)" = "x86_64" || fail "this example requires an x86_64 Intel CPU"
  for flag in amx_tile amx_int8 avx512_vnni; do
    if ! awk -v flag="$flag" \
      '/^flags/ { for (i = 1; i <= NF; i++) if ($i == flag) found = 1 } END { exit !found }' \
      /proc/cpuinfo; then
      fail "CPU lacks required flag: $flag"
    fi
  done
}
