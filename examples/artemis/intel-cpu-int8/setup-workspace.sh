#!/usr/bin/env bash

# One-time, idempotent setup for an Artemis runner hosting this example.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

PYTHON_BIN="${PYTHON_BIN:-python3}"
cache_override=""
venv_override=""
while test "$#" -gt 0; do
  case "$1" in
    --cache-root)
      test "$#" -ge 2 || fail "--cache-root requires a path"
      cache_override="$2"
      shift 2
      ;;
    --venv)
      test "$#" -ge 2 || fail "--venv requires a path"
      venv_override="$2"
      shift 2
      ;;
    --python)
      test "$#" -ge 2 || fail "--python requires an executable"
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage: setup-workspace.sh [--cache-root PATH] [--venv PATH] [--python PYTHON]

Creates or reuses a persistent Python environment, compiler cache, and
content-addressed CPU-kernel wheel cache. Use a distinct cache root per
baseline or per concurrent Artemis worker.
EOF
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

# Re-source after argument overrides so all derived paths follow cache-root.
if test -n "$cache_override"; then
  ARTEMIS_CACHE_ROOT="$cache_override"
  export ARTEMIS_CACHE_ROOT
  unset ARTEMIS_VENV_MARKER ARTEMIS_WHEEL_CACHE ARTEMIS_CCACHE_DIR
  unset ARTEMIS_LOCK_FILE ARTEMIS_SEED_FILE
  if test -z "$venv_override"; then
    unset ARTEMIS_VENV_DIR
  fi
fi
if test -n "$venv_override"; then
  ARTEMIS_VENV_DIR="$venv_override"
  export ARTEMIS_VENV_DIR
fi
source "$HELPER_DIR/common.sh"

require_command "$PYTHON_BIN"
require_command ccache
require_command cmake
require_command flock
require_command g++
require_command git
require_command sha256sum
require_amx

case "$ARTEMIS_BUILD_JOBS" in
  ''|*[!0-9]*|0) fail "ARTEMIS_BUILD_JOBS must be a positive integer" ;;
esac

mkdir -p "$ARTEMIS_CACHE_ROOT" "$ARTEMIS_WHEEL_CACHE" "$ARTEMIS_CCACHE_DIR"
exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

seed_commit="$(git -C "$ARTEMIS_REPO_ROOT" rev-parse HEAD)"
if test -f "$ARTEMIS_SEED_FILE"; then
  recorded_seed="$(<"$ARTEMIS_SEED_FILE")"
  test "$recorded_seed" = "$seed_commit" || fail \
    "cache belongs to seed $recorded_seed, not $seed_commit; use a new ARTEMIS_CACHE_ROOT"
fi

if test ! -x "$ARTEMIS_VENV_DIR/bin/python"; then
  echo "Creating persistent environment: $ARTEMIS_VENV_DIR"
  "$PYTHON_BIN" -m venv "$ARTEMIS_VENV_DIR"
  "$ARTEMIS_VENV_DIR/bin/python" -m pip install --upgrade pip wheel
  "$ARTEMIS_VENV_DIR/bin/python" -m pip install \
    "scikit-build-core>=0.10" ninja pytest
  "$ARTEMIS_VENV_DIR/bin/python" -m pip install \
    --index-url https://download.pytorch.org/whl/cpu "torch==2.12.0"
fi

"$ARTEMIS_VENV_DIR/bin/python" - <<'PY'
import importlib.util
import sys

required = ("torch", "pytest", "scikit_build_core")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    sys.exit(f"workspace environment lacks: {', '.join(missing)}")

import torch

if torch.__version__.split("+", 1)[0] != "2.12.0":
    sys.exit(f"expected torch 2.12.0, found {torch.__version__}")
print(f"Python environment ready: torch {torch.__version__}")
PY

printf '%s\n' "$ARTEMIS_VENV_DIR" >"$ARTEMIS_VENV_MARKER"
printf '%s\n' "$seed_commit" >"$ARTEMIS_SEED_FILE"

# compile.sh owns this same lock. Release it before invoking the helper.
flock -u 9
exec 9>&-

echo "Warming CPU-kernel build cache."
"$HELPER_DIR/compile.sh"
echo "Workspace ready: $ARTEMIS_CACHE_ROOT"
