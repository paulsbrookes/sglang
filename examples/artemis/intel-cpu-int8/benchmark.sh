#!/usr/bin/env bash

# Fast Artemis benchmark. Writes artemis_results.json at repository root.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"
activate_workspace

test -d "$ARTEMIS_REPO_ROOT/.artemis-kernel" ||
  fail "candidate kernel is absent; run $HELPER_DIR/compile.sh first"

exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

rm -f -- "$ARTEMIS_REPO_ROOT/artemis_results.json" \
  "$ARTEMIS_REPO_ROOT/artemis_results.json.tmp"
python "$HELPER_DIR/benchmark.py"
