#!/usr/bin/env bash

# Fast correctness gate: test the candidate wheel, without a model checkpoint.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"
activate_workspace

test -d "$ARTEMIS_REPO_ROOT/.artemis-kernel" ||
  fail "candidate kernel is absent; run $HELPER_DIR/compile.sh first"

exec 9>"$ARTEMIS_LOCK_FILE"
flock 9

python "$HELPER_DIR/parity.py"
