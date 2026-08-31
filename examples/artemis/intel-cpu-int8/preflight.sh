#!/usr/bin/env bash

# Validate the complete fast onboarding loop before creating Discovery.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"
require_workspace

echo "== compile candidate kernel =="
"$HELPER_DIR/compile.sh"

echo "== focused numerical gate =="
"$HELPER_DIR/test.sh"

mkdir -p "$ARTEMIS_CACHE_ROOT/preflight"
first="$ARTEMIS_CACHE_ROOT/preflight/run-1.json"
second="$ARTEMIS_CACHE_ROOT/preflight/run-2.json"

echo "== benchmark repetition 1 =="
"$HELPER_DIR/benchmark.sh"
cp "$ARTEMIS_REPO_ROOT/artemis_results.json" "$first"

echo "== benchmark repetition 2 =="
"$HELPER_DIR/benchmark.sh"
cp "$ARTEMIS_REPO_ROOT/artemis_results.json" "$second"

python3 - "$HELPER_DIR/metrics-schema.json" "$first" "$second" <<'PY'
import json
import math
import os
import sys

schema_path, first_path, second_path = sys.argv[1:]
schema = json.load(open(schema_path, encoding="utf-8"))
first = json.load(open(first_path, encoding="utf-8"))
second = json.load(open(second_path, encoding="utf-8"))

expected = {item["name"] for item in schema}
if len(expected) != len(schema):
    raise SystemExit("metrics schema contains duplicate names")
if any(item.get("source") != "worker" for item in schema):
    raise SystemExit("all metrics must use source=worker")
if abs(sum(item["importance"] for item in schema) - 1.0) > 1e-9:
    raise SystemExit("metric importance values must sum to 1")

for label, metrics in (("run 1", first), ("run 2", second)):
    if set(metrics) != expected:
        raise SystemExit(
            f"{label} metric names differ from schema: "
            f"missing={sorted(expected - set(metrics))}, "
            f"extra={sorted(set(metrics) - expected)}"
        )
    for name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise SystemExit(f"{label} has invalid metric {name}={value!r}")

limit = float(os.environ.get("ARTEMIS_PREFLIGHT_MAX_DRIFT", "0.15"))
for name in sorted(expected):
    relative_drift = abs(first[name] - second[name]) / max(first[name], second[name])
    print(
        f"{name}: run1={first[name]:.2f}, run2={second[name]:.2f}, "
        f"drift={relative_drift:.1%}"
    )
    if relative_drift > limit:
        raise SystemExit(
            f"{name} drift {relative_drift:.1%} exceeds {limit:.1%}; "
            "rerun on an idle, exclusive runner"
        )

print("Preflight passed: build, parity, metric contract, and noise checks are ready.")
PY
