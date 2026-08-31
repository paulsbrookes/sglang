#!/usr/bin/env bash

# Optional end-to-end confirmation for a microbenchmark champion.
# This is deliberately not the default Artemis benchmark.

set -euo pipefail

HELPER_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$HELPER_DIR/common.sh"

: "${ARTEMIS_SERVING_VENV:?set ARTEMIS_SERVING_VENV to a full SGLang CPU environment}"
: "${SGLANG_MODEL_PATH:?set SGLANG_MODEL_PATH to a compatible local checkpoint}"
test -x "$ARTEMIS_SERVING_VENV/bin/python" ||
  fail "Python environment not found: $ARTEMIS_SERVING_VENV"
test -e "$SGLANG_MODEL_PATH/config.json" ||
  fail "checkpoint config not found: $SGLANG_MODEL_PATH/config.json"
test -d "$ARTEMIS_REPO_ROOT/.artemis-kernel" ||
  fail "candidate kernel is absent; run $HELPER_DIR/compile.sh first"

# shellcheck source=/dev/null
source "$ARTEMIS_SERVING_VENV/bin/activate"
export PYTHONPATH="$ARTEMIS_REPO_ROOT/.artemis-kernel:$ARTEMIS_REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_USE_CPU_ENGINE=1
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/lib/x86_64-linux-gnu}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

port="${SGLANG_PORT:-30000}"
server_log="$ARTEMIS_CACHE_ROOT/serving-server.log"
benchmark_jsonl="$ARTEMIS_CACHE_ROOT/serving-benchmark.jsonl"
eval_log="$ARTEMIS_CACHE_ROOT/serving-eval.log"
mkdir -p "$ARTEMIS_CACHE_ROOT"
rm -f -- "$server_log" "$benchmark_jsonl" "$eval_log"

serve_args=(
  --model-path "$SGLANG_MODEL_PATH"
  --trust-remote-code
  --disable-overlap-schedule
  --device cpu
  --host 127.0.0.1
  --port "$port"
  --tp 1
)
if test -n "${SGLANG_QUANTIZATION:-}"; then
  serve_args+=(--quantization "$SGLANG_QUANTIZATION")
fi
if test -n "${SGLANG_EXTRA_SERVE_FLAGS:-}"; then
  read -r -a extra_flags <<<"$SGLANG_EXTRA_SERVE_FLAGS"
  serve_args+=("${extra_flags[@]}")
fi

"$ARTEMIS_SERVING_VENV/bin/sglang" serve "${serve_args[@]}" >"$server_log" 2>&1 &
server_pid=$!
cleanup() {
  kill "$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 240); do
  if curl -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -n 80 "$server_log" >&2
    fail "SGLang server exited before becoming ready"
  fi
  sleep 5
done
curl -fsS --max-time 5 "http://127.0.0.1:$port/health" >/dev/null ||
  fail "SGLang server did not become ready; see $server_log"

for repeat in 1 2; do
  python -m sglang.benchmark.serving \
    --backend sglang --host 127.0.0.1 --port "$port" \
    --dataset-name random --random-range-ratio 1.0 --seed 42 \
    --warmup-requests 2 --flush-cache \
    --random-input-len 128 --random-output-len 256 \
    --num-prompts 8 --max-concurrency 4 --request-rate inf \
    --output-file "$benchmark_jsonl" --tag "c4_r$repeat" --disable-tqdm
done

python -m sglang.test.run_eval \
  --port "$port" --eval-name gsm8k \
  --num-examples "${SGLANG_EVAL_EXAMPLES:-50}" 2>&1 | tee "$eval_log"

python - "$benchmark_jsonl" "$eval_log" \
  "$ARTEMIS_REPO_ROOT/artemis_serving_results.json" <<'PY'
import json
import re
import statistics
import sys

benchmark_path, eval_path, output_path = sys.argv[1:]
rows = [json.loads(line) for line in open(benchmark_path, encoding="utf-8")]
throughputs = [float(row["output_throughput"]) for row in rows[-2:]]
if len(throughputs) != 2:
    raise SystemExit("expected two serving benchmark rows")

eval_text = open(eval_path, encoding="utf-8").read()
matches = re.findall(r"'score': (?:np\.float64\()?([0-9.]+)", eval_text)
if not matches:
    raise SystemExit("could not parse the SGLang evaluation score")
score = float(matches[-1])
if score < 0.94:
    raise SystemExit(f"gsm8k correctness gate failed: {score} < 0.94")

metrics = {
    "serving_decode_tok_s_c4": round(statistics.median(throughputs), 2),
    "serving_gsm8k_score": score,
}
with open(output_path, "w", encoding="utf-8") as output:
    json.dump(metrics, output, indent=2, sort_keys=True)
    output.write("\n")
print(json.dumps(metrics, sort_keys=True))
PY

echo "Optional serving confirmation passed."
