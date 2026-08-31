# SGLang INT8 CPU optimization with Artemis

This example gives Intel SGLang developers a short, reproducible first
Discovery. It optimizes the W8A8 INT8 GEMM used by SGLang on an AMX-capable
Xeon CPU.

The default loop is checkpoint-free:

1. Build the candidate's CPU kernel.
2. Check numerical parity at decode batch sizes 1, 4, and 32.
3. Measure cache-honest INT8 weight-streaming throughput.
4. Write three worker metrics to `artemis_results.json`.

The benchmark should complete well within three minutes after one-time machine
setup. It is a screening benchmark, not proof of end-to-end serving
performance.

## Requirements

- Linux x86_64 on an Intel CPU with AMX INT8 and AVX-512 VNNI.
- Python 3.10 or newer.
- `ccache`, CMake, a C++ compiler, Git, and `flock`.
- Enough free RAM for 1.5 GB of rotating weights plus build overhead.
- An Artemis runner installed on the same machine.

On Debian or Ubuntu, the native tools can be installed with:

```bash
sudo apt-get install build-essential ccache cmake git util-linux
```

## One-time persistent workspace

From the repository root:

```bash
./examples/artemis/intel-cpu-int8/setup-workspace.sh
```

The default machine-local workspace is:

```text
~/.cache/artemis/sglang-intel-cpu-int8/
├── venv/             persistent Python and PyTorch environment
├── wheels/           CPU-kernel wheels keyed by source content
├── ccache/           compiler object cache
├── preflight/        the two latest preflight measurements
├── seed-commit       baseline revision guard
└── workspace.lock    prevents concurrent mutation
```

Use a different location if required:

```bash
ARTEMIS_CACHE_ROOT=/mnt/fast-cache/sglang-int8 \
  ./examples/artemis/intel-cpu-int8/setup-workspace.sh
```

Use the same `ARTEMIS_CACHE_ROOT` in every Artemis command. Give concurrent
workers separate cache roots. The setup records the exact seed commit and
refuses to reuse that workspace for a different baseline.

Artemis evaluates versions in temporary Git checkouts. `compile.sh` hashes the
candidate CPU-kernel sources, reuses an existing wheel when available, and
uses persistent `ccache` objects on a miss. The resulting wheel is unpacked
under that candidate checkout's `.artemis-kernel/`; the shared environment is
never overwritten. This combines safe candidate isolation with incremental
build performance.

## Verify the harness before onboarding

Run:

```bash
./examples/artemis/intel-cpu-int8/preflight.sh
```

Preflight builds the pristine baseline, runs numerical parity, benchmarks it
twice, verifies the metric schema, and rejects a noisy host when corresponding
metrics drift by more than 15%. Use an idle, exclusive runner. Override the
limit only when there is a measured reason:

```bash
ARTEMIS_PREFLIGHT_MAX_DRIFT=0.20 \
  ./examples/artemis/intel-cpu-int8/preflight.sh
```

The expected baseline pattern is that M=4 has lower weight-streaming
efficiency than the neighboring operating points. Exact GB/s depends on the
Xeon SKU and memory configuration; preserve the two preflight JSON files as
the machine's baseline rather than copying numbers from another host.

## Import with Artemis CLI

The commands below match Artemis CLI 1.0.11. Install and configure the CLI,
then register a Git credential and note its ID:

```bash
artemis config set ARTEMIS_API_KEY <api-key>
artemis key add --name intel-sglang --provider github --token <github-token>
artemis key list
```

Import the prepared onboarding branch:

```bash
artemis --output-format json project import \
  --git-url https://github.com/paulsbrookes/sglang.git \
  --key-id <git-key-id> \
  --branch artemis-onboarding \
  --name sglang-intel-cpu-int8
```

Project import is asynchronous. Poll `artemis --output-format json project
list` until `importedStatus` is `success`, then verify its `gitHash` matches
the published onboarding branch tip.

List runners and configure the project:

```bash
artemis runner list

artemis project runner set \
  --project <project-id> \
  --runner <runner-name>

artemis project commands set \
  --project <project-id> \
  --compile "./examples/artemis/intel-cpu-int8/compile.sh" \
  --test "./examples/artemis/intel-cpu-int8/test.sh" \
  --benchmark "./examples/artemis/intel-cpu-int8/benchmark.sh"
```

If a non-default cache is used, include the same prefix in all three stored
commands, for example:

```text
ARTEMIS_CACHE_ROOT=/mnt/fast-cache/sglang-int8 ./examples/artemis/intel-cpu-int8/compile.sh
```

Create Discovery with the task text in
[`TASK_PROMPT.md`](TASK_PROMPT.md):

```bash
artemis discovery create \
  --project <project-id> \
  --task "<paste the Task text from TASK_PROMPT.md>" \
  --model <model-code-or-id> \
  --target-files python/sglang/kernels/aot/csrc/cpu/gemm_int8.cpp \
  --versions 10 \
  --runner <runner-name> \
  --mode automatic
```

Immediately inspect the created run:

```bash
artemis --output-format json discovery get <run-id> | jq '.metricsSchema'
```

Require the three worker metrics and weights in
[`metrics-schema.json`](metrics-schema.json). If agent metrics appear or the
weights differ, stop and recreate the run before spending the version budget.

## Commands used by Artemis

The stored commands are deliberately simple:

```text
compile:   ./examples/artemis/intel-cpu-int8/compile.sh
test:      ./examples/artemis/intel-cpu-int8/test.sh
benchmark: ./examples/artemis/intel-cpu-int8/benchmark.sh
```

The benchmark atomically writes a flat numeric map at repository root:

```json
{
  "mb_int8_mlp_down_m1_gbps": 0.0,
  "mb_int8_mlp_down_m4_gbps": 0.0,
  "mb_int8_mlp_down_m32_gbps": 0.0
}
```

The zeros above show the schema only; a real benchmark requires every value to
be finite and positive. M=4 is the primary objective. M=1 and M=32 guard
against shifting the regression elsewhere.

## Optional full SGLang serving confirmation

Promote a microbenchmark winner only after end-to-end validation. The optional
script uses SGLang's native `sglang.benchmark.serving` and `sglang.test.run_eval`
tools. It needs a full SGLang CPU environment, a compatible local checkpoint,
and enough memory to serve it:

```bash
ARTEMIS_SERVING_VENV=/path/to/full-sglang-cpu-venv \
SGLANG_MODEL_PATH=/path/to/Qwen3.8-27B-W8A8-Int8 \
SGLANG_QUANTIZATION=w8a8_int8 \
  ./examples/artemis/intel-cpu-int8/benchmark-serving.sh
```

It runs two concurrency-4 serving measurements, reports their median, and
requires the SGLang GSM8K score to remain at least 0.94. Results are written to
`artemis_serving_results.json`; this script is not the default Artemis
benchmark because model setup and validation are much slower.

For other SGLang projects, keep the same progression:

1. Use a fast, representative microbenchmark to screen versions.
2. Keep numerical or functional tests as hard gates.
3. Confirm promising versions with `sglang.benchmark.serving` or
   `sglang.benchmark.offline_throughput`.
4. Run the relevant `sglang.test.run_eval` quality suite before accepting a
   serving optimization.

## Troubleshooting

- `CPU lacks required flag`: move the runner to an AMX-capable Intel Xeon.
- `workspace is not initialized`: run `setup-workspace.sh` once on that runner.
- Candidate kernel resolves outside `.artemis-kernel`: run `compile.sh` and
  keep its `PYTHONPATH`; do not install candidates into the shared venv.
- High preflight drift: stop other workloads and reserve the runner
  exclusively. Shared-host collapses can make microbenchmark winners false.
- Cold builds remain slow: check `ccache -s` and make sure
  `ARTEMIS_CACHE_ROOT` persists across runner jobs.
- Changed branch baseline: initialize a new cache root. Reusing build state
  across unrelated seeds is intentionally rejected.
