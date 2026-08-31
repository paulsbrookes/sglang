# Discovery task prompt

Use the text below as the `--task` value for `artemis discovery create`.

## Task text

Improve the Intel CPU INT8 W8A8 GEMM used by SGLang at decode batch size
`M=4`. The cache-honest benchmark models the `mlp_down` projection from
Qwen3.8-27B (`N=5120`, `K=17408`) and cycles more weight data than the
last-level cache, so the measurement represents weight streaming rather than
an artificially hot microbenchmark.

Work primarily in:

- `python/sglang/kernels/aot/csrc/cpu/gemm_int8.cpp`

Requirements:

1. Maximize `mb_int8_mlp_down_m4_gbps`.
2. Keep `mb_int8_mlp_down_m1_gbps` and
   `mb_int8_mlp_down_m32_gbps` within 5% of their baselines.
3. Keep `examples/artemis/intel-cpu-int8/test.sh` passing.
4. Do not weaken, bypass, or modify the benchmark, test, build, or metric
   collection files.
5. Do not reduce benchmark repetitions or weight-replica coverage.
6. Treat microbenchmark gains as a screening result. Recommend full SGLang
   serving validation before claiming an end-to-end improvement.

Use worker metrics only. The metric importance contract is:

- `mb_int8_mlp_down_m4_gbps`: 0.70, higher is better.
- `mb_int8_mlp_down_m1_gbps`: 0.15, higher is better.
- `mb_int8_mlp_down_m32_gbps`: 0.15, higher is better.

Do not add agent- or LLM-scored metrics. Read numeric metric values and command
outcomes rather than inferring success from lifecycle status.

## Operator check after creation

Before allowing the run to continue, inspect its metric schema:

```bash
artemis --output-format json discovery get <run-id> | jq '.metricsSchema'
```

Require exactly the three worker metrics above, with no agent metrics. Stop
and recreate the run if the schema differs.
