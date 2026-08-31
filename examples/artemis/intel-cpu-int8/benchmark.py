#!/usr/bin/env python3
"""Cache-honest INT8 GEMM benchmark for the Artemis onboarding example."""

from __future__ import annotations

import json
import math
import os
import pathlib
import statistics
import time

import torch

LABEL = "mlp_down"
N = 5120
K = 17408
M_VALUES = (1, 4, 32)
CYCLE_BYTES = 1_500_000_000
CALLS_PER_RUN = 50
BASE_REPEATS = 5
COLLAPSE_SPREAD = 0.20


def candidate_kernel_path() -> pathlib.Path:
    import sgl_kernel

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    kernel_root = (repo_root / ".artemis-kernel").resolve()
    loaded = pathlib.Path(sgl_kernel.__file__).resolve()
    if kernel_root not in loaded.parents:
        raise SystemExit(
            f"refusing to benchmark {loaded}; expected the candidate kernel under "
            f"{kernel_root}"
        )
    return loaded


def make_replicas() -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    packed_bytes = N * (K + 4)
    replica_count = max(2, math.ceil(CYCLE_BYTES / packed_bytes))
    replicas = []
    for _ in range(replica_count):
        weight = torch.randint(-127, 128, (N, K), dtype=torch.int8)
        packed = torch.ops.sgl_kernel.convert_weight_packed(weight)
        scales = torch.rand(N, 1, dtype=torch.float32)
        replicas.append((packed, scales))
    return replicas, packed_bytes


def time_once(
    activations: torch.Tensor,
    replicas: list[tuple[torch.Tensor, torch.Tensor]],
) -> float:
    for packed, scales in replicas[:3]:
        torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            activations, packed, scales, None, torch.bfloat16, True
        )

    started = time.perf_counter()
    for index in range(CALLS_PER_RUN):
        packed, scales = replicas[index % len(replicas)]
        torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            activations, packed, scales, None, torch.bfloat16, True
        )
    return (time.perf_counter() - started) / CALLS_PER_RUN * 1e6


def benchmark_point(
    m: int,
    replicas: list[tuple[torch.Tensor, torch.Tensor]],
    packed_bytes: int,
) -> tuple[float, float, list[float]]:
    activations = torch.rand(m, K, dtype=torch.bfloat16)
    runs = [time_once(activations, replicas) for _ in range(BASE_REPEATS)]
    median_us = statistics.median(runs)
    if (max(runs) - min(runs)) / median_us > COLLAPSE_SPREAD:
        runs.extend(time_once(activations, replicas) for _ in range(2))
        median_us = statistics.median(runs)
    gbps = packed_bytes / (median_us / 1e6) / 1e9
    return median_us, gbps, runs


def write_metrics(metrics: dict[str, float]) -> pathlib.Path:
    if not metrics or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in metrics.values()
    ):
        raise SystemExit("metrics must be a non-empty map of finite numbers")

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    destination = repo_root / "artemis_results.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    with temporary.open("r+") as output:
        os.fsync(output.fileno())
    os.replace(temporary, destination)
    return destination


def main() -> None:
    kernel = candidate_kernel_path()
    print(f"sgl_kernel={kernel}")
    print(f"threads={torch.get_num_threads()}")

    replicas, packed_bytes = make_replicas()
    print(
        f"{LABEL}: N={N} K={K} replicas={len(replicas)} "
        f"({len(replicas) * packed_bytes / 1e9:.2f} GB cycled)"
    )

    metrics: dict[str, float] = {}
    for m in M_VALUES:
        median_us, gbps, runs = benchmark_point(m, replicas, packed_bytes)
        spread = (max(runs) - min(runs)) / statistics.median(runs)
        name = f"mb_int8_{LABEL}_m{m}_gbps"
        metrics[name] = round(gbps, 2)
        print(
            f"M={m}: {median_us:8.1f} us  {gbps:6.1f} GB/s "
            f"(runs={len(runs)}, spread={spread:.1%})"
        )

    destination = write_metrics(metrics)
    print(f"wrote {destination}")
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
