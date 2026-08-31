#!/usr/bin/env python3
"""Focused numerical gate for the INT8 GEMM optimized by this example."""

from __future__ import annotations

import pathlib

import torch


def quantize_per_token(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    values_f32 = values.float()
    absmax = values_f32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10)
    scales = absmax / 127
    quantized = torch.round(values_f32 * (127 / absmax)).to(torch.int8)
    return quantized, scales


def main() -> None:
    import sgl_kernel

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    kernel_root = (repo_root / ".artemis-kernel").resolve()
    loaded = pathlib.Path(sgl_kernel.__file__).resolve()
    if kernel_root not in loaded.parents:
        raise SystemExit(
            f"refusing to test {loaded}; expected the candidate kernel under {kernel_root}"
        )

    torch.manual_seed(1234)
    n, k = 384, 544
    weight = torch.randint(-127, 128, (n, k), dtype=torch.int8)
    weight_scales = torch.rand(n, dtype=torch.float32) * 1e-2
    packed_weight = torch.ops.sgl_kernel.convert_weight_packed(weight)

    for m in (1, 4, 32):
        activations = torch.randn(m, k, dtype=torch.bfloat16) / 10
        quantized, activation_scales = quantize_per_token(activations)
        reference = (
            torch.matmul(quantized.float(), weight.float().T)
            * activation_scales
            * weight_scales.view(1, -1)
        ).to(torch.bfloat16)

        actual = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            activations,
            packed_weight,
            weight_scales,
            None,
            torch.bfloat16,
            True,
        )
        torch.testing.assert_close(actual, reference, atol=1e-2, rtol=1e-2)
        print(f"M={m}: parity ok")

    print(f"Candidate kernel parity passed: {loaded}")


if __name__ == "__main__":
    main()
