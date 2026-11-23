# SPDX-License-Identifier: Apache-2.0
"""Check independent Step5 Top8 across the Triton dispatch boundary."""

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


@pytest.mark.parametrize(
    "rows", [1, 32, 39, 40, 64, 128, 156, 157, 256, 512, 513, 2816]
)
@pytest.mark.parametrize("id_dtype", ["int32", "int64"])
def test_independent_sigmoid_topk(monkeypatch, rows, id_dtype):
    import torch
    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router

    logits = torch.zeros((rows, 352), device="cuda")
    logits[1:] = torch.arange(352, device="cuda") / 96 - 2
    logits[::2, 9] = float("nan")
    bias = torch.zeros(352, device="cuda")
    bias[351] = 0.01
    padding = torch.zeros(rows, device="cuda", dtype=torch.bool)
    if rows > 1:
        padding[-1] = True
    monkeypatch.setattr(fused_topk_bias_router, "_get_padding_mask", lambda _: padding)

    def allocate():
        return (
            torch.empty((rows, 8), device="cuda"),
            torch.empty((rows, 8), device="cuda", dtype=getattr(torch, id_dtype)),
            torch.empty((rows, 8), device="cuda", dtype=torch.int32),
        )

    expected = allocate()
    actual = allocate()
    ops.topk_sigmoid(*expected, logits, True, bias, 2.0, is_padding=padding)
    fused_topk_bias_router.vllm_topk_sigmoid(*actual, logits, True, bias, 2.0)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want)
