# SPDX-License-Identifier: Apache-2.0
"""Validate Step5 split-K routing against the FP32 gate and native TopK."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("rows", [1, 32, 128, 512, 1024, 2048, 2816, 11264])
@pytest.mark.parametrize("renormalize", [False, True])
def test_step5_fused_router(monkeypatch, rows, renormalize):
    import torch
    from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
        fused_topk_bias,
    )
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.is_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    monkeypatch.setenv("VLLM_SAIL_STEP5_FUSED_ROUTER", "1")
    vllm_sail.register_out_of_tree()
    torch.manual_seed(19)
    x = torch.randn(rows, 4096, device="cuda", dtype=torch.bfloat16)
    bf16_weight = torch.randn(352, 4096, device="cuda", dtype=torch.bfloat16) / 64
    weight = bf16_weight.float()
    bias = torch.randn(352, device="cuda") * 0.01
    logits = torch.nn.functional.linear(x.float(), weight)
    expected_w, expected_i = fused_topk_bias(
        x,
        logits,
        "sigmoid",
        bias,
        8,
        renormalize,
        torch.int64,
        routed_scaling_factor=2.0,
    )
    actual_w, actual_i = torch.ops.vllm.ppu_step5_fused_router(
        x,
        weight,
        bf16_weight,
        bias,
        renormalize,
        2.0,
    )
    if rows <= 2048:
        torch.testing.assert_close(actual_i, expected_i, atol=0, rtol=0)
        torch.testing.assert_close(actual_w, expected_w, atol=2e-6, rtol=2e-5)
    else:
        # FP32 accumulation order can swap nearly tied candidates. Check
        # their ranking scores and the weights of the experts actually chosen.
        scores = logits.sigmoid() + bias
        torch.testing.assert_close(
            scores.gather(1, actual_i),
            scores.gather(1, expected_i),
            atol=5e-6,
            rtol=0,
        )
        ids = actual_i.sort(dim=1).values
        assert not torch.any(ids[:, 1:] == ids[:, :-1])
        selected_weights = logits.gather(1, actual_i).sigmoid()
        if renormalize:
            selected_weights /= selected_weights.sum(dim=1, keepdim=True)
        torch.testing.assert_close(
            actual_w, selected_weights * 2.0, atol=2e-6, rtol=2e-5
        )


@pytest.mark.parametrize("parallel", [False, True])
def test_step5_router_nan_matches_independent_topk(parallel):
    import torch
    from vllm.platforms import current_platform

    from vllm_sail.models.step5.ops.fused_router import _reduce_router_kernel

    if not current_platform.is_ppu() or not current_platform.is_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    logits = torch.linspace(-1, 1, 352, device="cuda").view(1, 352)
    logits[0, 0] = torch.nan
    bias = torch.zeros_like(logits)
    bias[0, 0] = 2
    weights = torch.empty((1, 8), device="cuda")
    ids = torch.empty((1, 8), device="cuda", dtype=torch.int64)
    _reduce_router_kernel[(1,)](
        logits,
        bias,
        weights,
        ids,
        None,
        352,
        352,
        SPLIT_K=1,
        RENORMALIZE=False,
        SCALING=1.0,
        IDX64=True,
        PARALLEL_TOPK=parallel,
        HAS_PADDING=False,
        num_warps=4,
    )
    probs = logits.sigmoid().nan_to_num(nan=0.0)
    expected = (probs + bias).topk(8).indices
    torch.testing.assert_close(ids, expected, atol=0, rtol=0)
    torch.testing.assert_close(weights, probs.gather(1, expected))
