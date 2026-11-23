# SPDX-License-Identifier: Apache-2.0
"""Check fused SWIGLUSTEP against BF16 activation followed by MXFP4 downcast."""

import pytest

pytestmark = [pytest.mark.ppu, pytest.mark.cap89]


@pytest.mark.parametrize("hidden", [128, 320, 1024])
def test_masked_swiglustep_quant(hidden):
    import torch
    from vllm.model_executor.layers.fused_moe.activation import (
        MoEActivation,
        apply_moe_activation,
    )

    from vllm_sail.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        silu_mul_mxfp4_quant_masked,
    )
    from vllm_sail.model_executor.layers.quantization.utils.mxfp4_utils import (
        downcast_to_mxfp4,
    )

    counts = [0, 1, 9, 33, 47]
    tokens = 48
    generator = torch.Generator(device="cuda").manual_seed(23)
    x = torch.randn(
        (len(counts), tokens, 2 * hidden),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    ) * 8
    valid = torch.arange(tokens, device="cuda")[None, :] < torch.tensor(
        counts, device="cuda"
    )[:, None]
    x.masked_fill_(~valid[:, :, None], float("nan"))
    expected_act = torch.zeros(
        (len(counts), tokens, hidden), dtype=x.dtype, device=x.device
    )
    valid_counts = torch.tensor(counts, dtype=torch.int32, device="cuda")
    apply_moe_activation(
        MoEActivation.SWIGLUSTEP, expected_act, x, valid_token_counts=valid_counts
    )
    expected, expected_scales = downcast_to_mxfp4(expected_act, axis=-1)
    actual, scales = silu_mul_mxfp4_quant_masked(
        x, valid_counts, is_swiglustep=True
    )
    torch.testing.assert_close(actual[valid], expected[valid], rtol=0, atol=0)
    torch.testing.assert_close(
        scales.transpose(1, 2).view(torch.int16)[valid],
        expected_scales.view(torch.int16)[valid],
        rtol=0,
        atol=0,
    )
