# SPDX-License-Identifier: Apache-2.0
"""Verify Step5 FP8 prefill with remapped paged K/V and causal tails."""

import pytest

pytestmark = pytest.mark.ppu


def test_fp8_prefill_inputs_strided_and_invalid_pages():
    import torch

    from vllm_sail.models.step5.ops.fp8_prefill_inputs import (
        prepare_fp8_prefill_inputs,
    )

    cache = torch.randn(5, 1, 64, 400, device="cuda", dtype=torch.bfloat16)
    cache = cache[..., :384]
    pages = torch.tensor([4, 0, -1, 0, 5, 0, 2, 0], device="cuda")[::2]
    regions = torch.arange(-2, 2049, device="cuda")[::2].reshape(2, 513)
    regions[-1, -1] = 2**62
    key, value, slots = prepare_fp8_prefill_inputs(cache, pages, regions, 197)

    kv = cache.index_select(0, pages.clamp(0, 4)).reshape(-1, 1, 384)[:197]
    torch.testing.assert_close(key, kv[..., :192], atol=0, rtol=0)
    torch.testing.assert_close(value, kv[..., 192:], atol=0, rtol=0)
    valid = (regions >= 0) & (regions <= 24) & ((regions < 8) | (regions >= 24))
    expected = torch.where(valid, regions, -1).to(torch.int32).unsqueeze(1)
    torch.testing.assert_close(slots, expected)


@pytest.mark.parametrize("heads", [8, 16])
def test_fp8_prefill_paged_kv(heads):
    import torch
    from vllm.platforms import current_platform

    from vllm_sail.models.step5.ops.dsa_native import fp8_prefill_request

    if not current_platform.is_device_capability(89):
        pytest.skip("Native Step5 FP8 GQA needs M890")
    torch.manual_seed(172)
    device = "cuda"
    cache = torch.randn(4, 1, 64, 384, device=device, dtype=torch.bfloat16) * 0.5
    pages = torch.tensor([3, 1, 0], device=device, dtype=torch.int32)
    positions = torch.arange(65, 73, device=device)
    query = torch.randn(8, heads, 192, device=device, dtype=torch.bfloat16) * 0.5
    regions = torch.tensor([0, 8, 9, -1], device=device).expand(8, -1).contiguous()
    result = torch.empty_like(query)
    scale = 192**-0.5
    fp8_prefill_request(query, cache, regions, pages, positions, 73, scale, result)
    kv = torch.cat((cache[3, 0], cache[1, 0, :9])).float()
    scores = torch.einsum("qhd,kd->qhk", query.float(), kv[:, :192]) * scale
    keys = torch.arange(73, device=device)
    valid = (keys < 8) | ((keys >= 64) & (keys <= positions[:, None]))
    expected = torch.einsum(
        "qhk,kd->qhd",
        scores.masked_fill(~valid[:, None], -torch.inf).softmax(-1),
        kv[:, 192:],
    )
    relative_error = (result.float() - expected).norm() / expected.norm()
    assert relative_error.item() < 0.09
