# SPDX-License-Identifier: Apache-2.0
"""Validate Step5 BF16 paged KV insertion through the installed SAIL patch."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("rotary_dim", [64, 192])
@pytest.mark.parametrize(
    "rows,heads,kv_heads", [(4, 16, 1), (64, 16, 2), (513, 8, 2), (2048, 16, 1)]
)
def test_qknorm_rope_cache(monkeypatch, rotary_dim, rows, heads, kv_heads):
    import torch
    from vllm import _custom_ops
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.has_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    monkeypatch.setenv("VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV", "1")
    vllm_sail.register_out_of_tree()
    from vllm_sail.models.step5.ops.qknorm_rope_cache import register_ops
    from vllm_sail.patch.performance.step5_qknorm_rope_cache import _original

    register_ops()
    torch.manual_seed(42)
    dtype, device = torch.bfloat16, "cuda"
    dim, page_size = 192, 64
    width = (heads + 2 * kv_heads) * dim
    qkv = torch.randn(rows, width + 16, device=device, dtype=dtype)[:, :width]
    weights = torch.randn(2, dim, device=device, dtype=torch.float32) * 0.1
    angles = torch.randn(32, rotary_dim // 2, device=device)
    rope = torch.cat((angles.cos(), angles.sin()), dim=1)
    positions = torch.arange(rows, device=device) % 32
    blocks = (rows + page_size - 1) // page_size
    cache = torch.full(
        (blocks, kv_heads, page_size, 2 * dim), -7, device=device, dtype=dtype
    )
    reference_cache = cache.clone()
    slots = torch.randperm(blocks * page_size, device=device)[: rows - 1]
    slots[::5] = -1

    def run(op, pages):
        key, value = pages.transpose(1, 2).split(dim, dim=-1)
        return op(
            qkv,
            weights[0],
            weights[1],
            rope,
            positions,
            slots,
            key,
            value,
            heads,
            kv_heads,
            dim,
            rotary_dim,
            1e-6,
            1.0,
        )

    expected = run(_original, reference_cache)
    actual = run(_custom_ops.fused_qknorm_rope_cache, cache)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0.008, rtol=0.008)
    torch.testing.assert_close(cache, reference_cache, atol=0.008, rtol=0.008)
    untouched = torch.ones(blocks * page_size, device=device, dtype=torch.bool)
    untouched[slots[slots >= 0]] = False
    assert torch.all(
        cache.transpose(1, 2).reshape(-1, kv_heads, 2 * dim)[untouched] == -7
    )
    assert actual[2].untyped_storage().data_ptr() == qkv.untyped_storage().data_ptr()

    if rows not in (64, 513):
        return

    # Capture padded slots, then update their contents between replays.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = run(_custom_ops.fused_qknorm_rope_cache, cache)
    for step in (1, 2):
        qkv.add_(step * 0.125)
        positions.add_(step).remainder_(32)
        slots.copy_(torch.roll(slots, step))
        graph.replay()
        expected = run(_original, reference_cache)
        for got, want in zip(replayed, expected, strict=True):
            torch.testing.assert_close(got, want, atol=0.008, rtol=0.008)
        torch.testing.assert_close(cache, reference_cache, atol=0.008, rtol=0.008)

    monkeypatch.setenv("VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV", "0")
    fallback = run(_custom_ops.fused_qknorm_rope_cache, cache)
    for got, want in zip(fallback, expected, strict=True):
        torch.testing.assert_close(got, want, atol=0, rtol=0)
