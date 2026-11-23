# SPDX-License-Identifier: Apache-2.0
"""Compare Step5's slot-first indexer preparation to the compatibility path."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("tokens", [4, 2048, 11264])
@pytest.mark.parametrize("heads", [4, 8])
@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("rotary", [32, 64])
def test_slot_first_indexer_norm_rope(monkeypatch, tokens, heads, half, rotary):
    import torch
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu():
        pytest.skip("Requires PPU")
    # Exercise slot-first even above the per-token dispatch threshold.
    monkeypatch.setenv("VLLM_SAIL_STEP5_INDEXER_PER_TOKEN", "0")
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", str(int(half)))
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia.ops import indexer_norm_rope as indexer

    torch.manual_seed(42)
    dtype, device = torch.bfloat16, "cuda"
    q = torch.randn(tokens, heads * 256, dtype=dtype, device=device)
    k = torch.randn(tokens, 256, dtype=dtype, device=device)
    weights = torch.randn(tokens, heads, device=device)
    qw, kw, kb = (torch.randn(256, dtype=dtype, device=device) * 0.1 for _ in range(3))
    phase = torch.randn(37, rotary // 2, device=device)
    rope = torch.cat((phase.cos(), phase.sin()), dim=1).to(dtype)
    positions = torch.arange(tokens, device=device) % 37
    args = (q, k, weights, qw, kw, kb, rope, positions, 1e-6, True)
    launch = indexer._indexer_norm_rope
    assert launch.optimize_indexer
    monkeypatch.setattr(launch, "optimize_indexer", False)
    expected = indexer.indexer_norm_rope(*args)
    monkeypatch.setattr(launch, "optimize_indexer", True)
    actual = indexer.indexer_norm_rope(*args)
    assert actual[0].dtype == (torch.float16 if half else torch.float8_e4m3fn)
    for got, want in zip(actual, expected, strict=True):
        # Compare FP8 bytes even on devices without native FP8 arithmetic.
        torch.testing.assert_close(
            got.view(torch.uint8), want.view(torch.uint8), atol=0, rtol=0
        )


@pytest.mark.parametrize("half", [False, True])
def test_joint_indexer_csa_with_int64_slots(monkeypatch, half):
    """The joint Triton helper must keep program IDs and cache slots separate."""
    from types import SimpleNamespace

    import torch
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu():
        pytest.skip("Requires PPU")
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", str(int(half)))
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia.ops.qkv_indexer_norm_rope import qkv_indexer_norm_rope

    dtype, device, tokens = torch.bfloat16, "cuda", 4
    q = torch.ones((tokens, 4 * 256), dtype=dtype, device=device)
    k = torch.ones((tokens, 256), dtype=dtype, device=device)
    w = torch.ones((tokens, 4), device=device)
    unit = torch.ones(256, dtype=dtype, device=device)
    rope = torch.ones((tokens, 64), dtype=dtype, device=device)
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    slots = positions.clone()
    mean = torch.zeros(
        (1, 8, 256), dtype=torch.float16 if half else torch.uint8, device=device
    )
    ring = torch.zeros((8, 512), dtype=dtype, device=device)
    bases = torch.zeros_like(positions)
    state = SimpleNamespace(
        rows=ring, capacity=8, metadata=lambda: SimpleNamespace(base=bases)
    )
    main_qkv = torch.ones((tokens, 18 * 192), dtype=dtype, device=device)
    main_w = torch.ones(192, dtype=dtype, device=device)
    main_kc = torch.zeros((1, 64, 1, 192), dtype=dtype, device=device)
    main_vc = torch.zeros_like(main_kc)

    def prepare():
        return qkv_indexer_norm_rope(
            q,
            k,
            w,
            unit,
            unit,
            unit,
            rope,
            positions,
            1e-6,
            True,
            main_inputs=(
                main_qkv,
                main_w,
                main_w,
                rope,
                slots,
                main_kc,
                main_vc,
                16,
                1,
                64,
                1e-6,
                1.0,
                None,
                None,
            ),
            decode_cache=SimpleNamespace(mean=mean),
            decode_ring=state,
            decode_z=k,
            decode_slots=slots,
        )

    prepare()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        index_q, index_k, _, _, main_k, main_v = prepare()
    graph.replay()
    assert index_q.dtype == (torch.float16 if half else torch.float8_e4m3fn)
    torch.testing.assert_close(ring[:tokens, :256], index_k[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(ring[:tokens, 256:], k, atol=0, rtol=0)
    torch.testing.assert_close(main_kc[0, :tokens, 0], main_k, atol=0, rtol=0)
    torch.testing.assert_close(main_vc[0, :tokens, 0], main_v, atol=0, rtol=0)


@pytest.mark.parametrize("half", [False, True])
@pytest.mark.parametrize("tokens", [1023, 1024])
def test_v023_per_token_indexer(monkeypatch, half, tokens):
    import torch
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu() or not current_platform.is_device_capability(89):
        pytest.skip("Requires an M890 PPU")
    monkeypatch.setenv("VLLM_STEP5_INDEXER_FP8_AS_FP16", str(int(half)))
    vllm_sail.register_out_of_tree()
    from vllm.models.step5.nvidia.ops.indexer_norm_rope import indexer_norm_rope

    torch.manual_seed(29)
    dtype = torch.bfloat16
    projection = torch.randn(tokens, 1540, dtype=dtype, device="cuda")
    q, k = projection[:, :1024], projection[:, 1024:1280]
    weights = torch.randn(tokens, 4, device="cuda")
    qw, kw, kb = (torch.randn(256, dtype=dtype, device="cuda") for _ in range(3))
    phase = torch.randn(37, 16, device="cuda")
    rope = torch.cat((phase.cos(), phase.sin()), dim=1).to(dtype)
    positions = torch.arange(tokens, device="cuda") % 37
    args = (q, k, weights, qw, kw, kb, rope, positions, 1e-6, True)

    monkeypatch.setenv("VLLM_SAIL_STEP5_INDEXER_PER_TOKEN", "0")
    expected = indexer_norm_rope(*args, native_rope=True)
    monkeypatch.setenv("VLLM_SAIL_STEP5_INDEXER_PER_TOKEN", "1")
    actual = indexer_norm_rope(*args, native_rope=True)

    assert actual[0].dtype == expected[0].dtype
    assert actual[1].dtype == dtype
    torch.testing.assert_close(actual[2], expected[2], atol=0, rtol=0)
    assert torch.count_nonzero(actual[0].float()[:, :, 4:]) == 0
    for got, want in zip(actual[:2], expected[:2], strict=True):
        if tokens < 1024:
            torch.testing.assert_close(got.view(torch.uint8), want.view(torch.uint8))
        else:
            want = want.float()
            relative_error = (got.float() - want).norm() / want.norm()
            assert relative_error < 0.02
