# SPDX-License-Identifier: Apache-2.0
"""Check graph-safe overlap."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


def test_step5_overlap_graph(monkeypatch):
    import torch

    from vllm_sail.models.step5 import mega_moe_dispatch as dispatch

    class Experts:
        _sail_sequence_parallel = True

        def _sail_shared_expert(self):
            return lambda x: x * 2

        def __call__(self, x, weights, ids, padding):
            return x + 3

    monkeypatch.setenv("VLLM_SAIL_STEP5_MEGA_MOE_OVERLAP", "1")
    monkeypatch.setattr(
        dispatch,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={"experts": Experts()}),
    )
    x = torch.randn(64, 192, device="cuda", dtype=torch.bfloat16)
    name = dispatch._encode_layer_name("experts")
    stream = dispatch.aux_stream()
    if stream is None:
        pytest.skip("Auxiliary stream unavailable")

    def run():
        return dispatch._dispatch(x, x, x, None, name, True)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        shared, routed = run()
    x.fill_(2)
    graph.replay()
    torch.testing.assert_close(shared, torch.full_like(x, 4))
    torch.testing.assert_close(routed, torch.full_like(x, 5))
