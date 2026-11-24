# SPDX-License-Identifier: Apache-2.0
"""Check graph-safe overlap and optional UltraEP probing."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.ppu


def test_ultra_ep_missing_dependency(monkeypatch):
    from vllm_sail.models.step5 import mega_moe, ultra_ep

    def missing(name):
        raise ModuleNotFoundError(name)

    probe = ultra_ep.available.__wrapped__
    monkeypatch.setattr(ultra_ep, "import_module", missing)
    assert not probe()
    monkeypatch.setattr(ultra_ep, "import_module", lambda _: SimpleNamespace())
    assert not probe()
    monkeypatch.setattr(ultra_ep, "available", probe)
    monkeypatch.setattr(
        ultra_ep, "all_ranks_supported", lambda group, supported: supported
    )
    monkeypatch.setattr(mega_moe, "get_ep_group", lambda: object())
    experts = SimpleNamespace(
        mma_format=SimpleNamespace(kernel="compute"), ultra_ep=None
    )
    api = SimpleNamespace(compute=object())
    mega_moe.PPUStep5MegaMoEExperts._prepare_ultra_ep(experts, api, None)
    assert experts.ultra_ep is None


def test_step5_overlap_graph(monkeypatch):
    import torch

    from vllm_sail.models.step5 import mega_moe_dispatch as dispatch

    class Experts:
        ultra_ep = None
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


@pytest.mark.parametrize(
    "graph_mode, capture_sizes, rejected",
    [
        ("NONE", [1024], False),
        ("PIECEWISE", [508], False),
        ("FULL_AND_PIECEWISE", [508, 512], True),
        ("FULL", [4], True),
    ],
)
def test_ultra_ep_checks_pinned_graph_config(
    monkeypatch, graph_mode, capture_sizes, rejected
):
    from functools import partial

    from vllm.config import CompilationConfig, CUDAGraphMode

    from vllm_sail.models.step5 import mega_moe, ultra_ep

    cls = mega_moe.PPUStep5MegaMoEExperts
    experts = SimpleNamespace(
        mma_format=SimpleNamespace(kernel="compute"),
        ultra_ep=None,
        ultra_ep_min_tokens=128,
        _ultra_ep_sp_size=4,
        _ultra_ep_compilation=CompilationConfig(
            cudagraph_mode=CUDAGraphMode[graph_mode],
            cudagraph_capture_sizes=capture_sizes,
            max_cudagraph_capture_size=max(capture_sizes),
        ),
    )
    experts._ultra_ep_compilation_rejection = partial(
        cls._ultra_ep_compilation_rejection, experts
    )
    assert bool(experts._ultra_ep_compilation_rejection()) is rejected
    if rejected:

        def compute(*, l1_weights_replicas=None, l2_weights_replicas=None):
            raise AssertionError("MegaMoE compute should not run during preparation")

        monkeypatch.setattr(mega_moe, "get_ep_group", lambda: object())
        monkeypatch.setattr(ultra_ep, "available", lambda: True)
        monkeypatch.setattr(
            ultra_ep, "all_ranks_supported", lambda _, supported: supported
        )
        cls._prepare_ultra_ep(experts, SimpleNamespace(compute=compute), None)
        assert experts.ultra_ep is None
