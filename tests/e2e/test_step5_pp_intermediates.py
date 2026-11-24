# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for the Step5 pipeline receive buffer."""

import pytest

pytestmark = pytest.mark.ppu


@pytest.mark.parametrize("fp32_residual", [False, True])
def test_step5_pipeline_preserves_residual_dtype(fp32_residual):
    from types import SimpleNamespace

    import torch
    from vllm.platforms import current_platform

    import vllm_sail

    if not current_platform.is_ppu():
        pytest.skip("Requires PPU")
    vllm_sail.register_out_of_tree()
    from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
    from vllm.models.step5.nvidia.model import Step5ForCausalLM

    allocator = make_empty_intermediate_tensors_factory(["hidden_states"], 8)
    model = SimpleNamespace(
        fp32_residual_connection=fp32_residual,
        model=SimpleNamespace(make_empty_intermediate_tensors=allocator),
    )
    buffer = Step5ForCausalLM.make_empty_intermediate_tensors(
        model, 3, torch.bfloat16, torch.device("cuda")
    )["hidden_states"]
    assert buffer.dtype == (torch.float32 if fp32_residual else torch.bfloat16)
    sent = torch.full((3, 8), 1.001, device="cuda", dtype=torch.float32)
    buffer.copy_(sent)
    torch.testing.assert_close(buffer, sent.to(buffer.dtype), atol=0, rtol=0)
