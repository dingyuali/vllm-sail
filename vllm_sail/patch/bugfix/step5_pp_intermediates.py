# SPDX-License-Identifier: Apache-2.0
"""Preserve FP32 residuals across Step5 pipeline stages."""

from vllm_sail.patch.utils import patch

_TARGET = "vllm.models.step5.nvidia.model"
_installed = False


def install() -> None:
    global _installed
    if _installed:
        return

    import torch
    from vllm.models.step5.nvidia.model import Step5ForCausalLM
    from vllm.platforms import current_platform

    original = Step5ForCausalLM.make_empty_intermediate_tensors

    @patch(
        _TARGET,
        "Step5ForCausalLM.make_empty_intermediate_tensors",
        reason="Prevent the PP receive buffer narrowing Step5 FP32 residuals.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="Step5 allocates PP residual buffers in the residual dtype.",
    )
    def make_empty_intermediate_tensors(self, batch_size, dtype, device):
        if current_platform.is_ppu() and self.fp32_residual_connection:
            dtype = torch.float32
        return original(self, batch_size, dtype, device)

    _installed = True
