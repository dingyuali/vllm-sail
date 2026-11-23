# SPDX-License-Identifier: Apache-2.0
"""Optimize Step5 independent and split-K routing on M890."""

from vllm_sail import envs

_installed = False


def _install_independent_topk() -> None:
    import functools

    import torch
    from vllm import envs as vllm_envs
    from vllm.model_executor.layers.fused_moe.router import fused_topk_bias_router

    from vllm_sail.models.step5.ops.fused_router import _reduce_router_kernel
    from vllm_sail.patch.utils import patch

    original = fused_topk_bias_router.vllm_topk_sigmoid

    @patch(
        fused_topk_bias_router.__name__,
        "vllm_topk_sigmoid",
        reason="Reduce latency of independent Step5 sigmoid Top8 at small M on M890.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="Native biased sigmoid TopK matches small-M Triton latency.",
    )
    @functools.wraps(original)
    def topk_sigmoid(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
        routed_scaling_factor=1.0,
    ):
        rows = gating_output.shape[0]
        bias = e_score_correction_bias
        if (
            vllm_envs.VLLM_BATCH_INVARIANT
            or not 0 < rows <= 512
            or gating_output.shape != (rows, 352)
            or topk_weights.shape != (rows, 8)
            or gating_output.dtype != torch.float32
            or not gating_output.is_cuda
            or bias is None
            or bias.shape != (352,)
            or bias.dtype != torch.float32
            or not all(
                t.is_contiguous()
                for t in (
                    gating_output,
                    bias,
                    topk_weights,
                    topk_indices,
                    token_expert_indices,
                )
            )
            or topk_indices.dtype not in (torch.int32, torch.int64)
        ):
            return original(
                topk_weights,
                topk_indices,
                token_expert_indices,
                gating_output,
                renormalize,
                bias,
                routed_scaling_factor,
            )
        sms = torch.cuda.get_device_properties(
            gating_output.device
        ).multi_processor_count
        warps = 8 if rows <= sms else 4 if rows <= 4 * sms else 1
        padding = fused_topk_bias_router._get_padding_mask(rows)
        if padding is not None:
            padding = padding.contiguous()
        _reduce_router_kernel[(rows,)](
            gating_output,
            bias,
            topk_weights,
            topk_indices,
            padding,
            352,
            0,
            SPLIT_K=1,
            RENORMALIZE=renormalize,
            SCALING=routed_scaling_factor,
            IDX64=topk_indices.dtype == torch.int64,
            PARALLEL_TOPK=False,
            HAS_PADDING=padding is not None,
            source_rows_ptr=token_expert_indices,
            num_warps=warps,
            num_stages=2,
        )
        return topk_weights, topk_indices


def install() -> None:
    global _installed
    if _installed:
        return

    from vllm.platforms import current_platform

    if not current_platform.is_ppu() or not current_platform.is_device_capability(89):
        return

    _install_independent_topk()
    if not envs.VLLM_SAIL_STEP5_FUSED_ROUTER:
        _installed = True
        return

    import functools

    import torch
    from vllm import envs as vllm_envs
    from vllm.forward_context import get_forward_context, is_forward_context_available
    from vllm.models.step5.nvidia import model

    from vllm_sail.models.step5.mega_moe import PPUStep5MegaMoEExperts
    from vllm_sail.models.step5.ops.fused_router import register_ops
    from vllm_sail.patch.utils import patch

    meta = dict(
        reason="Fuse Step5 MegaMoE split-K router reduction and sigmoid TopK on M890.",
        affected_versions=">=0.33.0,<0.34.0",
        remove_when="Step5 MegaMoE exposes a fused gate and TopK router entry point.",
    )
    register_ops()
    original_load = model.Step5ForCausalLM.load_weights
    original_forward = model.Step5MoE.forward

    @patch(model.__name__, "Step5ForCausalLM.load_weights", **meta)
    @functools.wraps(original_load)
    def load_weights(self, weights):
        loaded = original_load(self, weights)
        for layer in self.modules():
            if not isinstance(layer, model.Step5MoE):
                continue
            experts = getattr(layer.experts, "routed_experts", None)
            if not isinstance(experts, PPUStep5MegaMoEExperts):
                continue
            weight = layer.gate.weight.detach()
            bf16_weight = None
            if weight.shape == (352, 4096) and weight.dtype == torch.float32:
                candidate = weight.to(torch.bfloat16).contiguous()
                if torch.equal(weight, candidate.float()):
                    bf16_weight = candidate
            layer.register_buffer(
                "_sail_router_bf16_weight", bf16_weight, persistent=False
            )
        return loaded

    @patch(model.__name__, "Step5MoE.forward", **meta)
    @functools.wraps(original_forward)
    def forward(self, hidden_states, padding=None):
        bf16_weight = getattr(self, "_sail_router_bf16_weight", None)
        if (
            bf16_weight is None
            or hidden_states.dtype != torch.bfloat16
            or bf16_weight.device != hidden_states.device
            or self.enable_eplb
            or vllm_envs.VLLM_BATCH_INVARIANT
        ):
            return original_forward(self, hidden_states, padding)

        experts = self.experts
        router = experts.router
        if (
            router.scoring_func != "sigmoid"
            or router.top_k != 8
            or router.capture_fn is not None
        ):
            return original_forward(self, hidden_states, padding)

        # PPU MODIFICATION: begin — bypass materialized logits for the MegaMoE path.
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        shared = self.share_expert(hidden_states)
        weights, ids = torch.ops.vllm.ppu_step5_fused_router(
            hidden_states,
            self.gate.weight,
            bf16_weight,
            self.router_bias,
            router.renormalize,
            router.routed_scaling_factor,
        )
        if (
            padding is None
            and not experts.sequence_parallel
            and vllm_envs.VLLM_MOE_SKIP_PADDING
            and is_forward_context_available()
        ):
            mask = get_forward_context().is_padding
            padding = mask[: hidden_states.shape[0]] if mask is not None else None
        return shared, experts.routed_experts(hidden_states, weights, ids, padding)
        # PPU MODIFICATION: end

    _installed = True


install()
