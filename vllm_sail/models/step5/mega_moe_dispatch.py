# SPDX-License-Identifier: Apache-2.0
"""Opaque dispatch for optional Step5 MegaMoE stream scheduling."""

import torch
from vllm import envs as vllm_envs
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    aux_stream,
    direct_register_custom_op,
)

from vllm_sail import envs


def overlap_allowed(tokens, sequence_parallel):
    return (
        envs.VLLM_SAIL_STEP5_MEGA_MOE_OVERLAP
        and sequence_parallel
        and not vllm_envs.VLLM_DISABLE_SHARED_EXPERTS_STREAM
        and 0 < tokens <= vllm_envs.VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD
    )


def _dispatch(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    padding: torch.Tensor | None,
    layer_name: LayerNameType,
    run_shared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    experts = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    shared_expert = experts._sail_shared_expert()
    stream = None
    if overlap_allowed(hidden_states.shape[0], experts._sail_sequence_parallel):
        stream = aux_stream()
    if stream is None:
        return (
            shared_expert(hidden_states),
            experts(hidden_states, weights, ids, padding),
        )
    main = torch.cuda.current_stream()
    hidden_states.record_stream(stream)
    stream.wait_stream(main)
    with torch.cuda.stream(stream):
        shared = shared_expert(hidden_states)
    routed = experts(hidden_states, weights, ids, padding)
    main.wait_stream(stream)
    shared.record_stream(main)
    return shared, routed


def _dispatch_fake(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    ids: torch.Tensor,
    padding: torch.Tensor | None,
    layer_name: LayerNameType,
    run_shared: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    shared = (
        torch.empty_like(hidden_states) if run_shared else hidden_states.new_empty((0,))
    )
    return shared, torch.empty_like(hidden_states)


def register_ops():
    direct_register_custom_op(
        "sail_step5_mega_moe_dispatch", _dispatch, fake_impl=_dispatch_fake
    )


def can_overlap(layer, hidden_states):
    return not layer.share_expert.shard_sequence_parallel and overlap_allowed(
        hidden_states.numel() // hidden_states.shape[-1], layer.use_sequence_parallel
    )


def dispatch(layer, hidden_states, weights, ids, padding):
    experts = layer.experts.routed_experts
    if can_overlap(layer, hidden_states):
        return torch.ops.vllm.sail_step5_mega_moe_dispatch(
            hidden_states,
            weights,
            ids,
            padding,
            _encode_layer_name(experts.prefix),
            True,
        )
    shared = layer.share_expert(hidden_states)
    return shared, experts(hidden_states, weights, ids, padding)
