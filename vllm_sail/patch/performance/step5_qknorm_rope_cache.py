# SPDX-License-Identifier: Apache-2.0
"""Route supported Step5 BF16 cache writes to the PPU Triton fusion."""

import functools

import torch
from vllm import _custom_ops
from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.patch.utils import patch


def _supported(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    slot_mapping,
    key_cache,
    value_cache,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    kv_cache_dtype,
):
    if (
        kv_cache_dtype not in ("auto", "bfloat16")
        or head_dim != 192
        or rotary_dim not in (64, 192)
        or num_q_heads <= 0
        or num_kv_heads <= 0
        or qkv.ndim != 2
        or not qkv.shape[0]
        or qkv.shape[1] != (num_q_heads + 2 * num_kv_heads) * head_dim
        or qkv.stride(1) != 1
        or qkv.dtype != torch.bfloat16
        or not qkv.is_cuda
        or q_weight.dtype != k_weight.dtype
        or q_weight.dtype not in (torch.bfloat16, torch.float32)
        or cos_sin_cache.dtype not in (torch.bfloat16, torch.float32)
        or cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[1] != rotary_dim
        or cos_sin_cache.stride(1) != 1
        or positions.shape != (qkv.shape[0],)
        or positions.dtype != torch.int64
        or slot_mapping.ndim != 1
        or slot_mapping.dtype != torch.int64
        or slot_mapping.numel() > qkv.shape[0]
    ):
        return False
    for t in (q_weight, k_weight, positions, slot_mapping):
        if not t.is_contiguous() or t.device != qkv.device:
            return False
    if (
        q_weight.shape != (head_dim,)
        or k_weight.shape != (head_dim,)
        or cos_sin_cache.device != qkv.device
        or key_cache.shape != value_cache.shape
        or key_cache.ndim != 4
        or key_cache.shape[0] == 0
        or key_cache.shape[1] == 0
        or key_cache.shape[2:] != (num_kv_heads, head_dim)
    ):
        return False
    return all(
        t.dtype == qkv.dtype
        and t.device == qkv.device
        and t.stride(2) >= head_dim
        and t.stride(3) == 1
        for t in (key_cache, value_cache)
    )


_original = _custom_ops.fused_qknorm_rope_cache


@patch(
    "vllm._custom_ops",
    "fused_qknorm_rope_cache",
    reason="Reduce Step5 BF16 QK norm, RoPE and KV insertion latency on PPU.",
    affected_versions=">=0.33.0,<0.34.0",
    remove_when="The upstream op matches the PPU Triton path on Step5 shapes.",
)
@functools.wraps(_original)
def fused_qknorm_rope_cache(
    qkv,
    q_weight,
    k_weight,
    cos_sin_cache,
    positions,
    slot_mapping,
    key_cache,
    value_cache,
    num_q_heads,
    num_kv_heads,
    head_dim,
    rotary_dim,
    eps,
    norm_weight_bias=0.0,
    kv_cache_dtype="auto",
    k_scale=None,
    v_scale=None,
):
    if (
        current_platform.is_ppu()
        and envs.VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV
        and current_platform.has_device_capability(89)
        and _supported(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            slot_mapping,
            key_cache,
            value_cache,
            num_q_heads,
            num_kv_heads,
            head_dim,
            rotary_dim,
            kv_cache_dtype,
        )
    ):
        q, k = torch.ops.vllm.ppu_step5_qknorm_rope_cache(
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            slot_mapping,
            key_cache,
            value_cache,
            num_q_heads,
            num_kv_heads,
            rotary_dim,
            eps,
            norm_weight_bias,
        )
        return q, k, qkv[:, (num_q_heads + num_kv_heads) * head_dim :]
    return _original(
        qkv,
        q_weight,
        k_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        key_cache,
        value_cache,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        norm_weight_bias,
        kv_cache_dtype,
        k_scale,
        v_scale,
    )


# Register outside the forward path so Dynamo sees the cache writes.
if envs.VLLM_SAIL_STEP5_QKNORM_ROPE_CACHEKV and current_platform.is_ppu():
    from vllm_sail.models.step5.ops.qknorm_rope_cache import register_ops

    register_ops()
