# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 QK normalization, NeoX RoPE and BF16 paged KV insertion."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _partial_rope(
    y,
    t,
    t_mask,
    tm,
    positions_ptr,
    cos_sin_ptr,
    cache_stride,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROT_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
):
    """Apply partial NeoX RoPE using a register flip for the paired values."""
    GROUPS: tl.constexpr = BLOCK_D // ROT_DIM
    d = tl.arange(0, BLOCK_D)
    y4 = tl.reshape(y, (BLOCK_T, GROUPS, 2, HALF_ROT))
    partner_y = tl.reshape(tl.flip(y4, 2), (BLOCK_T, BLOCK_D))

    position = tl.load(positions_ptr + t, mask=t_mask, other=0).to(tl.int64)
    cache_row = cos_sin_ptr + position[:, None] * cache_stride
    j = tl.arange(0, HALF_ROT)[None, :]
    cos_j = tl.load(cache_row + j, mask=tm, other=0.0).to(tl.float32)
    sin_j = tl.load(cache_row + HALF_ROT + j, mask=tm, other=0.0).to(tl.float32)
    # [t, g, h, j] -> cos_j[t, j] is independent of g and h, which is
    # exactly cos[d] = cache[d % HALF_ROT] and
    # sin[d] = cache[HALF_ROT + d % HALF_ROT].  The shape tuple has to be
    # spelled inline: binding it to a local name first made Triton's parser
    # hand broadcast_to something it read as a tensor.
    cos = tl.reshape(
        tl.broadcast_to(cos_j[:, None, None, :], (BLOCK_T, GROUPS, 2, HALF_ROT)),
        (BLOCK_T, BLOCK_D),
    )
    sin = tl.reshape(
        tl.broadcast_to(sin_j[:, None, None, :], (BLOCK_T, GROUPS, 2, HALF_ROT)),
        (BLOCK_T, BLOCK_D),
    )

    rotated = (d < ROT_DIM)[None, :] & tm
    sign = tl.where(d < HALF_ROT, -1.0, 1.0)[None, :]
    return tl.where(rotated, y * cos + sign * partner_y * sin, y)


@triton.jit
def _neox_partial_rope_gather(
    y,
    qkv_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    col,
    qkv_row_stride,
    rstd,
    t,
    t_mask,
    tm,
    d,
    cache_stride,
    ROT_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
):
    """Reload the paired value when the rotary width does not divide the tile."""
    rot = d < ROT_DIM
    first = d < HALF_ROT
    # NEOX pairs lane d with d + HALF_ROT in the first half and with
    # d - HALF_ROT in the second, and cos/sin for both halves live at
    # index d % HALF_ROT (cos) and HALF_ROT + d % HALF_ROT (sin).
    partner_d = tl.where(first, d + HALF_ROT, d - HALF_ROT)
    cache_j = tl.where(first, d, d - HALF_ROT)
    sign = tl.where(first, -1.0, 1.0)[None, :]
    pm = tm & rot[None, :]

    partner_x = tl.load(
        qkv_ptr + t[:, None] * qkv_row_stride + col + partner_d[None, :],
        mask=pm,
        other=0.0,
    ).to(tl.float32)
    partner_w = tl.load(weight_ptr + partner_d, mask=rot, other=0.0)
    # Same rounding rule as the caller applies to y: fp32 through the norm,
    # one rounding to the activation dtype before the rotation products.
    partner = (
        (partner_x * rstd * (partner_w.to(tl.float32) + WEIGHT_OFFSET)[None, :])
        .to(qkv_ptr.dtype.element_ty)
        .to(tl.float32)
    )

    position = tl.load(positions_ptr + t, mask=t_mask, other=0).to(tl.int64)
    cache_row = cos_sin_ptr + position[:, None] * cache_stride
    cos = tl.load(cache_row + cache_j[None, :], mask=pm, other=0.0).to(tl.float32)
    sin = tl.load(cache_row + HALF_ROT + cache_j[None, :], mask=pm, other=0.0).to(
        tl.float32
    )
    return tl.where(pm, y * cos + sign * partner * sin, y)


@triton.jit
def _gemma_norm_partial_rope(
    x,
    qkv_ptr,
    weight_ptr,
    positions_ptr,
    cos_sin_ptr,
    col,
    qkv_row_stride,
    t,
    t_mask,
    tm,
    d,
    cache_stride,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROT_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    ROT_DIVIDES: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
):
    """Normalize one head and round to the activation dtype before RoPE."""
    weight = tl.load(weight_ptr + d, mask=d < HEAD_DIM, other=0.0)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / HEAD_DIM + EPS)[:, None]
    y = x * rstd * (weight.to(tl.float32) + WEIGHT_OFFSET)[None, :]
    # Round once before rotating, the way ir.ops.rms_norm hands a bf16
    # tensor to rotary_embedding.  Skipping it would leave the whole
    # epilogue in fp32 and drift further from the unfused result.
    y = y.to(qkv_ptr.dtype.element_ty).to(tl.float32)
    if ROT_DIM > 0:
        if ROT_DIVIDES:
            y = _partial_rope(
                y,
                t,
                t_mask,
                tm,
                positions_ptr,
                cos_sin_ptr,
                cache_stride,
                BLOCK_T,
                BLOCK_D,
                ROT_DIM,
                HALF_ROT,
            )
        else:
            y = _neox_partial_rope_gather(
                y,
                qkv_ptr,
                weight_ptr,
                positions_ptr,
                cos_sin_ptr,
                col,
                qkv_row_stride,
                rstd,
                t,
                t_mask,
                tm,
                d,
                cache_stride,
                ROT_DIM,
                HALF_ROT,
                WEIGHT_OFFSET,
            )
    return y


@triton.jit
def _scatter_kv_head(
    value,
    cache_ptr,
    slot_mapping_ptr,
    head_offset,
    d,
    t,
    t_mask,
    dm,
    block_stride,
    token_stride,
    num_slot_rows,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter one head, skipping negative slots and padded rows."""
    page = tl.load(
        slot_mapping_ptr + t, mask=t_mask & (t < num_slot_rows), other=-1
    ).to(tl.int64)
    valid = (page >= 0)[:, None] & dm
    ptr = (
        cache_ptr
        + (page // BLOCK_SIZE)[:, None] * block_stride
        + (page % BLOCK_SIZE)[:, None] * token_stride
        + head_offset
        + d[None, :]
    )
    tl.store(ptr, value.to(cache_ptr.dtype.element_ty), mask=valid)


@triton.jit
def _qknorm_rope_cache_kernel(
    qkv_ptr,
    positions_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    q_out_ptr,
    k_out_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    qkv_row_stride,
    cache_stride,
    kc_block_stride,
    kc_token_stride,
    kc_head_stride,
    vc_block_stride,
    vc_token_stride,
    vc_head_stride,
    num_tokens,
    num_slot_rows,
    Q_SIZE: tl.constexpr,
    KV_SIZE: tl.constexpr,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROT_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    ROT_DIVIDES: tl.constexpr,
    EPS: tl.constexpr,
    WEIGHT_OFFSET: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    WRITE_CACHE: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Assign each program one Q, K or V head and a tile of tokens."""
    slot = tl.program_id(0)
    pid_t = tl.program_id(1)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t < num_tokens
    tm = t_mask[:, None]
    d = tl.arange(0, BLOCK_D)
    dm = (d < HEAD_DIM)[None, :]
    load_mask = tm & dm

    if slot < NQ:
        col = slot * HEAD_DIM
        x = tl.load(
            qkv_ptr + t[:, None] * qkv_row_stride + col + d[None, :],
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        y = _gemma_norm_partial_rope(
            x,
            qkv_ptr,
            q_weight_ptr,
            positions_ptr,
            cos_sin_ptr,
            col,
            qkv_row_stride,
            t,
            t_mask,
            tm,
            d,
            cache_stride,
            BLOCK_T,
            BLOCK_D,
            HEAD_DIM,
            ROT_DIM,
            HALF_ROT,
            ROT_DIVIDES,
            EPS,
            WEIGHT_OFFSET,
        )
        tl.store(
            q_out_ptr + t[:, None] * Q_SIZE + col + d[None, :],
            y.to(q_out_ptr.dtype.element_ty),
            mask=load_mask,
        )
    elif slot < NQ + NK:
        head_col = (slot - NQ) * HEAD_DIM
        col = Q_SIZE + head_col
        x = tl.load(
            qkv_ptr + t[:, None] * qkv_row_stride + col + d[None, :],
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        y = _gemma_norm_partial_rope(
            x,
            qkv_ptr,
            k_weight_ptr,
            positions_ptr,
            cos_sin_ptr,
            col,
            qkv_row_stride,
            t,
            t_mask,
            tm,
            d,
            cache_stride,
            BLOCK_T,
            BLOCK_D,
            HEAD_DIM,
            ROT_DIM,
            HALF_ROT,
            ROT_DIVIDES,
            EPS,
            WEIGHT_OFFSET,
        )
        out = y.to(k_out_ptr.dtype.element_ty)
        tl.store(
            k_out_ptr + t[:, None] * KV_SIZE + head_col + d[None, :],
            out,
            mask=load_mask,
        )
        if WRITE_CACHE:
            _scatter_kv_head(
                out,
                key_cache_ptr,
                slot_mapping_ptr,
                (slot - NQ) * kc_head_stride,
                d,
                t,
                t_mask,
                dm,
                kc_block_stride,
                kc_token_stride,
                num_slot_rows,
                BLOCK_SIZE,
            )
    else:
        head_col = (slot - NQ - NK) * HEAD_DIM
        col = Q_SIZE + KV_SIZE + head_col
        raw = tl.load(
            qkv_ptr + t[:, None] * qkv_row_stride + col + d[None, :],
            mask=load_mask,
            other=0.0,
        )
        if WRITE_CACHE:
            _scatter_kv_head(
                raw,
                value_cache_ptr,
                slot_mapping_ptr,
                (slot - NQ - NK) * vc_head_stride,
                d,
                t,
                t_mask,
                dm,
                vc_block_stride,
                vc_token_stride,
                num_slot_rows,
                BLOCK_SIZE,
            )


def _fused_prepare(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    norm_weight_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare Q and K and scatter K/V into the caller-owned cache."""
    rows = qkv.shape[0]
    # Full RoPE pairs two 96-wide halves. The specialized load is faster
    # for the measured large sliding shapes with one token per program.
    if (
        rotary_dim == 192
        and rows >= 512
        and (num_q_heads, num_kv_heads)
        in (
            (16, 1),
            (8, 2),
        )
    ):
        from .sliding_qknorm_rope_cache import sliding_paired

        q, k, _ = sliding_paired(
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
            192,
            rotary_dim,
            eps,
            norm_weight_bias,
            tile=1,
        )
        return q, k
    q_size, kv_size = num_q_heads * 192, num_kv_heads * 192
    q = qkv.new_empty((rows, q_size))
    k = qkv.new_empty((rows, kv_size))
    _qknorm_rope_cache_kernel[(num_q_heads + 2 * num_kv_heads, rows)](
        qkv,
        positions,
        q_weight,
        k_weight,
        cos_sin_cache,
        q,
        k,
        key_cache,
        value_cache,
        slot_mapping,
        qkv.stride(0),
        cos_sin_cache.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        rows,
        slot_mapping.numel(),
        Q_SIZE=q_size,
        KV_SIZE=kv_size,
        NQ=num_q_heads,
        NK=num_kv_heads,
        HEAD_DIM=192,
        BLOCK_D=256,
        ROT_DIM=rotary_dim,
        HALF_ROT=rotary_dim // 2,
        ROT_DIVIDES=rotary_dim == 64,
        EPS=eps,
        WEIGHT_OFFSET=norm_weight_bias,
        BLOCK_SIZE=key_cache.shape[1],
        WRITE_CACHE=True,
        BLOCK_T=1,
        num_warps=1,
    )
    return q, k


def _fused_prepare_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    norm_weight_bias: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        qkv.new_empty((qkv.shape[0], num_q_heads * 192)),
        qkv.new_empty((qkv.shape[0], num_kv_heads * 192)),
    )


_registered = False


def register_ops() -> None:
    """Register the cache mutations before model compilation."""
    global _registered
    if _registered:
        return
    direct_register_custom_op(
        op_name="ppu_step5_qknorm_rope_cache",
        op_func=_fused_prepare,
        mutates_args=["key_cache", "value_cache"],
        fake_impl=_fused_prepare_fake,
    )
    _registered = True
