# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 per-token indexer preparation ported from vLLM v0.23."""

from vllm.triton_utils import tl, triton

from vllm_sail.models.step5.ops.soft_fp8 import _indexer_store_value


@triton.jit
def _dsa_indexer_apply_partial_rope(
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
    """NEOX split-half RoPE over the leading ROT_DIM columns of ``y``.

    Two changes against the permuted-index partner read above:

    * the partner comes from a register flip of the
      ``(BLOCK_T, GROUPS, 2, HALF_ROT)`` reshape instead of a second global
      read plus a second normalization -- the Triton form of deepgpu-torch
      taking it from ``__shared__ normed[]``
      (``gpu_kernel/cuda/rms_norm.cuh:2330``);
    * cos/sin are two DENSE ``(BLOCK_T, HALF_ROT)`` loads broadcast in
      registers, instead of two ``(BLOCK_T, BLOCK_D)`` masked gathers whose
      index arithmetic ``where(d < HALF_ROT, d, d - HALF_ROT)`` fetched only
      2*ROT_DIM distinct cache values per token through BLOCK_D predicated
      lanes.
    """
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
def _dsa_indexer_qknorm_rope_kernel_per_token(
    q_ptr,
    k_ptr,
    positions_ptr,
    q_weight_ptr,
    k_weight_ptr,
    k_bias_ptr,
    cos_sin_ptr,
    q_out_ptr,
    k_out_ptr,
    weights_ptr,
    weights_out_ptr,
    weights_stride,
    q_stride_t,
    k_stride_t,
    cache_stride,
    q_col_offset,
    k_col_offset,
    num_tokens,
    Q_HEADS: tl.constexpr,
    PAD_HEADS: tl.constexpr,
    PROXY_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROT_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    EPS: tl.constexpr,
    Q_WEIGHT_OFFSET: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Compute all indexer Q heads and K for each token in one CTA."""
    pid_t = tl.program_id(0)
    ROWS: tl.constexpr = BLOCK_T * Q_HEADS
    QD: tl.constexpr = Q_HEADS * PROXY_DIM
    inv_dim = 1.0 / PROXY_DIM
    d = tl.arange(0, BLOCK_D)

    t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t < num_tokens
    tm = t_mask[:, None]
    qrow = q_ptr + t[:, None] * q_stride_t

    # ---- q: Q_HEADS Gemma-RMSNorm rows as one (ROWS, PROXY_DIM) tile ----
    xq = tl.load(
        qrow + q_col_offset + tl.arange(0, QD)[None, :], mask=tm, other=0.0
    ).to(tl.float32)
    xq = tl.reshape(xq, (ROWS, BLOCK_D))
    rstd = tl.rsqrt(tl.sum(xq * xq, axis=1) * inv_dim + EPS)[:, None]
    wq = tl.load(q_weight_ptr + d).to(tl.float32) + Q_WEIGHT_OFFSET
    yq = xq * rstd * wq[None, :]

    # Row r of the q tile is token pid_t*BLOCK_T + r // Q_HEADS, head
    # r % Q_HEADS. Output rows also account for Sail head padding.
    rows = tl.arange(0, ROWS)
    tq = pid_t * BLOCK_T + rows // Q_HEADS
    tq_mask = tq < num_tokens
    yq = _dsa_indexer_apply_partial_rope(
        yq,
        tq,
        tq_mask,
        tq_mask[:, None],
        positions_ptr,
        cos_sin_ptr,
        cache_stride,
        ROWS,
        BLOCK_D,
        ROT_DIM,
        HALF_ROT,
    )
    head = rows % Q_HEADS
    rounded = yq.to(q_ptr.dtype.element_ty)
    tl.store(
        q_out_ptr + (tq * PAD_HEADS + head)[:, None] * PROXY_DIM + d[None, :],
        _indexer_store_value(rounded, q_out_ptr.dtype.element_ty == tl.float16),
        mask=tq_mask[:, None],
    )

    # Adapt v0.23 outputs to the Sail scorer layout.
    heads = tl.arange(0, PAD_HEADS)
    if Q_HEADS < PAD_HEADS:
        tl.store(
            q_out_ptr
            + (t[:, None, None] * PAD_HEADS + heads[None, :, None]) * PROXY_DIM
            + d[None, None, :],
            0,
            t_mask[:, None, None] & (heads[None, :, None] >= Q_HEADS),
        )
    weights = tl.load(
        weights_ptr + t[:, None] * weights_stride + heads[None, :],
        tm & (heads[None, :] < Q_HEADS),
        other=0,
    )
    tl.store(
        weights_out_ptr + t[:, None] * PAD_HEADS + heads[None, :],
        weights.to(tl.float32) * Q_HEADS**-0.5,
        tm,
    )

    # ---- k: LayerNorm, one row per token ----
    krow = k_ptr + t[:, None] * k_stride_t
    xk = tl.load(krow + k_col_offset + d[None, :], mask=tm, other=0.0).to(tl.float32)
    mean = tl.sum(xk, axis=1) * inv_dim
    xkc = xk - mean[:, None]
    rstdk = tl.rsqrt(tl.sum(xkc * xkc, axis=1) * inv_dim + EPS)[:, None]
    wk = tl.load(k_weight_ptr + d).to(tl.float32)
    bk = tl.load(k_bias_ptr + d).to(tl.float32)
    yk = xkc * rstdk * wk[None, :] + bk[None, :]
    yk = _dsa_indexer_apply_partial_rope(
        yk,
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
    tl.store(
        k_out_ptr + t[:, None] * PROXY_DIM + d[None, :],
        yk.to(k_out_ptr.dtype.element_ty),
        mask=tm,
    )
