# SPDX-License-Identifier: Apache-2.0
"""Paired full-RoPE fusion for Step5 sliding layers."""

from vllm.triton_utils import tl, triton


@triton.jit
def _sliding_paired(
    QKV,
    QW,
    KW,
    Rope,
    Pos,
    Slots,
    Q,
    K,
    KC,
    VC,
    rows,
    slot_count,
    row_stride,
    rope_stride,
    kb,
    kt,
    kh,
    vb,
    vt,
    vh,
    NQ: tl.constexpr,
    NK: tl.constexpr,
    PAGE: tl.constexpr,
    EPS: tl.constexpr,
    BIAS: tl.constexpr,
    TILE: tl.constexpr,
):
    head = tl.program_id(0)
    row = tl.program_id(1) * TILE + tl.arange(0, TILE)
    pair = tl.arange(0, 128)
    live = (row[:, None] < rows) & (pair[None, :] < 96)
    src = QKV + row[:, None] * row_stride + head * 192 + pair[None, :]
    x = tl.load(src, live, other=0)
    z = tl.load(src + 96, live, other=0)
    if head < NQ + NK:
        weight = QW if head < NQ else KW
        a, b = x.to(tl.float32), z.to(tl.float32)
        sums = tl.sum(a * a, axis=1) + tl.sum(b * b, axis=1)
        rstd = tl.rsqrt(sums / 192.0 + EPS)[:, None]
        wa = tl.load(weight + pair, pair < 96, other=0).to(tl.float32) + BIAS
        wb = tl.load(weight + 96 + pair, pair < 96, other=0).to(tl.float32) + BIAS
        a = (a * rstd * wa[None, :]).to(x.dtype).to(tl.float32)
        b = (b * rstd * wb[None, :]).to(z.dtype).to(tl.float32)
        pos = tl.load(Pos + row, row < rows, other=0).to(tl.int64)
        rope = Rope + pos[:, None] * rope_stride + pair[None, :]
        c = tl.load(rope, live, other=0).to(tl.float32)
        s = tl.load(rope + 96, live, other=0).to(tl.float32)
        x = (a * c - b * s).to(x.dtype)
        z = (b * c + a * s).to(z.dtype)
        if head < NQ:
            out = Q + row[:, None] * (NQ * 192) + head * 192 + pair[None, :]
        else:
            out = K + row[:, None] * (NK * 192) + (head - NQ) * 192 + pair[None, :]
        tl.store(out, x, live)
        tl.store(out + 96, z, live)
    if head >= NQ:
        slot = tl.load(Slots + row, (row < rows) & (row < slot_count), other=-1).to(
            tl.int64
        )
        live = live & (slot[:, None] >= 0)
        page = (slot // PAGE)[:, None]
        offset = (slot % PAGE)[:, None]
        if head < NQ + NK:
            dst = KC + page * kb + offset * kt + (head - NQ) * kh + pair[None, :]
        else:
            dst = VC + page * vb + offset * vt + (head - NQ - NK) * vh + pair[None, :]
        tl.store(dst, x, live)
        tl.store(dst + 96, z, live)


def sliding_paired(
    qkv,
    qw,
    kw,
    rope,
    positions,
    slots,
    kc,
    vc,
    heads,
    kv_heads,
    head_dim,
    rotary_dim,
    eps,
    bias,
    *,
    tile,
):
    assert head_dim == rotary_dim == 192
    rows = qkv.shape[0]
    q = qkv.new_empty((rows, heads * 192))
    k = qkv.new_empty((rows, kv_heads * 192))
    _sliding_paired[(heads + 2 * kv_heads, triton.cdiv(rows, tile))](
        qkv,
        qw,
        kw,
        rope,
        positions,
        slots,
        q,
        k,
        kc,
        vc,
        rows,
        slots.numel(),
        qkv.stride(0),
        rope.stride(0),
        *kc.stride()[:3],
        *vc.stride()[:3],
        heads,
        kv_heads,
        kc.shape[1],
        eps,
        bias,
        tile,
        num_warps=1,
    )
    return q, k, qkv[:, (heads + kv_heads) * 192 :]
