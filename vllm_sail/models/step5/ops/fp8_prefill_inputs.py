# SPDX-License-Identifier: Apache-2.0
"""Prepare dense FP8 prefill inputs from SAIL's interleaved BF16 KV pages."""

from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_fp8_prefill_inputs(
    Cache,
    Pages,
    Regions,
    Key,
    Value,
    Slots,
    cache_page_stride,
    cache_token_stride,
    page_stride,
    region_row_stride,
    region_col_stride,
    kv_len,
    num_slots,
    num_pages,
    kv_tiles,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    SLOTS: tl.constexpr,
    DENSE_REGIONS: tl.constexpr,
    WIDE_SLOTS: tl.constexpr,
):
    tile = tl.program_id(0)
    if tile < kv_tiles:
        token = tile * ROWS + tl.arange(0, ROWS)
        page = tl.load(
            Pages + token // PAGE_SIZE * page_stride, token < kv_len, other=0
        )
        # Match the clamped gather. The region path masks invalid pages.
        page = tl.maximum(0, tl.minimum(page, num_pages - 1)).to(tl.int64)
        offset = page * cache_page_stride + token % PAGE_SIZE * cache_token_stride
        col = tl.arange(0, 256)
        live = (token[:, None] < kv_len) & (col[None, :] < 192)
        src = Cache + offset[:, None] + col[None, :]
        dst = token[:, None].to(tl.int64) * 192 + col[None, :]
        key = tl.load(src, live, other=0)
        value = tl.load(src + 192, live, other=0)
        tl.store(Key + dst, key, live)
        tl.store(Value + dst, value, live)
    else:
        slot_tile = tile - kv_tiles
        if WIDE_SLOTS:
            slot_tile = slot_tile.to(tl.int64)
        slot = slot_tile * SLOTS + tl.arange(0, SLOTS)
        slot_live = slot < num_slots
        if DENSE_REGIONS:
            region_offset = slot
        else:
            region_offset = (slot // WIDTH).to(tl.int64) * region_row_stride
            region_offset += (slot % WIDTH).to(tl.int64) * region_col_stride
        region = tl.load(
            Regions + region_offset,
            slot_live,
            other=-1,
        )
        valid = (region >= 0) & (region <= (kv_len - 1) // 8)
        safe_region = tl.where(valid, region, 0).to(tl.int32)
        slot_page = tl.load(
            Pages + (safe_region * 8 // PAGE_SIZE).to(tl.int64) * page_stride,
            valid,
            other=-1,
        )
        tl.store(
            Slots + slot,
            tl.where(valid & (slot_page >= 0) & (slot_page < num_pages), region, -1),
            slot_live,
        )


def prepare_fp8_prefill_inputs(cache, pages, regions, kv_len):
    """Gather K/V and mask invalid region IDs with one launch."""
    import torch

    rows, width = regions.shape
    kv = cache.new_empty((2, kv_len, 1, 192))
    slots = torch.empty((rows, 1, width), dtype=torch.int32, device=regions.device)
    kv_tiles = triton.cdiv(kv_len, 8)
    num_slots = rows * width
    _prepare_fp8_prefill_inputs[(kv_tiles + triton.cdiv(num_slots, 256),)](
        cache,
        pages,
        regions,
        kv[0],
        kv[1],
        slots,
        cache.stride(0),
        cache.stride(2),
        pages.stride(0),
        *regions.stride(),
        kv_len,
        num_slots,
        cache.shape[0],
        kv_tiles,
        PAGE_SIZE=cache.shape[2],
        WIDTH=width,
        ROWS=8,
        SLOTS=256,
        DENSE_REGIONS=regions.is_contiguous(),
        WIDE_SLOTS=num_slots >= 2**31,
    )
    return kv[0], kv[1], slots
