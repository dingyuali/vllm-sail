# SPDX-License-Identifier: Apache-2.0
"""Stable Step5 prefill selection for rows with a single cutoff slot."""

from vllm.models.step5.nvidia.ops.sparse_indexer import _stable_topk_regions
from vllm.triton_utils import tl, triton


@triton.jit
def _stable_prefill_regions(
    Scores,
    Current,
    Candidates,
    Scratch,
    Output,
    score_stride,
    width,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK: tl.constexpr,
    SCAN: tl.constexpr,
):
    row = tl.program_id(0)
    length = tl.load(Current + row).to(tl.int32)
    col = tl.arange(0, BLOCK_K)
    candidate = tl.load(Candidates + row * (K + 1) + col, col < K, other=-1)
    valid = (col < K) & (candidate >= 0) & (candidate < length)
    values = tl.load(Scores + row * score_stride + candidate, valid, other=float("inf"))
    threshold = tl.min(values, 0)
    above = valid & (values > threshold)
    above_count = tl.sum(above.to(tl.int32), 0)
    valid_count = tl.sum(valid.to(tl.int32), 0)
    if (
        (length > K)
        & (length <= width)
        & (valid_count == K)
        & (above_count == K - 1)
        & (tl.abs(threshold) < float("inf"))
    ):
        # Find the smallest region at the cutoff across the entire history.
        # The radix selector can return any tied candidate in this final slot.
        region = tl.arange(0, SCAN)
        history = tl.load(
            Scores + row * score_stride + region, region < length, other=0
        )
        first = tl.min(
            tl.where((region < length) & (history == threshold), region, SCAN), 0
        )
        chosen = tl.where(above, candidate, tl.where(valid, first, 2147483647))
        ordered = tl.sort(chosen, descending=False)
        tl.store(Output + row * (K + 1) + col, ordered, col < K)
        tl.store(Output + row * (K + 1) + K, length)
    else:
        _stable_topk_regions(
            Scores,
            Current,
            Candidates,
            Scratch,
            Output,
            score_stride,
            K + 1,
            K,
            BLOCK_K,
            1024,
            BLOCK,
        )
