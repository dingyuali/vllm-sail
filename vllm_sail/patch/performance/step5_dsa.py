# SPDX-License-Identifier: Apache-2.0
"""Route Step5's DSA indexer scoring kernel to the PPU resident-block version.

Upstream's ``_paged_scores_sm90`` launches one block per ``(row, tile)``, so
every row reloads its query tile and gate ``cdiv(max_seq_len, TILE)`` times
(258 times at the production shape). The replacement lives in
``vllm_sail/models/step5/ops/sparse_attention_opt.py``.

The only consumer, ``vllm.models.step5.nvidia.flashinfer_sparse``, imports
``paged_mqa_logits_sm90`` lazily inside ``Step5FlashInferSparse.__init__``, so
replacing the module attribute is enough; no body needs to be copied.

Turning the switch off or removing this patch only costs performance, never
correctness, hence the performance category.
"""

from __future__ import annotations

import functools
import importlib
import sys

from vllm.platforms import current_platform

from vllm_sail import envs
from vllm_sail.models.step5.ops import score_pipeline
from vllm_sail.patch.utils import patch
from vllm_sail.step5_sparse_cases import profiled

_MODULE = "vllm.models.step5.nvidia.ops.sparse_indexer_scores"
_CONSUMERS = ("vllm.models.step5.nvidia.flashinfer_sparse",)
_NAME = "paged_mqa_logits_sm90"
_REASON = (
    "Step5's indexer scorer launches one block per (row, region tile), which "
    "reloads the row's query tile and gate once per tile -- 258 times on the "
    "production shape. The plugin keeps blocks resident per row instead "
    "(3.188 -> 2.070 ms on ZW-M890P). Opt-in via VLLM_SAIL_STEP5_DSA_SCORES=v1/v2/v3."
)
_AFFECTED = ">=0.33.0,<0.34.0"
_REMOVE_WHEN = (
    "Step5's _paged_scores_sm90 itself strides over region tiles from a "
    "resident grid, making this replacement redundant."
)
_installed = False
# Per-bucket worker counts measured with short and long histories.
_VERIFY_WORKERS = {32: 16, 64: 8, 96: 4, 128: 4}


def _wrap_decode_scores(module, original):
    @functools.wraps(original)
    def decode_summary_scores(
        query,
        weights,
        mean,
        block_table,
        slots,
        seq_lens,
        capacity: int,
        *,
        num_sms: int,
        row_groups: int = 1,
        query_len: int = 1,
    ):
        import torch
        from vllm.triton_utils import triton

        # PPU MODIFICATION: begin — dispatch only the measured FP8 verification buckets.
        if not (
            current_platform.is_ppu()
            and envs.VLLM_SAIL_STEP5_DSA_SCORES in ("v1", "v2")
            and envs.VLLM_SAIL_STEP5_DSA_SCORE_DECODE == "v0"
            and num_sms == 39
            and query_len == 4
            and row_groups == 1
            and query.shape[0] in _VERIFY_WORKERS
            and query.dtype == torch.float8_e4m3fn
            and mean.dtype == torch.uint8
            and mean.shape[1] == 8
            and 8192 <= capacity <= 131072
            and (capacity <= 16384 or query.shape[0] in (32, 64, 128))
            and current_platform.is_device_capability(89)
        ):
            return original(
                query, weights, mean, block_table, slots, seq_lens, capacity,
                num_sms=num_sms, row_groups=row_groups, query_len=query_len,
            )
        # PPU MODIFICATION: end

        tokens, groups = query.shape[:2]
        assert groups == row_groups and query.stride(3) == 1 and query.stride(2) == 256
        assert weights.shape == query.shape[:3] and weights.stride(2) == 1
        assert mean.stride(1) == 256 and mean.stride(2) == 1
        batch = tokens * groups
        scores = torch.empty(
            (batch, triton.cdiv(capacity, 256) * 256),
            device=query.device,
            dtype=torch.float32,
        )
        active = torch.empty(batch, device=query.device, dtype=torch.bool)
        positions = torch.empty(batch, device=query.device, dtype=torch.int64)
        current = torch.empty_like(positions)
        lengths = torch.empty(batch, device=query.device, dtype=torch.int32)
        topk_lengths = torch.empty_like(lengths)
        # PPU MODIFICATION: begin — select the M890 verification launch schedule.
        # Larger grids improve memory parallelism when each request touches
        # different cache pages in the 1M-context M=64 verification bucket.
        workers = 19 if batch == 64 and capacity == 131072 else _VERIFY_WORKERS[batch]
        workers = min(triton.cdiv(capacity, 32), workers)
        module._direct_scores[(workers, batch)](
            query,
            weights,
            mean,
            block_table,
            slots,
            seq_lens,
            scores,
            active,
            positions,
            current,
            lengths,
            topk_lengths,
            query.stride(0),
            query.stride(1),
            weights.stride(0),
            weights.stride(1),
            block_table.shape[1],
            block_table.stride(0),
            slots.stride(0),
            seq_lens.stride(0),
            mean.shape[0],
            mean.shape[1],
            mean.stride(0),
            capacity,
            scores.stride(0),
            8,
            32,
            workers,
            row_groups,
            query_len,
            num_warps=1,
            enable_fp_fusion=False,
        )
        # PPU MODIFICATION: end
        metadata = module.DecodeIndexerMetadata(
            active, positions, current, lengths, block_table, topk_lengths
        )
        return scores[:, :capacity], metadata

    return decode_summary_scores


def install() -> None:
    global _installed
    if _installed:
        return
    try:
        module = importlib.import_module(_MODULE)
    except ModuleNotFoundError as exc:
        # A vLLM build without the Step5 tree is not an error.
        if exc.name and (_MODULE == exc.name or _MODULE.startswith(exc.name + ".")):
            return
        raise

    original = getattr(module, _NAME)
    optimized = None

    @profiled("score")
    @functools.wraps(original)
    def paged_mqa_logits_sm90(*args, **kwargs):
        # Read the switch per call to keep envs lazy; non-PPU stays upstream.
        # The resident kernel needs hardware E4M3; PPU 1.0 keeps upstream's
        # launcher, which reaches the software-E4M3 scorer.
        if not (
            current_platform.is_ppu()
            and envs.VLLM_SAIL_STEP5_DSA_SCORES in ("v1", "v2", "v3")
            and current_platform.has_device_capability(89)
        ):
            return original(*args, **kwargs)
        nonlocal optimized
        if optimized is None:
            # Deferred to first use so installing the patch never pulls in the
            # triton compilation stack.
            from vllm_sail.models.step5.ops.sparse_attention_opt import (
                paged_mqa_logits_sm90 as optimized_impl,
            )

            optimized = optimized_impl
        return optimized(*args, **kwargs)

    patch(
        _MODULE, _NAME, reason=_REASON,
        affected_versions=_AFFECTED, remove_when=_REMOVE_WHEN,
    )(paged_mqa_logits_sm90)
    # The consumer imports lazily inside __init__ and normally holds no copy.
    # If it already does, replace that too, or its instances keep calling
    # upstream's kernel.
    for consumer in _CONSUMERS:
        loaded = sys.modules.get(consumer)
        if loaded is not None and getattr(loaded, _NAME, None) is original:
            patch(
                consumer, _NAME, reason=_REASON,
                affected_versions=_AFFECTED, remove_when=_REMOVE_WHEN,
            )(paged_mqa_logits_sm90)
    # Only new targets are installed here. Logits/plan/prefill delegation stays
    # composed with the existing FP16/native wrappers, independent of order.
    attention = "vllm.models.step5.nvidia.sparse_attention"
    adapter = _CONSUMERS[0]
    attention_module = importlib.import_module(attention)
    adapter_module = importlib.import_module(adapter)
    impl = attention_module.Step5DSAAttentionImpl
    cls = adapter_module.Step5FlashInferSparse
    for target, name, replacement in (
        (attention, "Step5DSAAttentionImpl.forward",
         score_pipeline.wrap_layer_forward(impl.forward, current_platform)),
        (adapter, "Step5FlashInferSparse.pack_summaries",
         score_pipeline.wrap_layer_pack(cls.pack_summaries)),
    ):
        patch(target, name, reason=score_pipeline.LAYER_REASON,
              affected_versions=_AFFECTED,
              remove_when=score_pipeline.LAYER_REMOVE_WHEN)(replacement)
    name = "decode_summary_scores"
    decode = getattr(module, name, None)
    if decode is not None:
        replacement = _wrap_decode_scores(module, decode)
        metadata = dict(
            reason="Tune resident FP8 direct scores for Step5 M890 verification.",
            affected_versions=_AFFECTED,
            remove_when="Step5 direct summary scoring selects a per-bucket schedule.",
        )
        patch(_MODULE, name, **metadata)(replacement)
        for consumer in _CONSUMERS:
            loaded = sys.modules.get(consumer)
            if loaded is not None and getattr(loaded, name, None) is decode:
                patch(consumer, name, **metadata)(replacement)
    # Grouped delegation composes the existing logits/native prefill entries
    # instead of patching the same target twice.
    _installed = True


install()
