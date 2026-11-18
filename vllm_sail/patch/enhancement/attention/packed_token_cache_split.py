# SPDX-License-Identifier: Apache-2.0
"""Split a block-table-written packed cache into its kernel blocks.

Kimi-K3.1 packs the DSA indexer pages beside the MLA pages of the same block row
(``KimiK3IndexerBackend`` only supports BLHNC), so a KDA state page sizes the
whole row and ``Platform._align_hybrid_block_size`` raises the manager block
size far past the 64-token pages the indexer kernels address. The indexer
therefore writes its keys at slots derived from the block table
(``KimiK3IndexerMetadataBuilder``): the contract
``AttentionGroup.map_kv_cache_to_kernel_blocks`` describes as "compressed
caches, written from their block table".

The guard there derives that contract from ``tokens_per_state`` alone and
refuses every cache that stores one state per token, so binding the packed
indexer cache fails with "cannot be split into". A backend that declares
``writes_kv_cache_from_block_table`` meets the contract without compression, so
take the same split path for it. Every other cache, and every other platform,
keeps upstream's check.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.worker.utils import AttentionGroup

from vllm_sail.patch.utils import patch

logger = init_logger(__name__)

_AFFECTED = ">=0.32.0,<0.33.0"

# Captured before patching so the delegating path stays upstream.
_upstream_map_kv_cache_to_kernel_blocks = AttentionGroup.map_kv_cache_to_kernel_blocks

_reported: set[str] = set()


def _report_once(key: str, message: str, *args) -> None:
    """One line per backend, so a split that falls back stays diagnosable."""
    if key in _reported:
        return
    _reported.add(key)
    logger.info(message, *args)


@patch(
    "vllm.v1.worker.utils",
    "AttentionGroup.map_kv_cache_to_kernel_blocks",
    reason=(
        "Kimi-K3.1's packed DSA indexer cache holds one state per token and is "
        "written from its block table, but the guard infers that from "
        "tokens_per_state instead of consulting writes_kv_cache_from_block_table, "
        "so the packed cache cannot be split into the 64-token kernel blocks its "
        "kernels address and KV cache binding fails."
    ),
    affected_versions=_AFFECTED,
    remove_when=(
        "map_kv_cache_to_kernel_blocks asks the backend "
        "(writes_kv_cache_from_block_table) instead of inferring it from "
        "tokens_per_state."
    ),
)
def map_kv_cache_to_kernel_blocks(self, kv_cache):
    writes_from_block_table = getattr(
        self.backend, "writes_kv_cache_from_block_table", None
    )
    is_ppu = current_platform.is_ppu()
    declared = None if writes_from_block_table is None else writes_from_block_table()
    backend_name = self.backend.get_name()
    if not (is_ppu and declared):
        _report_once(
            f"skip:{backend_name}",
            "Packed kernel-block split skipped for %s (ppu=%s, "
            "writes_kv_cache_from_block_table=%s).",
            backend_name,
            is_ppu,
            declared,
        )
        return _upstream_map_kv_cache_to_kernel_blocks(self, kv_cache)
    # PPU MODIFICATION: begin — verbatim upstream body with the
    # tokens_per_state test dropped for a backend that writes its pages from
    # the block table; the shape and stride tests still decide whether the
    # split is addressable, and an unaddressable cache keeps upstream's error.
    assert self.kernel_block_size is not None
    spec = self.kv_cache_spec
    kernel_rows = spec.get_num_kernel_states(self.kernel_block_size)
    if kv_cache.shape[-2] == kernel_rows:
        return kv_cache
    if kv_cache.shape[1] != 1 or kv_cache.stride(0) % (
        kernel_rows * kv_cache.stride(-2)
    ):
        _report_once(
            f"unsplittable:{backend_name}",
            "Packed %s cache cannot be split into %d-token kernel blocks "
            "(shape=%s, strides=%s); keeping upstream's error.",
            backend_name,
            self.kernel_block_size,
            tuple(kv_cache.shape),
            kv_cache.stride(),
        )
        return _upstream_map_kv_cache_to_kernel_blocks(self, kv_cache)
    # Kernel block j of block b is b * kernel_block_stride + j. The view spans
    # the other layers' pages between blocks (never addressing them), which
    # ``view`` cannot express.
    page_stride = kernel_rows * kv_cache.stride(-2)
    self.kernel_block_stride = kv_cache.stride(0) // page_stride
    num_pages = (kv_cache.shape[0] - 1) * self.kernel_block_stride + (
        kv_cache.shape[-2] // kernel_rows
    )
    _report_once(
        f"split:{backend_name}",
        "Split %s cache into %d-token kernel blocks: block_size=%d shape=%s "
        "strides=%s pages=%d kernel_block_stride=%d.",
        backend_name,
        self.kernel_block_size,
        spec.block_size,
        tuple(kv_cache.shape),
        kv_cache.stride(),
        num_pages,
        self.kernel_block_stride,
    )
    return kv_cache.as_strided(
        (num_pages, *kv_cache.shape[1:-2], kernel_rows, kv_cache.shape[-1]),
        (page_stride, *kv_cache.stride()[1:]),
    )
    # PPU MODIFICATION: end


logger.info(
    "Packed kernel-block split for block-table-written KV caches installed."
)
