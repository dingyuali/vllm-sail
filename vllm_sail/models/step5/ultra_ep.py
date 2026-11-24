# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional UltraEP placement and replica weights for PPU MegaMoE.

Each layer owns peer-visible master weights. Layers with the same geometry
share one manager and one replica pool per EP group, so replica storage does
not grow with the layer count. MegaMoE keeps ownership of token communication,
expert GEMMs and combine, using its separate master/replica weight arguments.
"""

from dataclasses import dataclass, field
from functools import cache
from importlib import import_module
from typing import Any
from weakref import WeakKeyDictionary, WeakValueDictionary

import torch

WeightPair = tuple[torch.Tensor, torch.Tensor]


@cache
def available() -> bool:
    """Discover the optional PPU adapter without importing it on other routes."""
    try:
        module = import_module("ultra_ep.ppu")
    except (ImportError, OSError):
        return False
    return all(
        callable(getattr(module, name, None))
        for name in (
            "Manager",
            "PhysicalWeightPool",
            "ReplicaWeightPool",
            "allocate_local_expert_tensors",
            "allocate_symmetric_expert_tensors",
        )
    )


_CAPABILITIES = WeakKeyDictionary()


def all_ranks_supported(group, supported: bool) -> bool:
    if group.device_group not in _CAPABILITIES:
        flag = torch.tensor(int(supported), dtype=torch.int32, device="cpu")
        torch.distributed.all_reduce(
            flag, op=torch.distributed.ReduceOp.MIN, group=group.cpu_group
        )
        _CAPABILITIES[group.device_group] = bool(flag.item())
    return _CAPABILITIES[group.device_group]


def _manager_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # The PPU JIT has no E8M0 argument type. Copies are byte-exact, while the
    # tensors exposed to MegaMoE retain their original dtype and layout.
    if str(tensor.dtype).startswith("torch.float8_"):
        return tensor.view(torch.uint8)
    return tensor


def _validate_expert_tensor(tensor: torch.Tensor, masters: int) -> None:
    if not tensor.is_cuda or tensor.ndim < 1 or tensor.shape[0] != masters:
        raise ValueError("UltraEP weights must be CUDA tensors with M expert rows")
    if any(int(size) <= 0 for size in tensor.shape):
        raise ValueError("UltraEP does not support empty expert weight rows")
    if any(int(stride) < 0 for stride in tensor.stride()):
        raise ValueError("UltraEP does not support negative weight strides")

    # Permit dense inner-dimension permutations, including transposed scales,
    # without accepting overlaps or holes hidden by an equal storage span.
    row_elements = 1
    dimensions = sorted(
        zip(tensor.shape[1:], tensor.stride()[1:], strict=True), key=lambda x: x[1]
    )
    for size, stride in dimensions:
        if int(size) == 1:
            continue
        if int(stride) != row_elements:
            raise ValueError("UltraEP requires dense, non-overlapping expert rows")
        row_elements *= int(size)
    if int(tensor.stride(0)) != row_elements:
        raise ValueError("UltraEP requires packed expert rows")
    row_bytes = row_elements * tensor.element_size()
    if row_bytes % 128:
        raise ValueError("UltraEP expert rows must be a positive multiple of 128 bytes")

    manager_dtype = _manager_tensor(tensor).dtype
    if manager_dtype not in (
        torch.int8,
        torch.uint8,
        torch.uint16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ):
        raise ValueError(f"UltraEP PPU JIT does not support {tensor.dtype}")


@dataclass
class _SharedResources:
    manager: Any
    l1_replicas: WeightPair
    l2_replicas: WeightPair
    allocation: Any
    layer_ids: set[int] = field(default_factory=set)


_GROUP_RESOURCES: WeakValueDictionary[tuple[Any, ...], _SharedResources] = (
    WeakValueDictionary()
)


def _shared_resources(
    group: Any,
    tensors: tuple[torch.Tensor, ...],
    num_redundant_experts: int,
) -> _SharedResources:
    geometry = tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in tensors)
    key = (
        id(group),
        tensors[0].device,
        geometry,
        num_redundant_experts,
    )
    existing = _GROUP_RESOURCES.get(key)
    if existing is not None:
        return existing

    # Keep the optional dependency out of the normal MegaMoE import path.
    from ultra_ep.ppu import Manager, ReplicaWeightPool, allocate_local_expert_tensors

    manager = Manager(
        group,
        int(tensors[0].shape[0]),
        num_redundant_experts,
        balance_threshold=1.0,
    )
    allocation = allocate_local_expert_tensors(tensors, num_redundant_experts)
    w1, s1, w2, s2 = allocation.tensors
    manager.register_replica_weight_pool(
        ReplicaWeightPool(
            w1=_manager_tensor(w1),
            w1_scale=_manager_tensor(s1),
            w2=_manager_tensor(w2),
            w2_scale=_manager_tensor(s2),
            keepalive=(allocation.backing,),
        )
    )
    shared = _SharedResources(
        manager=manager,
        l1_replicas=(w1, s1),
        l2_replicas=(w2, s2),
        allocation=allocation,
    )
    _GROUP_RESOURCES[key] = shared
    return shared


class PPUMegaMoEUltraEP:
    """Adapt transformed MegaMoE weights and logical top-k IDs to UltraEP.

    Construction is collective and must occur outside graph capture, in the
    same layer order on every EP rank. ``prepare_topk`` is called only after
    every EP rank has agreed to enable replicas for the current prefill.
    """

    def __init__(
        self,
        group: Any,
        layer_id: int,
        l1_weights: WeightPair,
        l2_weights: WeightPair,
        num_redundant_experts: int,
    ) -> None:
        if len(l1_weights) != 2 or len(l2_weights) != 2:
            raise ValueError("UltraEP expects weight/scale pairs for both GEMMs")
        tensors = (*l1_weights, *l2_weights)
        if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("UltraEP weight/scale pairs must contain tensors")
        if tensors[0].ndim < 1 or int(tensors[0].shape[0]) <= 0:
            raise ValueError("UltraEP needs a positive local master expert count")
        masters = int(tensors[0].shape[0])
        for tensor in tensors:
            _validate_expert_tensor(tensor, masters)
        device = tensors[0].device
        if any(tensor.device != device for tensor in tensors):
            raise ValueError("UltraEP weight tensors must share one device")
        if device.index != torch.accelerator.current_device_index():
            raise ValueError("UltraEP weights must belong to the current CUDA device")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "UltraEP weights must be initialized before graph capture"
            )

        redundant = int(num_redundant_experts)
        if redundant <= 0:
            raise ValueError("UltraEP redundant experts must be positive")

        from ultra_ep.ppu import PhysicalWeightPool, allocate_symmetric_expert_tensors

        # Symmetric pools use the driver allocator. Return cached Torch weight
        # blocks from earlier layers before allocating their replacements.
        # This is shared by FP4/FP8 and runs only during weight initialization.
        torch.accelerator.empty_cache()
        shared = _shared_resources(group, tensors, redundant)
        self.layer_id = int(layer_id)
        if self.layer_id in shared.layer_ids:
            raise RuntimeError(f"UltraEP layer {self.layer_id} is already registered")
        allocation = allocate_symmetric_expert_tensors(group, tensors, 0)
        w1, s1, w2, s2 = allocation.tensors
        shared.manager.register_physical_weight_pool(
            self.layer_id,
            PhysicalWeightPool(
                w1=_manager_tensor(w1),
                w1_scale=_manager_tensor(s1),
                w1_peer_base_ptrs=allocation.peer_base_ptrs,
                w2=_manager_tensor(w2),
                w2_scale=_manager_tensor(s2),
                w2_peer_base_ptrs=allocation.peer_base_ptrs,
                keepalive=(allocation.backing,),
            ),
        )
        shared.layer_ids.add(self.layer_id)

        self._shared = shared
        self._allocation = allocation
        self.manager = shared.manager
        self.l1_weights = (w1, s1)
        self.l2_weights = (w2, s2)
        self.l1_weights_replicas = shared.l1_replicas
        self.l2_weights_replicas = shared.l2_replicas
        self.num_local_master_experts = masters
        self.num_local_redundant_experts = redundant
        self.num_local_physical_experts = masters + redundant
        self.num_physical_experts = (
            shared.manager.num_ranks * self.num_local_physical_experts
        )
        self.shared_replica_storage_bytes = int(shared.allocation.backing.numel())
        self.device = device

    def prepare_topk(self, topk_ids: torch.Tensor) -> torch.Tensor:
        """Return physical int64 IDs without changing caller-owned logical IDs.

        Inputs contain valid logical expert IDs or the router's -1 sentinel.
        Decode and small/mixed batches bypass this adapter entirely. Prefill
        placement must execute outside graph replay.
        """
        if (
            topk_ids.device != self.device
            or topk_ids.dtype not in (torch.int32, torch.int64)
            or topk_ids.ndim != 2
            or not topk_ids.is_contiguous()
        ):
            raise ValueError(
                "UltraEP top-k IDs must be contiguous int32/int64 device matrices"
            )
        ids = (
            topk_ids.clone()
            if topk_ids.dtype == torch.int64
            else topk_ids.to(torch.int64)
        )
        self.manager.update_placement(self.layer_id, ids)
        event = self.manager.weight_sync(self.layer_id, async_finish=True)
        physical = self.manager.reroute(self.layer_id, ids)
        if event is not None:
            event.current_stream_wait()
        return physical
