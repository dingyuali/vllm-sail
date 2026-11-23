# SPDX-License-Identifier: Apache-2.0
"""Check that draft coordination consumes the collective mismatch decision."""

import ast
from contextvars import ContextVar
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "ticket, local, peer",
    [
        (False, False, False),
        (True, False, False),
        (True, False, True),
        (True, True, False),
    ],
)
def test_draft_prediction_fallback(ticket, local, peer):
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm_sail/patch/performance/step5_dp_coordination.py"
    )
    function = next(
        node
        for node in ast.walk(ast.parse(path.read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == "coordinate_draft"
    )
    function.decorator_list = []
    group = object()
    mode = object()
    votes = []
    fallbacks = []
    draft = ContextVar("draft", default=None)
    if ticket:
        draft.set(((4 if local else 8, 8, 1), "cached padding", 1))

    def reduce(value, *, op, group):
        assert op is mode and group is expected_group
        votes.append(value.item())
        value.item = lambda: int(local or peer)

    def coordinate(*args):
        fallbacks.append(args)
        return "native result"

    expected_group = group
    namespace = dict(
        draft_context=draft,
        torch=SimpleNamespace(
            tensor=lambda value, **kwargs: SimpleNamespace(item=lambda: value),
            int32=object(),
        ),
        dist=SimpleNamespace(all_reduce=reduce, ReduceOp=SimpleNamespace(MAX=mode)),
        dp_utils=SimpleNamespace(_get_device_and_group=lambda _: ("cpu", group)),
        coordinate=coordinate,
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    result = namespace["coordinate_draft"](8, False, "dp", 8, True, 1)

    assert votes == ([int(local)] if ticket else [])
    assert draft.get() is None
    if ticket and not (local or peer):
        assert result == (False, "cached padding", 1)
        assert not fallbacks
    else:
        assert result == "native result"
        assert fallbacks == [(8, False, "dp", 8, True, 1)]
