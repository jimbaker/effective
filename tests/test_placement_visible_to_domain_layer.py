"""Does a DOMAIN layer see the PLACED key? The assumption the telemetry join rests on.

ROLE: conformance. The telemetry join keys spans to the tape by having `traced` read
`current_placement()`. That design rests on one assumption:
**a domain layer executes inside `placement_scope`'s dynamic extent.**

Two moments matter, and the second is the one that could fail quietly. `traced` builds its span
*after* `yield op` returns (`telemetry.py:617-628`), so the scope must still be live when the layer
RESUMES as well as when it is entered. A cell torn down at the inner dispatch's exit would give a
placement on the way in, `None` on the way out, and a span keyed `None` on every op.

The scoped arm is the other half: a placement that carried no frames would be the LOCAL key wearing
the placed key's name, and `tool:{name}` alone addresses nothing.
"""

from uuid import uuid4

import pytest
from _conformance import CountingDomain, Fault

from effective.api import call_tool, scoped
from effective.keys import compose_key
from effective.layers import compose_domain, current_placement, domain_layer

pytestmark = pytest.mark.conformance

SCOPE = "probe"


def placement_probe_wf(run_id: str):
    """One op at top level, one under a scope — so the frames are observable, not just a key."""
    a = yield from call_tool("a", {}, int)
    b = yield from scoped(compose_key(t"probe:1"), lambda: call_tool("b", {}, int))
    return {"a": a, "b": b}


def recording_layer(seen: list[tuple[str, str | None, str | None]]):
    """Record the placement on the way IN and on the way OUT — `traced` needs the second."""

    @domain_layer
    def run(op):
        before = current_placement()
        out = yield op
        after = current_placement()
        seen.append(
            (
                op.name,
                None if before is None else before.stored(),
                None if after is None else after.stored(),
            )
        )
        return out

    return run


def test_a_domain_layer_sees_the_placed_key_on_both_engines(backend):
    seen: list[tuple[str, str | None, str | None]] = []
    run_id = f"r-{uuid4().hex[:8]}"
    name = f"place-{run_id}"
    # COMPOSED INTO THE DOMAIN, not passed as `layers=`. That argument installs OP layers, whose
    # alphabet is `Step` — a probe there sees the walk's name and proves nothing about the seam
    # `traced` actually occupies. Measured the wrong thing once before this line existed.
    domain = compose_domain((recording_layer(seen),), CountingDomain())
    backend.register(name, placement_probe_wf, domain, Fault(None), ())
    snap = backend.run_until_result(backend.spawn(name, run_id))

    assert snap is not None
    assert snap.state == "completed", snap
    assert [op for op, _, _ in seen] == ["a", "b"], seen  # CallTool.name, the tool's own name

    placements: dict[str, str] = {}
    for op, before, after in seen:
        assert before is not None, f"{op}: no placement on the way IN"
        assert after is not None, f"{op}: no placement on the way OUT — a span built after the "
        assert before == after, f"{op}: placement changed across the yield ({before} -> {after})"
        # Collected HERE rather than by a second comprehension: the assertion above is what makes
        # the value a `str`, and rebuilding the dict outside the loop throws that narrowing away.
        placements[op] = before

    # THE POINT: the scoped op's address carries its frame, so the two ops are distinguishable by
    # address alone. A local key would make both of them `tool:...` and address nothing.
    assert f"{SCOPE}:1" in placements["b"], placements["b"]
    assert f"{SCOPE}:1" not in placements["a"], placements["a"]
    assert placements["a"] != placements["b"]
