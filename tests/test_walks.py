"""Walk-invariance: one program, every interpreter, and they must agree.

The harness is `_walks.py`; this is what it is for. Three properties, in the order a reader
should meet them:

1. **the ambient probe** — every walk establishes the same per-run state, which is the assertion
   that would have caught `ReplayHandler` establishing none;
2. **the authority park** — every walk that can park binds the same name, the property an
   aliasing defect breaks;
3. **the harness can fail** — a walk that disagrees is reported, because an instrument that
   cannot go red is not evidence.

The walks are an inductive family (each a fold over one op stream, recursing the same way into a
`Scoped` body and a `Gather` branch), so these are deep slices parameterized over the family
rather than a case written per member. That is the point: a member nobody wrote a case for is a
member nobody notices is missing, which is how the walk count came to be wrong three times.
"""

from typing import Any

import pytest
from _walks import (
    ALL_WALK_NAMES,
    ALL_WALKS,
    Divergence,
    Parked,
    Ran,
    Returned,
    ran_here,
    walks_agree,
)

from effective.api import await_event, step
from effective.budget import Grant, depth_grant_name
from effective.domain import CallTool
from effective.keys import Key
from effective.layers import layer_run_state
from effective.ops import CHAIN_DEPTH, CHAIN_GENERATION

pytestmark = pytest.mark.adversarial

GRANT = depth_grant_name("r", depth=1, generation=0)
_CELL = Key.parse("probe:ambient")


def _ambient() -> dict[str, Any]:
    """Every substrate ambient a workflow can see, as a value.

    Three of the four. `layers._OP_NAME` is absent because a workflow CANNOT observe it — the
    walk publishes it around the handler's interpretation of an op while the workflow is
    suspended at the `yield`, so it reads `None` from here on every walk and would assert
    nothing. It is not uncovered: the name it mints IS the op's key, so it is compared in the
    `keys` dimension instead."""
    cell = layer_run_state(_CELL)
    cell["reads"] = cell.get("reads", 0) + 1
    return {
        "chain_depth": CHAIN_DEPTH.get(),
        "chain_generation": CHAIN_GENERATION.get(),
        "run_scope_accumulates": cell["reads"],
    }


def _probe_the_ambient():
    """Read the ambient, do an op, read it again. The second read is the load-bearing one: a run
    scope that is not established hands back a FRESH dict every call, so `reads` stays at 1."""
    before = _ambient()
    yield from step("a", CallTool(name="op", result_schema=str))
    return [before, _ambient()]


def test_every_walk_establishes_the_same_ambient():
    """The assertion that would have caught `ReplayHandler` establishing no run scope, by
    construction rather than by a fork noticing.

    Before that fix this reported `run_scope_accumulates` of 2 on the recording walk and 1 on
    replay — a workflow returning a different value on replay than it recorded, which is the one
    thing replay exists to prevent."""
    agreed = walks_agree(_probe_the_ambient, steps=("a",))

    assert agreed.outcome == Returned(
        [
            {"chain_depth": 0, "chain_generation": None, "run_scope_accumulates": 1},
            {"chain_depth": 0, "chain_generation": None, "run_scope_accumulates": 2},
        ]
    ), agreed.outcome
    # Nothing DECLINED: this program completes, so even replay participates. `ran_here` is what
    # makes that claim true off a live container — the absurd walk needs Postgres and the harness
    # drops it by name, which is the one absence this assertion tolerates.
    assert agreed.ran == ran_here(*ALL_WALK_NAMES), agreed.skipped


def test_every_walk_that_can_park_binds_the_same_authority_name():
    """The arc's own property. An authority name exists to be parked on, so the park IS the
    identity — and a walk that composed a different one would bind a different question.

    Replay declines rather than being skipped silently: a trace ending at a park has no op to
    replay past, which is structural and is stated in `Agreement.skipped` rather than left for a
    reader to infer from a count."""

    def parks():
        yield from step("a", CallTool(name="op", result_schema=str))
        return (yield from await_event(GRANT, Grant))

    agreed = walks_agree(parks, steps=("a",))

    assert agreed.outcome == Parked(GRANT), agreed.outcome
    assert agreed.ran == ran_here("recording", "sqlite", "absurd", "live_drive", "measured_drive")
    # `ran` and `skipped` partition the walks, so the tuple above already says which walks did
    # not run. What it cannot say is WHY, and replay's reason is the load-bearing one — look it
    # up by name rather than by position, which stopped being replay's the moment a second walk
    # could be skipped beside it.
    assert "no op to replay" in dict(agreed.skipped)["replay"]


def test_the_registered_walks_cover_every_site_that_mints_an_identity():
    """`placing(...)` is the mint every walk passes through, so a call site with no registered
    walk is a walk nobody is watching, and a hand-count of the sites goes wrong exactly that
    way.

    A test rather than a lint for now; `--walk-coverage` is the gate this becomes, and until it
    exists this is the claim. It pins the MAPPING, not a number: `fork.py` holds two sites for
    one walk because its replay recurses into scoped bodies, and asserting `len(sites) == 6`
    would be the same hand-count in a different costume."""
    import re
    from pathlib import Path

    root = Path(__file__).parent.parent / "src" / "effective"
    sites = {
        f"{path.relative_to(root)}"
        for path in root.rglob("*.py")
        for line in [path.read_text()]
        if re.search(r"with placing\(", line)
    }
    # Every module that mints is accounted for by at least one registered walk.
    covered = {
        "handlers/recording.py": "recording",
        "handlers/replay.py": "replay",
        "handlers/absurd.py": "sqlite",  # and "absurd" — one handler, two engines
        "fork.py": "live_drive",  # and "measured_drive", and the two replay-prefix sites
    }
    assert sites == set(covered), (
        f"a module mints identities with no registered walk: {sites - set(covered)}; "
        f"or a registered walk's module vanished: {set(covered) - sites}"
    )
    assert set(covered.values()) <= {w.name for w in ALL_WALKS}


def test_the_harness_reports_a_walk_that_disagrees():
    """An instrument that cannot go red is not evidence, and that holds for the thing doing the
    measuring.

    The failure output is a MATRIX rather than a diff of two values, because which walks
    CLUSTERED is what names the mechanism: an engine pair against the in-process pair reads
    differently from one lone driver."""

    class _LiarWalk:
        name = "liar"

        def run(self, program, answers, steps) -> Ran:
            return Ran(self.name, Returned("something else"))

    with pytest.raises(Divergence) as diverged:
        walks_agree(_probe_the_ambient, steps=("a",), walks=(*ALL_WALKS, _LiarWalk()))

    report = str(diverged.value)
    assert "walk-invariance failed on `outcome`" in report
    assert "liar" in report
    assert "<-- diverged" in report
    # The walks that agreed are shown too, so the reader sees the cluster rather than one pair.
    assert "recording" in report
    assert "sqlite" in report
