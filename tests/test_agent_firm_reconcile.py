"""`reconcile` over the AR cases: a clean payment matches, and each short payment is explained."""

from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from _conformance import Fault

from effective.api import step
from effective.domain import CallTool
from examples.agent_firm.ar import INVARIANTS
from examples.agent_firm.cases import CASES, Case
from examples.agent_firm.reconcile import reconcile
from examples.agent_firm.world import World

GAPS = {case.payment.id: str(case.gap) for case in CASES.values()}


def _explain(case: Case, broken: tuple[str, ...]):
    args = {"payment": case.payment.id, "broken": list(broken)}
    return (yield from step("explain", CallTool(name="explain", args=args, result_schema=str)))


def _reconciled(_run_id: str):
    done = yield from reconcile(list(CASES.values()), INVARIANTS, _explain)
    return {
        "matched": [case.name for case in done.matched],
        "exceptions": {u.item.name: [list(u.broken), u.explanation] for u in done.exceptions},
    }


def test_the_clean_payment_matches_and_each_short_payment_is_explained(backend):
    asked: list[Mapping[str, Any]] = []

    def explain(args: Mapping[str, Any]) -> str:
        asked.append(args)
        return f"short by {GAPS[args['payment']]}"

    world = World(tools={"explain": explain})
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(f"rec-{run_id}", _reconciled, world.domain(), Fault(None), ())
    snapshot = backend.run_until_result(backend.spawn(f"rec-{run_id}", run_id))
    assert snapshot.result == {
        "matched": ["clean"],
        "exceptions": {
            "sla-cap": [["paid-in-full"], "short by 3000.00"],
            "discount": [["paid-in-full"], "short by 400.00"],
        },
    }
    assert sorted(a["payment"] for a in asked) == sorted(
        CASES[name].payment.id for name in ("sla-cap", "discount")
    )


def test_every_invariant_holding_explains_nothing(backend):
    def matched(_run_id: str):
        done = yield from reconcile([CASES["clean"]], INVARIANTS, _explain)
        return [len(done.matched), len(done.exceptions)]

    world = World(tools={})
    run_id = f"r-{uuid4().hex[:8]}"
    backend.register(f"none-{run_id}", matched, world.domain(), Fault(None), ())
    assert backend.run_until_result(backend.spawn(f"none-{run_id}", run_id)).result == [1, 0]
    assert not world.calls
