"""The spend a budget gate reads is the task's replay-derived meter, and a run that does not meter
gives it none to read."""

import threading
from collections.abc import Callable
from typing import Any
from uuid import uuid4

import pytest
from _schedules import Turnstile, step_name

from effective.api import Effect, ask_llm, call_tool, gather
from effective.budget import Grant, MeasuredBudget, OnExhaust
from effective.budget import as_policy as budget_policy
from effective.cost import Contract, MeteredInterpreter, Usage
from effective.domain import DomainOp
from effective.govern import GateState, Policy, Proceed, Resolution, govern
from effective.handlers.base import _walk_run, walk_run
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.keys import Run, compose_key
from effective.layers import current_meter
from effective.ops import CompositionRefused, Step, WorkflowOp

DOLLAR = 1.0


def metered() -> MeteredInterpreter:
    return MeteredInterpreter(
        llm=lambda _op: ("ans", Usage(prompt_tokens=1, completion_tokens=1, cost=DOLLAR)),
        tools=lambda _op: 0,
    )


class Unmetered:
    """A domain that answers an ask and reports no usage."""

    def run(self, op: DomainOp[Any]) -> Any:
        return "ans"


def asks(*names: str) -> Callable[[str], Effect[list[str]]]:
    def program(_run_id: str) -> Effect[list[str]]:
        answers = []
        for name in names:
            answers.append((yield from ask_llm(name, name, str)))
        return answers

    return program


def gated(
    backend,
    program: Callable[[str], Effect[Any]],
    policy_for: Callable[[str], Policy],
    *,
    domain: Callable[[], Any] = metered,
    contract: Contract = Contract.V1,
    max_attempts: int = 3,
    outer_layers: tuple[Any, ...] = (),
) -> tuple[Any, Any]:
    """`program` under a `govern` gate holding `policy_for(run_id)`, run to where it stops.

    The gather is concurrent on both engines and there is no arm where it is not: an
    `AbsurdBackend` body is always handed a `ConcurrentAbsurdCtx` and SQLite's own ctx is
    concurrent-safe. A schedule is named with `_schedules.Turnstile` rather than selected here,
    and branch-index order IS the sequential one.

    `outer_layers` sit ABOVE the gate, since `drive_through` makes the first layer the outermost:
    one that blocks there decides when the gate below it sees its op."""
    name = compose_key(t"gate-meter:{Run(str(uuid4()))}").stored()

    def body(params: Any, ctx: Any) -> Any:
        run_id = params["run_id"]
        handler = DurableHandler(
            ctx,
            domain(),
            op_layers=(*outer_layers, govern(policy_for(run_id), gate="spend", run_id=run_id)),
            contract=contract,
        )
        return handler.run(lambda: program(run_id))

    backend.register_body(name, body)
    task = backend.spawn(name, str(uuid4()), max_attempts=max_attempts, contract=contract)
    return task, backend.run_until_result(task)


def ceiling(dollars: float, on_exhaust: OnExhaust = "fail") -> Callable[[str], Policy]:
    return lambda run_id: budget_policy(
        MeasuredBudget(overall=dollars, run_id=run_id, on_exhaust=on_exhaust)
    )


def granted(dollars: float) -> dict:
    return Resolution(answers={"budget": Grant(add_dollars=dollars).model_dump()}).model_dump()


def test_a_metered_run_refuses_the_ask_after_the_spend_crosses_the_ceiling(backend):
    task, snap = gated(backend, asks("first", "second"), ceiling(DOLLAR / 2))
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "BudgetRefused", snap.failure
    assert backend.checkpoint_keys(task) == ["step:first"], "the second ask was refused"


@pytest.mark.parametrize("grants", [[1.0], [0.25, 1.0]], ids=["enough", "short then enough"])
def test_a_grant_resumes_on_the_spend_the_replay_derives(backend, grants):
    """$1 asks under $1.50: the third parks at $2 spent. A resume replays the $2, and a grant
    short of it parks again under a fresh name."""
    task, snap = gated(backend, asks("a", "b", "c"), ceiling(1.5, "park"), max_attempts=5)
    names = []
    for dollars in grants:
        assert snap.state not in ("completed", "failed"), snap
        assert backend.checkpoint_keys(task) == ["step:a", "step:b"]
        (parked,) = backend.parked(task)
        names.append(parked.wake_event)
        backend.emit_event(task, parked.wake_event, granted(dollars))
        snap = backend.run_until_result(task)
    assert snap.state == "completed", snap
    assert backend.checkpoint_keys(task) == ["step:a", "step:b", "step:c"]
    assert len(set(names)) == len(grants), "each park has its own name"


def leaf(tag: str) -> Callable[[], Effect[str]]:
    return lambda: ask_llm(f"leaf-{tag}", "p", str)


def branch(i: int) -> Callable[[], Effect[list[str]]]:
    return lambda: gather([leaf(f"{i}a"), leaf(f"{i}b")])


def nested(_run_id: str) -> Effect[str]:
    yield from ask_llm("before", "p", str)
    yield from gather([branch(0), branch(1)])
    yield from ask_llm("after", "p", str)
    return "done"


LEAVES = ("leaf-0a", "leaf-0b", "leaf-1a", "leaf-1b")
"""The nested gather's four leaves, in branch-index order."""


@pytest.mark.parametrize(
    "order",
    [None, LEAVES, LEAVES[::-1]],
    ids=["whichever order the run gives", "leaves by index", "leaves reversed"],
)
def test_a_gate_in_a_branch_reads_the_root_spend(backend, order):
    """Every leaf of a nested gather reads the $1 spent before the outer gather: no sibling's
    spend, no inner barrier's. The ask after the outer barrier reads all five.

    `None` takes whichever interleaving the run gives; the other two name a schedule and get it.
    A `Turnstile` above the gate commits the leaves in the order given, so each leaf's gate is
    asked only after every earlier leaf has accrued its dollar. That is the schedule that would
    expose a shared running meter: under `leaves reversed`, `leaf-1b` commits first and
    `leaf-0a`'s gate is the last to read, with three sibling dollars already spent. It reads $1
    every way, because a branch's subtotal reaches the root's meter at the barrier and not before
    (`Budget.lean`'s Model B, `absurd.py:1899-1900`).

    A named order is reachable because the harness gathers concurrently on both engines: an
    `AbsurdBackend` body is always handed a `ConcurrentAbsurdCtx` and SQLite's own ctx is
    concurrent-safe. A genuinely sequential gather cannot run branch 1 first, and the turnstile
    would wait out its timeout saying so."""
    seen: dict[str, float] = {}
    threads: set[int] = set()
    siblings_in: dict[str | None, int] = {}
    turnstile = None if order is None else Turnstile(order, step_name)

    def recording(op: WorkflowOp, state: GateState) -> Proceed:
        assert state.meter is not None
        threads.add(threading.get_ident())
        seen.setdefault(state.op_key.rsplit(";", 1)[-1], state.meter.cost)
        if turnstile is not None:
            # Keyed by the label the turnstile orders on, so the two cannot disagree about a leaf.
            siblings_in.setdefault(step_name(op), len(turnstile.ended))
        return Proceed()

    _task, snap = gated(
        backend,
        nested,
        lambda _run_id: recording,
        outer_layers=() if turnstile is None else (turnstile.layer(),),
    )

    assert snap.state == "completed", snap
    assert seen == {
        "step:before": 0.0,
        "step:leaf-0a": DOLLAR,
        "step:leaf-0b": DOLLAR,
        "step:leaf-1a": DOLLAR,
        "step:leaf-1b": DOLLAR,
        "step:after": 5 * DOLLAR,
    }
    assert len(threads) > 1, "the branches ran on their own threads"
    if turnstile is None:
        return
    assert turnstile.kept_its_schedule(), f"the run would not take {order}: {turnstile.ended}"
    assert len(turnstile.threads) > 1, "the leaves committed from one thread"
    # The schedule was adversarial, and this is the proof rather than the intent: the turnstile
    # releases the k-th leaf only once k have committed, and no other leaf can commit while it
    # holds the turn, so the k-th gate is asked with exactly k sibling dollars already spent.
    assert [siblings_in[leaf] for leaf in order] == [0, 1, 2, 3]


RUNS = {
    "V0 over a metered domain": (Contract.V0, metered),
    "V1 over a domain that reports no usage": (Contract.V1, Unmetered),
}
"""A run that does not meter: the contract it runs under, and its domain."""


def only_the_second(policy: Policy) -> Policy:
    """`policy` on the second ask; every other op proceeds. A gate cannot see the budget policy
    inside, so the handler cannot refuse it when it is built."""

    def scoped(op: WorkflowOp, state: GateState):
        match op:
            case Step(name="second"):
                return policy(op, state)
            case _:
                return Proceed()

    return scoped


PLACEMENTS = {
    "a gate holding the policy": (lambda p: p, []),
    "a policy wrapped for the second ask": (only_the_second, ["step:first"]),
}
"""How the policy reaches the gate, and the ops that run before the refusal."""


@pytest.mark.parametrize("placement", PLACEMENTS)
@pytest.mark.parametrize("run", RUNS)
def test_a_budget_gate_refuses_to_compose_over_a_run_that_does_not_meter(backend, run, placement):
    """Without the refusal both asks go through: the unmetered handler's spend stays zero."""
    contract, domain = RUNS[run]
    wrap, ran = PLACEMENTS[placement]
    task, snap = gated(
        backend,
        asks("first", "second"),
        lambda run_id: wrap(ceiling(DOLLAR / 2)(run_id)),
        domain=domain,
        contract=contract,
    )
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "CompositionRefused", snap.failure
    assert "does not meter" in str(snap.failure)
    assert backend.checkpoint_keys(task) == ran
    assert backend.task_attempts(task) == 1, "a composition refusal is not retried"


def nothing(_run_id: str) -> Effect[None]:
    yield from ()


@pytest.mark.parametrize("run", RUNS)
def test_the_handler_refuses_a_gate_holding_a_budget_policy_when_it_is_built(backend, run):
    """A program with no op for the gate to see still fails: the refusal is the handler's."""
    contract, domain = RUNS[run]
    _task, snap = gated(backend, nothing, ceiling(DOLLAR / 2), domain=domain, contract=contract)
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "CompositionRefused", snap.failure


def test_the_recorder_does_not_meter():
    budget = MeasuredBudget(overall=DOLLAR / 2, run_id="r", on_exhaust="fail")
    handler = RecordingHandler({}, op_layers=[govern(budget_policy(budget), gate="g", run_id="r")])
    with pytest.raises(CompositionRefused, match="does not meter"):
        handler.run(lambda: call_tool("t", {}, int))


def test_a_run_nested_in_a_metered_one_starts_unmetered_and_leaves_its_host_metered():
    with _walk_run(lambda: Usage(cost=DOLLAR)):
        assert current_meter() == Usage(cost=DOLLAR)
        with walk_run():
            assert current_meter() is None
        assert current_meter() == Usage(cost=DOLLAR), "the host reads its own spend again"


def test_the_public_walk_entry_publishes_no_spend():
    """A layer that wraps `yield op` in a walk entry cannot hand the gates below it a meter: `ty`
    refuses the call, and so does the run."""
    with pytest.raises(TypeError):
        # the call under test is the one `ty` refuses
        walk_run(lambda: Usage(cost=DOLLAR))  # ty: ignore[too-many-positional-arguments]


def test_a_handlers_domain_is_fixed_when_it_is_built(backend):
    """Whether a run meters is decided from its domain when the run starts, so the domain stays."""
    name = compose_key(t"gate-meter:{Run(str(uuid4()))}").stored()

    def body(params: Any, ctx: Any) -> str:
        handler = DurableHandler(ctx, metered(), contract=Contract.V1)
        try:
            # the assignment under test is the one `ty` refuses
            handler.domain = Unmetered()  # ty: ignore[invalid-assignment]
        except AttributeError:
            return "fixed"
        return "replaced"

    backend.register_body(name, body)
    snap = backend.run_until_result(backend.spawn(name, str(uuid4()), contract=Contract.V1))
    assert snap.result == "fixed", snap
