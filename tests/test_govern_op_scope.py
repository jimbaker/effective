"""A gate settles ONE op per approval, while an accumulator's grant outlives the op.

A park name of `govern:{gate}:{run_id}:{pass_n}` with `GateState` built fresh per op would give
every gated op in a run the SAME name, so one delivered approval would satisfy all of them, on
SQLite and on Absurd/Postgres alike. The park name is op-keyed, and it serves two policy kinds
that want opposite scopes:

- **`permission` is a SETTLEMENT**: an approval decides one op. Its park name must be op-scoped,
  or an approval leaks forward.
- **`budget` is an ACCUMULATOR**: a grant raises the RUN's ceiling, so it must still be in force
  at the next op and must not be re-requested.

Under a run-wide name, budget's accrual would come *from the collision*: op2 re-absorbing op1's
delivered event. Separating the two answer scopes on `GateState` (`answers_for` per op,
`run_answers_for` per run) lets the park name be op-scoped while accrual rides state.
"""

from collections.abc import Iterator, Mapping
from typing import Any

from _gate import at_spend

from effective import permission
from effective.api import call_tool
from effective.budget import MeasuredBudget
from effective.budget import as_policy as budget_policy
from effective.combinators import Again, Chain, Done, Turn, respawn
from effective.cost import Usage
from effective.govern import GateState, Proceed, govern
from effective.handlers.base import step_key
from effective.handlers.durable import DurableHandler
from effective.handlers.recording import RecordingHandler
from effective.ops import Step

GATE, RUN = "spend", "r-1"


class _Dom:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def run(self, op):
        self.calls.append(getattr(op, "name", type(op).__name__))
        return {"ok": True}

    def run_metered(self, op):
        return self.run(op), Usage()


def _two_gated_calls():
    a = yield from call_tool("charge_card", {}, dict)
    b = yield from call_tool("wire_transfer", {}, dict)
    return [a, b]


def _park_name(op_name: str, pass_n: int = 0) -> str:
    """The name the gate mints for a gated `call_tool` — derived, not hard-coded, so this test
    cannot drift from `GateState.park_name`."""
    # `.stored()` — every consumer here wants the WIRE name: `emit_event` addresses by it, and
    # the substring check below is a claim about the composed text.
    return GateState(
        run_id=RUN, gate=GATE, pass_n=pass_n, op_key=step_key(f"tool:{op_name}").stored()
    ).park_name.stored()


def _run(tmp_path, sqlite_app, policy, *, max_attempts: int = 12):
    app = sqlite_app(str(tmp_path / "gate.db"))
    dom = _Dom()

    @app.register_task("t")
    def task(params, ctx):
        gate = govern(policy, gate=GATE, run_id=RUN)
        return DurableHandler(ctx, dom, ledger=None, op_layers=(gate,)).run(_two_gated_calls)

    return app, dom, app.spawn("t", {}, max_attempts=max_attempts)


def _gated(op) -> bool:
    return isinstance(op, Step) and getattr(op.op, "name", None) in (
        "charge_card",
        "wire_transfer",
    )


def test_the_park_name_is_op_scoped():
    """Two ops under one gate mint two names. The property, stated directly."""
    assert _park_name("charge_card") != _park_name("wire_transfer")
    assert "tool:charge_card" in _park_name("charge_card")


def test_one_approval_settles_exactly_one_gated_op(tmp_path, sqlite_app):
    """A positive pin: approving op1 must NOT authorize op2."""

    def escalate_everything(op, state):
        if not _gated(op):
            return Proceed()
        return permission.as_policy([lambda _op: permission.Escalate()])(op, state)

    app, dom, tid = _run(tmp_path, sqlite_app, escalate_everything)
    app.run_until_result(tid)
    assert dom.calls == [], "nothing should run before an approval"

    approve = {"answers": {"permission": {"decision": "approve"}}}
    app.emit_event(_park_name("charge_card"), approve)
    app.run_until_result(tid)
    assert dom.calls == ["charge_card"], (
        f"one approval authorized {len(dom.calls)} gated ops: {dom.calls}"
    )

    # the second op is parked on its OWN name and needs its own decision
    app.emit_event(_park_name("wire_transfer"), approve)
    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "completed"
    assert dom.calls == ["charge_card", "wire_transfer"]


def test_a_budget_grant_outlives_the_op_it_was_granted_for():
    """The accumulator half: a grant delivered while gating op1 is still in force at op2, so the
    gate does not re-ask — and it now comes from run-scoped STATE rather than from two ops sharing
    a park name.

    Driven at the policy/state level deliberately. A faithful durable version needs a
    **replay-derived, positional** meter — the real handler re-derives spend as checkpoints replay,
    so at op1's re-gate it reads what was spent *before op1*, not the run total. A test closure
    that
    accumulates is not that (it made op1 re-park on replay when I first wrote this), and faking it
    would test the fake. The durable metered path is covered against the real engine in
    `test_govern_durable.py`."""
    from effective.domain import CallTool
    from effective.govern import Park, Resolution

    budget = MeasuredBudget(overall=0.005, run_id=RUN, on_exhaust="park")
    over = 0.006  # already past the ceiling
    policy = at_spend(over, budget_policy(budget))
    call = CallTool(name="wire_transfer", args={}, result_schema=dict)
    op = Step(name="tool:wire_transfer", op=call)

    # op1's gate: over the ceiling, no grant yet -> park (the ask)
    first = GateState(run_id=RUN, gate=GATE, op_key=step_key("tool:charge_card").stored())
    assert isinstance(policy(op, first), Park)

    # a grant is delivered and absorbed
    granted = first.absorb(Resolution(answers={"budget": {"add_dollars": 1.0}}))

    # op2's gate: a FRESH settlement view, but the grant is inherited through the run scope,
    # so the raised ceiling still holds and the gate does NOT ask again
    second = GateState(
        run_id=RUN,
        gate=GATE,
        op_key=step_key("tool:wire_transfer").stored(),
        run_answers=dict(granted.run_answers),
    )
    assert isinstance(policy(op, second), Proceed), "the grant did not carry to the next op"

    # and WITHOUT the run scope op2 would ask again: the run scope, not a shared park name,
    # carries the grant
    unscoped = GateState(run_id=RUN, gate=GATE, op_key=step_key("tool:wire_transfer").stored())
    assert isinstance(policy(op, unscoped), Park)


def test_the_two_answer_scopes_are_distinct():
    """`answers_for` is per-op (settlement); `run_answers_for` is per-run (accumulation). Absorbing
    folds into both; carrying to a NEW op keeps only the run scope."""
    from effective.govern import Resolution, answers_for, run_answers_for

    state = GateState(run_id=RUN, gate=GATE, op_key=step_key("tool:a").stored())
    state = state.absorb(Resolution(answers={"budget": {"add_dollars": 1.0}}))
    assert len(answers_for(state, "budget")) == 1
    assert len(run_answers_for(state, "budget")) == 1

    # the next op starts with an empty settlement view and the inherited accumulator view
    next_op = GateState(
        run_id=RUN,
        gate=GATE,
        op_key=step_key("tool:b").stored(),
        run_answers=dict(state.run_answers),
    )
    assert answers_for(next_op, "budget") == ()
    assert len(run_answers_for(next_op, "budget")) == 1


# --- `op_key` is not OCCURRENCE-injective -------------------------------------------------
#
# The op-scoped park name separates only DISTINCT op keys. A `Step` carries the author's bare
# name, so an agent calling one tool twice yields two ops with the SAME key, and without an
# occurrence the second parks on a name whose event was already delivered, absorbing the first's
# approval. The engines suffix duplicate CHECKPOINTS `name#2` below the ctx; AUTHORITY names need
# the same. On both engines the hazard is a $5 approval authorizing a $5,000,000 charge of the
# same tool, the failure class budgeting exists to prevent, so it is pinned with those amounts.


def _repeated_charges():
    a = yield from call_tool("charge_card", {"amount": 5}, dict)
    b = yield from call_tool("charge_card", {"amount": 5_000_000}, dict)
    return [a, b]


def _one_charge():
    return (yield from call_tool("charge_card", {"amount": 5}, dict))


def _charge_once_per_turn(state, turn: Turn):
    """A gated op inside a chain generation, written EXACTLY as it is outside one — no generation
    argument at the call site, which is the property the respawn test asserts."""
    yield from _one_charge()
    return Done("finished") if turn.final else Again(state)


class _Amounts(_Dom):
    def __init__(self) -> None:
        super().__init__()
        self.amounts: list[int] = []

    def run(self, op):
        self.amounts.append(op.args.get("amount"))
        return super().run(op)


def _occ_name(occ: int, op_name: str = "charge_card") -> str:
    # `.stored()` — like `_park_name` above, every consumer is `emit_event`, which addresses
    # by the wire name.
    return GateState(
        run_id=RUN, gate=GATE, op_key=step_key(f"tool:{op_name}").stored(), occurrence=occ
    ).park_name.stored()


def test_each_occurrence_of_one_tool_needs_its_own_approval(tmp_path, sqlite_app):
    """The $5 / $5,000,000 case. One approval must not authorize a second charge of the SAME
    tool."""

    def escalate_everything(op, state):
        if not (isinstance(op, Step) and getattr(op.op, "name", None) == "charge_card"):
            return Proceed()
        return permission.as_policy([lambda _op: permission.Escalate()])(op, state)

    app = sqlite_app(str(tmp_path / "occ.db"))
    dom = _Amounts()

    @app.register_task("t")
    def task(params, ctx):
        gate = govern(escalate_everything, gate=GATE, run_id=RUN)
        return DurableHandler(ctx, dom, ledger=None, op_layers=(gate,)).run(_repeated_charges)

    tid = app.spawn("t", {}, max_attempts=16)
    app.run_until_result(tid)
    approve = {"answers": {"permission": {"decision": "approve"}}}

    app.emit_event(_occ_name(0), approve)
    app.run_until_result(tid)
    assert dom.amounts == [5], (
        f"one approval authorized {dom.amounts}: a repeated op_key shared one approval"
    )

    app.emit_event(_occ_name(1), approve)  # the second occurrence, its OWN decision
    snap = app.run_until_result(tid)
    assert snap is not None
    assert snap.state == "completed"
    assert dom.amounts == [5, 5_000_000]


def test_the_park_name_is_occurrence_scoped_and_injective():
    """Pinned as the CLASS: the name is injective in all five components, occurrence included."""
    assert _occ_name(0) != _occ_name(1)
    combos = [
        (g, r, n, occ, k)
        for g in ("spend", "approve")
        for r in ("a", "b")
        for n in range(2)
        for occ in range(3)
        for k in (step_key("tool:charge_card").stored(), step_key("tool:wire_transfer").stored())
    ]
    names = {
        GateState(run_id=r, gate=g, pass_n=n, occurrence=occ, op_key=k).park_name
        for g, r, n, occ, k in combos
    }
    assert len(names) == len(combos) == 48


class _AskedNames(Mapping[str, Any]):
    """Approve every gate park and RECORD the name it was asked on.

    A computing Mapping rather than a dict, which is the seam `RecordingHandler.responses`
    documents: the whole question is which name the gate composes, so a table that had to spell
    one in advance could not ask it.

    `__contains__` is spelled out because `Mapping`'s default calls `__getitem__`, and `_canned`
    probes membership before fetching — a recording `__getitem__` alone counts every ask twice.
    An instrument must not invent the thing it measures.
    """

    def __init__(self) -> None:
        self.asked: list[str] = []

    def _known(self, key: str) -> bool:
        return key.startswith("govern:") or "charge_card" in key

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and self._known(key)

    def __getitem__(self, key: str) -> Any:
        if key.startswith("govern:"):
            self.asked.append(key)
            return {"answers": {"permission": {"decision": "approve"}}}
        if "charge_card" in key:
            return {"ok": True}
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def _escalate_charges(op, state):
    if not (isinstance(op, Step) and getattr(op.op, "name", None) == "charge_card"):
        return Proceed()
    return permission.as_policy([lambda _op: permission.Escalate()])(op, state)


def test_the_gate_state_is_not_carried_in_a_factory_closure():
    """ONE gate object, two runs: run B must not inherit run A's occurrence count.

    **Asserted on the NAME run B composes, not on whether B proceeded, and that is what makes it
    engine-independent.** The counter lives in the handler's per-`run()` scope, so a closure leak
    surfaces as run B composing `occurrence=1` where run A composed `occurrence=0`. Nothing is
    emitted and no engine is involved, which is the point.

    Two SQLite tasks sharing one `RUN` would be the wrong instrument: their isolation comes from
    the ADDRESSED `events(task_id, name)` table rather than from the name, a property the deployed
    broadcast engine does not have (on real Absurd, B proceeds on A's approval).

    The equality is the assertion, so the sibling below supplies the positive control: a counter
    that never incremented at all would satisfy this one alone.
    """
    shared_gate = govern(_escalate_charges, gate=GATE, run_id=RUN)  # ONE gate, two runs
    names = _AskedNames()
    for _ in range(2):
        RecordingHandler(responses=names, op_layers=(shared_gate,)).run(_one_charge)

    assert names.asked == [_occ_name(0), _occ_name(0)], (
        f"run B did not restart the occurrence counter: {names.asked} — the gate state leaked "
        "through the factory closure"
    )


def test_a_real_respawn_chain_separates_each_generation_s_gate_park():
    """The cross-TASK half, through the machinery that actually threads the coordinate.

    Every other coordinate on `GateState` restarts at a generation boundary — `occurrence` counts
    in the handler's per-`run()` scope, `pass_n` starts a fresh gate, and the checkpoint ordinal
    is per task. A `respawn` chain keeps `run_id` stable by construction, so without this
    coordinate both generations compose `govern:spend,chain-r;step;tool:charge_card` and one
    approval settles both.

    **In-process on the recording walk, because what is asserted is the NAME each generation
    composes** — the same instrument `test_grant_aliasing`'s respawn case uses, and for the same
    reason: the engines' event semantics are pinned elsewhere, and a durable run would test them
    instead of this.

    Its sibling in `test_parked_reader.py` asserts the same property over EVERY settlement
    namespace from the minters alone. This one is the end-to-end confirmation that a real chain
    reaches those minters: a name that separates is worth nothing if `respawn` never sets
    the ambient it reads.
    """
    gate = govern(_escalate_charges, gate=GATE, run_id="chain-r")
    names = _AskedNames()
    for generation in (0, 1):
        chain = Chain(task="w", state=None, run_id="chain-r", generation=generation, params={})
        RecordingHandler(responses=names, op_layers=(gate,)).run(
            # `chain=chain` binds the loop variable at definition rather than at call. The lambda
            # is invoked immediately so late binding would read the same object either way — but a
            # closure over a loop variable is the shape that stops being harmless the moment
            # someone defers the call, and `run` taking a thunk is an invitation to.
            lambda chain=chain: respawn(_charge_once_per_turn, chain, budget=None)
        )

    assert len(set(names.asked)) == len(names.asked) == 2, names.asked
    assert names.asked[0] == "govern:spend,chain-r;step;tool:charge_card", names.asked
    assert names.asked[1] == "govern:spend,chain-r,generation=1;step;tool:charge_card", names.asked


def test_the_occurrence_counter_does_advance_within_one_run():
    """The positive control for the sibling above: the counter is per-run, and it is a counter.

    Two charges in ONE run compose two DISTINCT names. Without this, "both runs asked the same
    name" would be satisfied by a gate that never counted."""
    gate = govern(_escalate_charges, gate=GATE, run_id=RUN)
    names = _AskedNames()
    RecordingHandler(responses=names, op_layers=(gate,)).run(_repeated_charges)

    assert names.asked == [_occ_name(0), _occ_name(1)], names.asked
