"""Skill pins on the durable path, under both death modes.

The contract: a pin is a *recorded* value. Activation replays from its
checkpoint, never an ambient re-read of a moved-on registry; a refresh re-pins
explicitly under its own key (``skill:{name},refresh,{n}``, distinct from
the activation); a parked run resumes on its recorded pins, and the refresh
after resume sees the *current* tree, all of it recorded. Proven crash-at-every-
op (death mode 1: exception into ``work_batch``) and across worker death +
fresh-worker resume (death mode 2).

Substrate-pure: no domain imports; the skill pack is an in-memory registry.
Gated on the Podman test Postgres (``just pgt-up``); skips otherwise.
"""

from string.templatelib import Template
from typing import Any
from uuid import uuid4

import pytest
from _durable import (
    DSN,
    Fault,
    FaultCtx,
    absurd,
    ledger_kinds,
    pg_ready,
)
from pydantic import BaseModel

from effective.api import append_ledger, ask_llm, await_event
from effective.channels import Field, render, skill
from effective.domain import AskLLM, CallTool, DomainOp
from effective.handlers.durable import DurableHandler
from effective.keys import Key
from effective.ledger import PostgresLedger
from effective.ops import LedgerRow
from effective.skills import DISCLOSE_TOOL, SkillRegistry, activate_skill, refresh_skill

pytestmark = pytest.mark.skipif(not pg_ready(), reason="no Podman test Postgres (just pgt-up)")


class Note(BaseModel):
    note: str


class ResumeEvent(BaseModel):
    actor: str


def _pack(version: str) -> SkillRegistry:
    return SkillRegistry.in_memory(
        {"advice": ("versioned test advice", Template(f"USE {version}"))}
    )


V1_HASH = _pack("V1").disclose("advice").content_hash
V2_HASH = _pack("V2").disclose("advice").content_hash
V3_HASH = _pack("V3").disclose("advice").content_hash


def _use(pin: Any) -> Template:
    note = Field(str)
    return t'apply the skill: {skill("advice", pin=pin)} and respond with "note": {note}'


def skilled(run_id: str, *, park: bool):
    """activate -> pinned use -> [park] -> refresh -> pinned use -> ledger."""
    pin1 = yield from activate_skill("advice")
    first = yield from ask_llm("use:1", _use(pin1), Note)
    if park:
        yield from await_event(f"resume:{run_id}", ResumeEvent)
    pin2 = yield from refresh_skill("advice", 1)
    second = yield from ask_llm("use:2", _use(pin2), Note)
    yield from append_ledger(
        LedgerRow(
            event_id=Key.parse(f"{run_id}:skill-used"),
            kind="skill_used",
            hashes=[pin1.content_hash, pin2.content_hash],
        )
    )
    return {"h1": pin1.content_hash, "h2": pin2.content_hash, "notes": [first.note, second.note]}


class SkillDomain:
    """Discloses from a swappable in-memory registry. The registry "moves on" to
    ``bump_to`` right after the activation's disclose (so a refresh sees new
    content — the tree changing mid-run). ``AskLLM`` renders the pinned template
    REGISTRY-FREE and echoes back the version marker the render actually saw —
    so the assertions test what the model would have read, not bookkeeping."""

    def __init__(self, version: str = "V1", bump_to: str | None = "V2") -> None:
        self.registry = _pack(version)
        self.bump_to = bump_to
        self.calls: list[str] = []

    def run(self, op: DomainOp[Any]) -> Any:
        if isinstance(op, CallTool) and op.name == DISCLOSE_TOOL:
            event = op.args["event"]
            self.calls.append(f"disclose:{event}")
            pin = self.registry.disclose(op.args["skill"])
            if self.bump_to is not None and event == "activate":
                self.registry = _pack(self.bump_to)
                self.bump_to = None
            return pin
        if isinstance(op, AskLLM):
            self.calls.append("ask")
            prompt = render(op.messages, output=op.response_schema)  # pinned: no registry
            text = " ".join(m.content for m in prompt.messages)
            marker = next(v for v in ("V1", "V2", "V3") if v in text)
            return prompt.resolve({"note": marker})
        raise AssertionError(f"unexpected op {op!r}")


def _run_one(app: Any, rid: str, crash_at: int | None) -> tuple[Any, SkillDomain, Fault]:
    """Spawn one no-park skilled run, crashing before op ``crash_at`` (None = clean)."""
    domain = SkillDomain()  # V1; the tree moves to V2 after activation
    fault = Fault(crash_at)
    name = f"skillcrash-{crash_at}-{rid}"

    @app.register_task(name, default_max_attempts=3)
    def task(params, ctx, _domain=domain, _fault=fault):
        ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
        try:
            return DurableHandler(FaultCtx(ctx, _fault), _domain, ledger=ledger).run(
                lambda: skilled(params["run_id"], park=False)
            )
        finally:
            ledger.close()

    spawned = app.spawn(name, {"run_id": rid})
    snap = app.run_until_result(spawned)
    return snap, domain, fault


REF = {"h1": V1_HASH, "h2": V2_HASH, "notes": ["V1", "V2"]}
ONCE = ["disclose:activate", "ask", "disclose:refresh", "ask"]


def test_crash_at_every_op_replays_pins_to_identical_outcome():
    app = absurd()
    try:
        # Clean baseline: reference outcome + the op count N.
        base = f"skillbase-{uuid4().hex[:8]}"
        snap, domain, fault = _run_one(app, base, crash_at=None)
        assert snap is not None
        assert snap.state == "completed", f"{snap.state}: {snap.failure}"
        assert snap.result == REF  # activation pinned V1; refresh re-pinned V2
        assert domain.calls == ONCE
        n_ops = fault.count
        # activate, use:1, refresh:1, use:2, ledger — five durable boundaries
        assert n_ops == 5, n_ops
        assert ledger_kinds(base) == ["skill_used"]

        for k in range(1, n_ops + 1):
            rid = f"skillcrash{k}-{uuid4().hex[:8]}"
            snap, domain, fault = _run_one(app, rid, crash_at=k)
            assert fault.armed is False, f"k={k}: fault never fired"
            assert snap is not None, f"k={k}: no result"
            assert snap.state == "completed", f"k={k}: {snap.state} {snap.failure}"
            # Pins are recorded values: identical hashes and identical *rendered
            # content* at every crash boundary — never a re-read of the moved tree.
            assert snap.result == REF, f"k={k}: {snap.result}"
            # Exactly-once discloses across crash+replay (a doubled disclose here
            # means a checkpoint was wrongly re-run; a stale V1 note at use:2 would
            # mean the refresh collided with the activation's key):
            assert domain.calls == ONCE, f"k={k}: {domain.calls}"
            assert ledger_kinds(rid) == ["skill_used"], f"k={k}"
    finally:
        app.close()


def test_a_duplicate_activation_is_a_fresh_pin_event_on_the_durable_path():
    """The contract as documented: the SDK occurrence-counts duplicate step
    names (``name#2``), so a second ``activate_skill(x)`` performs a FRESH
    disclose, an unnamed refresh rather than a replay of the first pin.
    Deterministic on replay, but a new pin event; activate once and thread the
    pin. (The in-memory RecordingHandler cans by name and diverges here; this
    test pins the production semantics.)"""
    app = absurd()
    try:
        rid = f"skilldup-{uuid4().hex[:8]}"
        name = f"t-{rid}"
        domain = SkillDomain(version="V1", bump_to="V2")  # tree moves after 1st disclose

        def wf():
            p1 = yield from activate_skill("advice")
            p2 = yield from activate_skill("advice")  # duplicate call, same name
            return {"h1": p1.content_hash, "h2": p2.content_hash}

        @app.register_task(name, default_max_attempts=3)
        def task(params, ctx, _domain=domain):
            return DurableHandler(ctx, _domain).run(wf)

        spawned = app.spawn(name, {"run_id": rid})
        snap = app.run_until_result(spawned)
        assert snap is not None
        assert snap.state == "completed", f"{snap.state}: {snap.failure}"
        # the duplicate saw the moved tree — a fresh disclose, not the first pin:
        assert snap.result == {"h1": V1_HASH, "h2": V2_HASH}
        assert domain.calls == ["disclose:activate", "disclose:activate"]
    finally:
        app.close()


def test_parked_run_resumes_on_recorded_pins_after_worker_death():
    """Death mode 2, the motivating story: a run parks at a human gate; the
    worker dies; the tree moves on. A FRESH worker resumes: the pre-park pins
    replay from checkpoints (the resumed render still sees V1, with no ambient
    freshness even though the new worker's registry holds V3), and the explicit
    refresh after resume pins the *current* tree (V3), all recorded."""
    rid = f"skillpark-{uuid4().hex[:8]}"
    name = f"t-{rid}"

    def register(app: Any, domain: SkillDomain) -> None:
        @app.register_task(name, default_max_attempts=3)
        def task(params, ctx, _domain=domain):
            ledger = PostgresLedger(DSN, workflow_run_id=params["run_id"])
            try:
                return DurableHandler(ctx, _domain, ledger=ledger).run(
                    lambda: skilled(params["run_id"], park=True)
                )
            finally:
                ledger.close()

    app1 = absurd()
    app2 = None
    try:
        domain1 = SkillDomain(version="V1", bump_to=None)
        register(app1, domain1)
        spawned = app1.spawn(name, {"run_id": rid})
        app1.work_batch()  # activate (pins V1), use:1, park at the await

        snap = app1.fetch_task_result(spawned)
        assert snap is not None
        assert snap.state != "completed"  # parked
        assert domain1.calls == ["disclose:activate", "ask"]

        # The worker dies; the skill tree moves on to V3 before anyone resumes.
        app2 = absurd()
        domain2 = SkillDomain(version="V3", bump_to=None)
        register(app2, domain2)
        app2.emit_event(f"resume:{rid}", {"actor": "approver"})
        snap = app2.run_until_result(spawned)

        assert snap is not None
        assert snap.state == "completed", f"{snap.state}: {snap.failure}"
        # Pre-park content replayed (V1, though this worker's tree is V3);
        # the post-resume refresh explicitly re-pinned the current tree (V3):
        assert snap.result == {"h1": V1_HASH, "h2": V3_HASH, "notes": ["V1", "V3"]}
        # The fresh worker re-ran NOTHING before the park — activation and use:1
        # were checkpoint lookups; only the refresh and second use executed:
        assert domain2.calls == ["disclose:refresh", "ask"]
        assert ledger_kinds(rid) == ["skill_used"]
    finally:
        app1.close()
        if app2 is not None:
            app2.close()
