"""The AR firm's resolutions driven from a manifest on either engine, approved by a controller,
for the suites that read what a resolution read, wrote and cost."""

from dataclasses import dataclass, field
from typing import Any, Protocol
from uuid import UUID, uuid4

from _conformance import Fault

from effective.cost import Contract, MeteredInterpreter, Usage
from effective.govern import GateState, Policy, Proceed, Verdict, govern, rulers_for
from effective.handlers.durable import DurableHandler
from effective.keys import Segment
from effective.parked import answer
from effective.viewing import ViewingCtx
from examples.agent_firm.ar import Round
from examples.agent_firm.cases import SLA_CAP, Case
from examples.agent_firm.critic import CLAIMED
from examples.agent_firm.engage import CritiqueWire
from examples.agent_firm.knowledge import K0, Pinned
from examples.agent_firm.manifest import Manifest
from examples.agent_firm.resolve import APPROVAL, admitted, announced, approval, ruled, terms_of
from examples.agent_firm.world import World

PROPOSER, CONTROLLER = Segment("ar-clerk"), Segment("controller")
GENESIS = Pinned(height=0, digest=K0.digest)
CAPPED = {"verdict": "revise", "credit": "1000", "cites": ["MSA-NW-2025"], "reason": "4.2 caps"}


@dataclass
class Briefed:
    """A scripted world, and the brief each of its models was asked with."""

    world: World
    proposals: list[str] = field(default_factory=list)
    critiques: list[str] = field(default_factory=list)


def briefed(critique: dict[str, Any] = CAPPED, **world: Any) -> Briefed:
    def proposer(op: Any) -> Any:
        scripted.proposals.append(str(op.messages))
        return CLAIMED

    def critic(op: Any) -> tuple[Any, Usage]:
        scripted.critiques.append(str(op.messages))
        return CritiqueWire.model_validate(critique), Usage()

    ok = lambda args: {"ok": True}  # noqa: E731
    scripted = Briefed(
        World(
            llm=proposer,
            critic=critic,
            critiques=frozenset({CritiqueWire}),
            tools={"notify": ok, "collect": ok},
            **world,
        )
    )
    return scripted


class WorldHolder(Protocol):
    world: World


@dataclass(frozen=True)
class Run:
    task_id: UUID
    run_id: str
    snapshot: Any
    params: dict[str, Any]
    asked_at_park: dict[str, int] | None
    """The model asks by schema when the run parked."""
    parks: int
    passed: list[tuple[Segment | None, ...]]
    """Who had ruled on the approval each time the policy let a settlement through."""

    @property
    def result(self) -> Any:
        return self.snapshot.result

    @property
    def failure(self) -> Any:
        return self.snapshot.failure


def watched(policy: Policy, passed: list[tuple[Segment | None, ...]]) -> Policy:
    """`policy`, recording each pass into `passed`."""

    def watching(op: Any, state: GateState) -> Verdict:
        match verdict := policy(op, state):
            case Proceed() if terms_of(op) is not None:
                passed.append(tuple(rulers_for(state, APPROVAL)))
        return verdict

    return watching


def manifest(
    pinned: Pinned, round: Round, *, critic: bool = True, case: Case = SLA_CAP, arm: str = "main"
) -> Manifest:
    models = {"proposer": "scripted", "critic": "scripted"}
    return Manifest(
        case=case,
        round=round,
        critic=critic,
        knowledge=pinned,
        arm=arm,
        code="tests",
        models=models,
    )


def gate(run_id: str, policy: Policy | None = None):
    return govern(
        policy or approval(PROPOSER),
        gate="credit",
        run_id=run_id,
        announce=announced,
        ruled=ruled,
    )


def resolved(
    backend,
    scripted: WorldHolder,
    admission: Manifest,
    *,
    hypothetical: bool = False,
    fault: Fault | None = None,
    approvers: tuple[Segment, ...] = (CONTROLLER,),
    decision: str = "approve",
    contract: Contract = Contract.V0,
) -> Run:
    """Resolve what `admission` admits under `contract`, answering each park with `decision` as
    each of `approvers` in turn."""
    run_id = f"r-{uuid4().hex[:8]}"
    passed: list[tuple[Segment | None, ...]] = []

    def program(params: dict[str, Any]):
        return (yield from admitted(Manifest.admitted(params))())

    backend.register(
        run_id,
        program,
        scripted.world.domain(),
        fault or Fault(None),
        (gate(run_id, watched(approval(PROPOSER), passed)),),
        admitted=True,
        hypothetical=hypothetical,
        budget_limit=None if admission.budget is None else float(admission.budget),
    )
    task_id = backend.spawn(run_id, run_id, contract=contract, params=admission.params())
    snapshot = backend.run_until_result(task_id)
    asked, parks, waiting = None, 0, list(approvers)
    while waiting and (parked := backend.parked(task_id)):
        asked = asked or dict(scripted.world.asked)
        (park,) = parked
        payload = {"answers": {APPROVAL: {"decision": decision}}}
        answer(backend.engine, park, payload, by=waiting.pop(0))
        parks += 1
        snapshot = backend.run_until_result(task_id)
    ((_, params),) = backend.enqueued(run_id)
    return Run(task_id, run_id, snapshot, params, asked, parks, passed)


def _unasked(op: Any) -> Any:
    raise AssertionError(f"a replay called the world: {op!r}")


def replayed(backend, run: Run) -> Any:
    raw = backend.unclaimed_ctx(run.task_id)
    try:
        ctx = raw
        if backend.name == "postgres":
            from effective.engines.absurd import ConcurrentAbsurdCtx

            ctx = ConcurrentAbsurdCtx(raw)
        viewer = ViewingCtx(ctx, backend.checkpoint_states(run.task_id))
        unasked = MeteredInterpreter(llm=_unasked, tools=_unasked, judge=_unasked)
        handler = DurableHandler(
            viewer,
            unasked,
            ledger=None,
            op_layers=(gate(run.run_id),),
            contract=Contract.from_params(run.params),
        )
        return handler.run(admitted(Manifest.admitted(run.params)))
    finally:
        if backend.name == "postgres":
            raw._conn.close()
