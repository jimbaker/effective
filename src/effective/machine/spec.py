"""What a state IS: the per-state declarations the trampoline reads, and nothing else.

`transition` says where a verdict goes; this module says what a state *does* to produce one. The
two vary independently: the edge table is a property of the design, and these fields are a
property of a deployment (which skill, which tier, which repertoire), so a tunable seam here takes
data.

**`canonical` is DECLARED.** Whether a state reaches the append-only ledger is the fact a reader
most needs and the hardest to discover, since the ledger write can sit two call frames away from
the transition that names it. A boolean means a reviewer reads one line, and
`test_only_the_commitment_points_reach_the_canonical_record` is driven by the declaration.

**The repertoire lives where it is consumed**, `specs.agent_worker(needs=...)`, which hands it to
a `decide_for` factory that masks the model's tool set at the decode seam. A second spelling here
would be a write-only field that drifts from the first.

**A skill binding is deliberately not expressible here.** It is not a substrate concept: skills
compose at the EMBODIMENT layer, where a convenience combinator can wrap a worker with an
activation.

**There is no `route` field.** `transition` is the one router, and a spec carrying its own would
be a second speller of the edge table — the defect this package exists to remove, one level up.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import StrEnum
from string.templatelib import Template
from typing import Any, cast

from effective.api import Effect
from effective.keys import Segment
from effective.machine.evidence import CommandRun, Measured


@dataclass(frozen=True, slots=True)
class Ctx[S: StrEnum, R: Measured = CommandRun]:
    """Where the machine is, handed to every state's `run`. Immutable per visit: a run
    that wants to remember something across visits records an op, so what it remembered survives
    a crash and re-derives on replay."""

    run_id: Segment
    """Typed, not `str`, because this reaches the canonical record's ADDRESS. `Segment` refuses a
    delimiter and anything that is not a well-formed atom, once, where the value is created, so a
    `run_id` that could not be composed into an `event_id` cannot get this far. A `str` validated
    inside the postamble would fail two committed ops past the point of no return."""

    goal: str | Template
    """What this run was asked to do. A `Template` where the ask was COMPOSED rather than written:
    a goal a parent built from what its child concluded carries model text, and a hole declaring
    `:data` is what keeps that text quoted rather than read as instruction."""

    state: S
    visit: int

    tree: Mapping[str, str] = field(default_factory=dict, hash=False)
    """The workspace this visit starts from: the caller's seed, advanced by every tree a state
    returned. A tool that edits the workspace takes it in its arguments, so a worker resumed on a
    fresh process sees the same files the crashed one did."""

    incoming: Report[Any, R] | None = field(default=None, hash=False)
    """What the PREVIOUS visit produced — the whole record, not the tree alone.

    Nothing carried this before. Across a state boundary only `Evidence.tree` survived, and it
    survived into the POSTAMBLE rather than into the next `Ctx`, so a state whose job is to read
    the last state's measurement had to re-run it — a second, differently-keyed op, and for a
    model judge a second non-deterministic call whose answer may differ from the one being judged.
    `tests/test_machine_fuse_or_split.py` is the standing measurement: RULE reads its
    predecessor's report instead of re-running the gate, pinned as `gate_calls == 5`.

    `None` on the first visit, and only there — which is a runtime fact the type cannot narrow,
    so a state entered only from another state says so with an assertion. That is the one
    guarantee this shape gives up: `Judge(evidence)` could not be called without evidence.

    **`hash=False`, and it is a compatibility fix rather than a preference.** `Ctx` is a frozen
    dataclass on the public surface, so it advertised hashability, and every `Ctx` was hashable
    before this field existed. A `Report` carries a `Mapping`, so embedding one made a `Ctx`
    *conditionally* unhashable — fine until a state produced a tree, then `TypeError` — which is
    worse than uniformly unhashable, because it fails only on the walk that happens to produce one.
    Excluding it from the hash keeps the coordinates as the identity, which is what they always
    were. It stays in `__eq__`: two contexts differing only in what they carry are not equal, and
    unequal objects sharing a hash is exactly what a hash is allowed to do.

    The verdict rides typed as `Any` because the producing state is not statically known here.
    `transition` has already consumed the verdict; a state reading `incoming` wants `measured`,
    `tree` and `summary`.

    **This field is why `Ctx` is generic over the RECORD, and why an embodiment over a record that
    is not `CommandRun` must say so at every worker and judge.** A bare `Ctx` is not
    unparameterized — the default fires, pinning the record to `CommandRun`, which then disagrees
    with the `Evidence[R]` beside it and is refused at `build_specs`. Note who pays: a WORKER never
    touches `incoming` and pays anyway, because the parameter is on the type it receives rather
    than on the field it uses. Declare an alias once (`type SweepCtx = Ctx[SweepState,
    SweepScore]`) — see `tests/test_machine_non_command_predicate.py`. An embodiment over
    `CommandRun` needs none.

    **What the parameter buys, as a command anyone can run.** `Ctx.incoming` is the only place in
    the machine where one state reads a value another state produced; everything else is checked
    against a signature at `build_specs`. So it is the only place an untyped read cannot be caught
    by the producer. In `tests/test_machine_non_command_predicate.py::propose`, misspell the field
    (`prior.asi` -> `prior.asi_typo`) and delete the local annotation on `prior`:

    - as written, `ty` reports `unresolved-attribute` — one diagnostic;
    - with this field widened to `Report[Any, Any]`, `ty` passes and the typo ships.

    Keep the annotation and both spellings error, so a measurement that keeps it measures the
    annotation rather than the parameter. The parameter's purchase is over the read an author
    did NOT restate the type for."""


@dataclass(frozen=True, slots=True)
class Evidence[R: Measured = CommandRun]:
    """What a worker produced, and all a judge may look at.

    **`measured` is what makes three of the states need no model.** A worker that ran the predicate
    hands the measurement forward, and the judge is then `verdicts.verdict_for_*` — a pure `match`
    over a `CommandRun` rather than an opinion. It is optional because EXPLORE, REPL and REVIEW
    have
    nothing mechanical to read; `None` says "not measured", where an empty `CommandRun` would say
    "measured, and clean"."""

    summary: str
    detail: str = ""
    measured: R | None = None
    tree: Mapping[str, str] | None = field(default=None, hash=False)
    """The workspace AS THIS WORKER LEFT IT, and the reason it travels here rather than in a
    shared object the tools mutate.

    A handler-side tool may edit a real workspace, but the WORKFLOW may only learn about it
    through a recorded op result — otherwise a replay, where no tool runs, re-derives different
    state and diverges: a postamble hashing a shared dict is caught by `ReplayHandler` at the
    artifact key. A worker
    that changed the tree returns it; one that did not leaves this `None` and the machine keeps
    what it had."""


@dataclass(frozen=True, slots=True)
class Report[V: StrEnum, R: Measured = CommandRun]:
    """What a STATE produced: its verdict, and the evidence that verdict stands on.

    One record rather than two returns, because the two travel together everywhere — the verdict
    goes to `transition`, the rest goes to the next visit's `Ctx.incoming` and to the postamble.
    A `tuple[V, Evidence[R]]` says the same thing and reads worse at every construction site.

    `Evidence` remains the FUSED form's currency: `specs.fuse` hands a worker's `Evidence` to a
    judge and lifts the pair into one of these. The two types differ by exactly one field, and
    that is the distinction — evidence is what you produced, a report is evidence plus a ruling
    on it."""

    @classmethod
    def of[V2: StrEnum, R2: Measured](cls, verdict: V2, evidence: Evidence[R2]) -> Report[V2, R2]:
        """Lift a worker's evidence and a judge's verdict into one report — STRUCTURALLY.

        `fuse` hand-enumerated the copy, and a reviewer showed what that costs: three of the four
        assignments could be DELETED with all 3857 tests still passing, so a fifth `Evidence`
        field would have vanished in silence. Reading the field list off `Evidence` means the copy
        cannot fall behind the type — and if `Evidence` grows a field `Report` lacks, this raises
        at the call rather than dropping it."""
        return cast(
            "Report[V2, R2]",
            cls(verdict, **{f.name: getattr(evidence, f.name) for f in fields(Evidence)}),
        )

    verdict: V
    summary: str = ""
    detail: str = ""
    measured: R | None = None
    tree: Mapping[str, str] | None = field(default=None, hash=False)
    """The workspace as THIS VISIT's state left it, and *only* this visit's.

    Not the machine's accumulated workspace: the trampoline REPLACES `carried` with whatever a
    state returns rather than merging, and this is `None` whenever the visit changed nothing — so
    a successor reading `ctx.incoming.tree` sees the predecessor's edit or `None`, never the
    running total. The running total is `Ctx.tree`.

    `hash=False` for the reason `Ctx.incoming` carries it: a `Mapping` field makes a frozen
    dataclass conditionally unhashable, and this type is exported."""


type Run[S: StrEnum, V: StrEnum, R: Measured = CommandRun] = Callable[
    [Ctx[S, R]], Effect[Report[V, R]]
]
"""What a state DOES — the ONE slot. `V` is this state's fibre of the dependent sum.

A worker followed by a judge is a CONSTRUCTION over this slot (`specs.fuse`) that an embodiment
may decline. Hardcoding the pair in the interpreter would bake a graph in as a product, one level
below the defect this package exists to remove."""

type Worker[S: StrEnum, R: Measured = CommandRun] = Callable[[Ctx[S, R]], Effect[Evidence[R]]]
type Judge[S: StrEnum, V: StrEnum, R: Measured = CommandRun] = Callable[
    [Ctx[S, R], Evidence[R]], Effect[V]
]
"""The FUSED form's two halves, kept because most states want them.

**A judge takes `Ctx` even though most judges ignore it**, and that uniformity is the point:
every state's run and every fused judge knows where it is (the transition and the postamble get
no `Ctx`, deliberately). Without it, a fused judge is the single thing in
the machine with no coordinates — the asymmetry that let a judge author a canonical `event_id` it
could not make distinct, so two visits collided and the run DIED on a `PlacedWriterCollision`,
losing the postamble's commit and outcome rows with it. Loudly, not silently — `op_key`'s
same-task/different-placement arm refuses rather than dropping.

The cost is a least-authority one, and it is real: a verdict that could only be a function of its
measurement can now reach for a visit number. `Ctx` carries nothing non-deterministic, so the
hazard is coupling rather than replay — and a judge that ignores it says so *visibly* by naming
the parameter `_ctx`, where the narrow type left that implicit.

Whether a judgment should be its own state is a separate question. Needing coordinates is
answered by `Ctx`; needing an ADDRESS is a reason to split: its own budget line, its own edge in
the transition table, or a binding target a pack can name."""


@dataclass(frozen=True, slots=True)
class Fused[S: StrEnum, V: StrEnum, R: Measured = CommandRun]:
    """A state's run as a worker and then a judge.

    Called, it is a `Run`. The walk runs the two phases of a `Fused` itself, so a judge refused
    after its worker returned leaves that worker's tree, and a refusal from any other phase leaves
    the tree of the last visit that returned. A subclass, a `partial` or a callable forwarding to
    one is an ordinary run, so its judge's refusal leaves the last returned tree."""

    worker: Worker[S, R]
    judge: Judge[S, V, R]

    def __call__(self, ctx: Ctx[S, R]) -> Effect[Report[V, R]]:
        evidence = yield from self.worker(ctx)
        verdict = yield from self.judge(ctx, evidence)
        return Report.of(verdict, evidence)


@dataclass(frozen=True, slots=True)
class StateSpec[S: StrEnum, V: StrEnum, R: Measured = CommandRun]:
    """One state's declarations. `V` ties `run`'s verdict to this state's fibre, which is what
    makes an `Exhausted` return a type error rather than a convention."""

    state: S
    run: Run[S, V, R]
    canonical: bool = False
