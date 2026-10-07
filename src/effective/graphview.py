"""The run graph as a projection, and the projection that folds it back into cycles.

A workflow's graph is never declared, only projected from what ran. This is that projection, as
data and as Mermaid. It is a **pure function of recorded op keys**, with no DB, engine or `agent`
import: a caller hands it the key sequence a reader produced
(`effective.checkpoints.keys(read_sqlite_task(...))`) and gets nodes and edges back.

**The trace is a product space, and a view is a projection that drops an axis**: a 2D
projection of a 3D space, one dimension being time. A node's name carries a program
position and up to three coordinates that say WHICH execution of it this was:

        d:1;  gather:0,1;  ask   #2
        └─┬─┘  └────┬────┘  └┬┘  └┬┘
        scope     branch    pos  occ

| keep                     | view                                                               |
|--------------------------|--------------------------------------------------------------------|
| every coordinate         | the **unrolled DAG**: what actually ran, one node per execution.   |
|                          | Acyclic by construction, because the occurrence index makes every  |
|                          | repeat a distinct name.                                            |
| none of the unrolling    | the **cycle view**: the program's own shape, recovered. A loop     |
| coordinates              | that unrolled into a chain folds back into a loop; N identical     |
|                          | gather branches fold into one edge with a count; a `descend` drill |
|                          | folds into a self-edge.                                            |

The cycle view is the exact inverse of the unrolling (a cycle in the program is a chain in the
trace), and it is what makes a long run legible: a 300-node agent trace is a 6-node cycle graph
with counts on the edges.

Which coordinates count as unrolling is a ruling rather than a syntax question, and it is made at
the MINT: every coordinate declares a role (`keys.marker.Role`), the lint records it beside the
field, and `KeyMap.project` reads it back off a stored key. A fold drops an `Index` and keeps a
`Name`, so one term carrying both (`govern:{gate},{run_id},pass-n={n}`) is answerable, which a
per-tag table could not be.

Both are projections of the same record, and neither is authoritative. Nothing here can describe an
edge that did not execute.

**Three projections live here, and they are two relational operations.** Which one a caller wants
follows from what the view is FOR:

- `fold_cycles`: π with the axes DECLARED. It reads the ROLES each coordinate declared at its
  mint, so its quotient does not depend on the tape. That is what cross-version alignment and
  equivalence need: two runs of one program must project the same way or a diff between them
  means nothing.
- `project`: π, with the axes DISCOVERED from the tape (`axes`). An all-integer column that
  varies is a counter the program walked; anything in BIJECTION with one is the same axis wearing
  two names; a constant is not an axis. It needs no tuple and no declaration, and it reaches
  coordinates a list of scope tags structurally cannot: a lane's ident, a depth echoed inside an
  event name, a ledger id. Its quotient moves with its data, which is right for making *this* run
  legible and wrong for comparing two.
- `restrict`: selection, over ROWS where π chose columns. Edges CONTRACT rather than delete, and
  `RunGraph.hidden` records what was not drawn, because filtering is a projection and never a
  redaction. It composes with π in either order.

Both π's share `regroup`, so a new quotient supplies only its equivalence and inherits the
arithmetic that must hold whatever the quotient is.

**One node is the exception, and it is marked as one**: `from_keys(..., pending=…)` appends the
await a run is suspended on *right now*. No engine checkpoints a pending await, so that node is
synthesized rather than read, which is why it arrives through its own parameter, in state
`PARKED`, instead of in the recorded key sequence.
"""

import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from itertools import pairwise
from string.templatelib import Interpolation, Template
from typing import assert_never

from effective.keys import Key
from effective.keys.frame import (
    ARM_TAGS,
    FRAME_ARMS,
    GATHER_ARM,
    RACE_ARM,
    branch_frames,
    is_branch_frame,
    past_frames,
    split_frames,
)
from effective.keys.grammar import (
    ARITY_SEPARATOR,
    FOREIGN_SIGIL,
    TAG_SEPARATOR,
    TERM_SEPARATOR,
    KeySyntaxError,
    Kind,
    ParsedKey,
    ProjectedKey,
    Term,
    parse,
    split_occurrence,
)
from effective.keys.grammar import kind_of as atom_kind
from effective.keys.marker import Index, Role
from effective.keys.registry import KeyMap

GATHER_TAG = f"{GATHER_ARM}{TAG_SEPARATOR}"
"""The branch-coordinate tag, kept for callers that ask about a gather by name; `KINDS` omits it
deliberately — a gather is structure, not a node kind."""

BRANCH = re.compile(
    rf"^{re.escape(GATHER_TAG)}(\d+){re.escape(ARITY_SEPARATOR)}(\d+){re.escape(TERM_SEPARATOR)}"
)
"""One level of gather nesting: `gather:{g},{i};` — the ordinal of the gather in its enclosing
scope, and the branch index within it. Nested gathers stack, so this matches repeatedly.

**Built from the grammar's own constants, not spelled.** The pattern states four facts — the arm's
tag, the tag separator, the arity separator, the term separator — and spelling them inline made a
regex the fifth place a grammar change has to reach, with nothing to say it had been missed. A
full term walk was measured as the alternative and is exactly equivalent over 7,883 corpus names,
but it costs a dozen lines to replace one and reads worse; the duplication was the defect, not the
regex."""

LEDGER_TAG = "ledger;"
"""`op_key(AppendLedgerRow)`'s tag. Named because `KINDS` and `ledger_collisions` both need it and
a second spelling is exactly how the fork scope token broke — one constant, both readers."""

AWAIT_TAG = "event;"
"""`op_key(AwaitEvent)`'s tag. A branch's await carries its coordinate INSIDE this arm
(`event;gather:0,0;ev:r1`), the spelling `parked.pending_key` documents."""

KINDS: tuple[tuple[str, str], ...] = (
    (LEDGER_TAG, "ledger"),
    ("artifact:", "artifact"),
    ("sleep:", "sleep"),
    (AWAIT_TAG, "await"),
    ("$awaitEvent:", "await"),
)
"""Leading tags that name a node's kind. Derived from the key grammar rather than from a separate
annotation, so a node's kind cannot drift from its identity. Anything unmatched is a `step`: the
author's own op.

**Two tags, one kind.** `event;` is `op_key(AwaitEvent)`, the substrate's name for the op, and
`$awaitEvent:` is Absurd's post-delivery checkpoint for the same await. Both map to `await`, the
word the domain uses. Both rows are needed: an `exclude=()` view genuinely contains
`$awaitEvent:` keys (pinned in `test_parked_reader.py`), and without its row a delivered await
would render as a plain rectangle."""


COMMITTED = "committed"
"""A node whose op ran and checkpointed — the only state a recorded key can be in, and therefore
the default."""

PARKED = "parked"
"""The one node in a graph that no bookkeeper holds: the await a run is suspended on right now,
synthesized from the park reader (`effective.parked.pending_key`) because neither engine writes a
checkpoint for an await while it is pending."""

STATE_PRIORITY: tuple[str, ...] = (COMMITTED, PARKED)
"""Which state survives when `fold_cycles` collapses several executions into one node: LAST wins.

The fold is a projection over a *class* of executions, so it has to answer what one node says
about many. An agent loop parked at its third `review` (`review, review#2, review#3(parked)`)
folds to a **parked** node in the cycle view, the default view.

Ordered by *what a reader has to act on*: a run sitting at a park is the live fact, and a sibling
execution of the same op having committed earlier does not make it less live. Only these two words
have a producer (`Node.state` defaults to `COMMITTED`; `PARKED` comes from `pending_key`'s node or
the `states` mapping), so only these two are ranked, and an **unranked state outranks every ranked
one** (`_rank`): a state with no producer yet (`refused`, `not-reached`) will surface rather than
be silently folded away when it grows one. Extending this is appending a word, in increasing order
of what it demands of a reader."""


def _rank(state: str) -> int:
    """A state's fold priority. An unknown state ranks above every known one — a fold must not
    silently drop a fact it has no vocabulary for, and the safe direction is loud."""
    return STATE_PRIORITY.index(state) if state in STATE_PRIORITY else len(STATE_PRIORITY)


def format_cost(usd: float) -> str:
    """A measured cost, rendered so a REAL one never reads as zero.

    A live gpt-5-nano turn can cost under $0.00005, which `.4f` renders `$0.0000`:
    indistinguishable from free, the claim the `None`/`0.0` distinction exists to avoid making.

    One helper because both renderers spell this (`graphlayout.svg._detail` and `to_mermaid`
    below), and a second spelling is how these drift apart."""
    return f"${usd:.6f}" if 0 < usd < 0.001 else f"${usd:.4f}"


def kind_of(key: str) -> str:
    """A node's kind, from its key alone. FRAMES are stripped first (`bare_name`) — both a
    `scoped(...)` atom and a gather coordinate — because a `ledger;` op inside a frame is a ledger
    node that happens to live there, not a node of kind 'gather' or 'step'."""
    return next((kind for tag, kind in KINDS if bare_name(key).startswith(tag)), "step")


def bare_name(key: str) -> str:
    """A key with every FRAME dropped — both `scoped(...)` atoms and gather coordinates — leaving
    the canonical name the author's op composed.

    **Peel one leading term at a time and stop at an ARM** — the same walk `grammar.past_frames`
    makes, with the widest possible drop set: every leading term is a frame until an arm says the
    op's own key has begun. `ARM_TAGS` is closed, so that boundary is decidable.

    **It stops at a foreign head, where `keys.frame.split_frames` peels the frames behind one.**
    Both callers here want this answer: `kind_of` matches `$awaitEvent:` as a node kind, and
    `ledger_collisions` never meets a wrapped key. A third caller asking about CONTAINMENT wants
    `split_frames` instead.

    **A walk that peels one term at a time, to a fixed point.** `unframed` returns from the
    EARLIEST boundary starting a known tag, so `gather:0,0;rec:0;ledger;x:1` comes back whole: a
    single strip would leave a ledger node reported as kind `step`, and a collision
    `ledger_collisions` could not see. And `unframed` LEAPS: from `rec:0;step;ledger;dup:m1` it
    jumps straight to `ledger;dup:m1`, reading past the `step` arm that `grammar.STEP_ARM` says no
    structural rule may read past, which would report two runs of one author-named step as a lost
    ledger row. A walk that peels one term cannot leap.

    Named and shared because two callers need this exact normalization and a second spelling is how
    the drift starts: `kind_of` asks *what kind of node is this*, `ledger_collisions` asks *is this
    the same identity as that one*.

    **It differs from `fold_cycles` deliberately.** This drops
    frames by looking for the first known OP tag, so it looks past every frame including a
    `sub:`/`task:` one — right for both its callers, since a ledger append inside a subagent is
    still an append of that id. `fold_cycles` asks a third question, *what program position is
    this*, and drops only coordinates whose declared role is in its drop set, so a `sub:`-scoped
    pair is one identity here and two positions there."""
    rest = key
    while True:
        head, separator, tail = rest.partition(TERM_SEPARATOR)
        tag = head.partition(TAG_SEPARATOR)[0]
        if not separator or tag.startswith(FOREIGN_SIGIL):
            return rest
        if tag in ARM_TAGS and tag not in FRAME_ARMS:
            return rest
        rest = tail


def strip_branches(key: str) -> str:
    """Drop every `gather:{g},{i};` frame — the branch coordinate."""
    while (match := BRANCH.match(key)) is not None:
        key = key[match.end() :]
    return key


def branch_path(key: str) -> tuple[tuple[int, int], ...]:
    """The `(gather ordinal, branch index)` pairs a key sits under, outermost first."""
    path: list[tuple[int, int]] = []
    while (match := BRANCH.match(key)) is not None:
        path.append((int(match.group(1)), int(match.group(2))))
        key = key[match.end() :]
    return tuple(path)


@dataclass(frozen=True)
class Node:
    """One node of a run graph: an op that committed (or, in a folded view, a class of them)."""

    key: str
    kind: str
    occurrence: int = 1
    path: tuple[tuple[int, int], ...] = ()
    count: int = 1  # >1 only in a folded view: how many executions this node stands for
    state: str = COMMITTED
    cost: float | None = None
    """Dollars attributed to this node, summed over its executions when folded.

    **`None` means UNMEASURED**, where `0.0` means measured and free; a default of `0.0` would
    report a plausible number for a measurement that never happened.

    | value   | producer's case                                                |
    |---------|----------------------------------------------------------------|
    | dollars | a model call whose usage carried a price                       |
    | `0.0`   | a model call whose usage carried no price (measured and free)  |
    | `None`  | a tool node: a TOOL span has no `effective.cost.usd` at all    |

    `telemetry.measurements` folds spans and `telemetry.sidecar_measurements` folds a JSONL
    sidecar, both into the mapping this field is filled from. They can key on op keys because
    `traced` reads `layers.current_placement()`, so the domain layer sees the placed key."""

    duration_ns: int | None = None
    """Wall-clock attributed to this node, summed over its executions when folded.
    `None` means unmeasured — see `cost`."""

    members: tuple[str, ...] = ()
    """In a folded view, the UNROLLED keys this node stands for — the drill-down.

    The fold is a **lens, not a summary**: a cycle projection can always be unrolled to see any
    specific step within it. The tape
    is the ground truth and the view is a projection over it, so a folded `ask x100` still knows
    the hundred keys it came from, and each of those is a key the checkpoint store can be queried
    with. Nothing is thrown away by looking at the compact picture — which is exactly what a
    declared-graph tool cannot promise, because there the cycle is the primitive and the
    individual iterations may never have been recorded as distinct things at all."""

    frames: tuple[str, ...] = ()
    """The scope frames this node sits inside, outermost first — the CONTAINMENT relation.

    `members` and this field are the projection's two multiplicities, and they answer different
    questions. `members` collapses peers: *show me the hundred executions this one stands for*.
    `frames` nests: *show me what this one is inside*. A machine run's `d:1;state:draft;step;
    tool:apply_fix` reports `('d:1', 'state:draft')` here and keeps `step;tool:apply_fix` as
    its own identity, which is what lets a renderer draw the trajectory as a tree.

    Minted once, in the projector, by `frame.split_frames`: the walk `past_frames` already
    performs. Every consumer that wants the nesting reads it here rather than re-parsing the key,
    so the shaping is shared along with the readers.

    A node's identity is `frames` and `key` together, and is derived rather than stored: the
    partition reconstructs the key, so a second field would be a second spelling of one fact."""

    order: int = 0
    """Where this node ran — its index in the tape, and under a fold the MINIMUM over `members`.

    The one field a reader cannot recompute from a key. `kind`, `occurrence`, `path` and `frames`
    are all functions of the name; position is a fact about the run, so a projection that drops it
    cannot get it back. `graphlayout.prepare` derives the same number by enumerating
    `graph.nodes` and says *"the index* is *the order"* — this carries it where a renderer that
    never goes through the layout IR can read it, which the tree pane needs to sort a level
    faithfully.

    The minimum rather than the first-seen, so a folded node sorts where its earliest execution
    ran. That agrees with the order `regroup` already emits nodes in, so the two cannot disagree
    about a graph."""


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    count: int = 1


@dataclass(frozen=True)
class RunGraph:
    """A run's graph. `cyclic` records which projection this is — the unrolled DAG is acyclic by
    construction, the folded view may contain genuine cycles (that is the point)."""

    run_id: str
    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    cyclic: bool = False
    dropped: tuple[str, ...] = field(default=())
    """Which axes this projection dropped — the view's own statement about itself.

    `fold_cycles` names the grammar's occurrence qualifier and then each role in `drop`, so
    `("occurrence", "index")` is exact about a graph that still carries every `Name` on every key.
    A label reading `"scope"` where a `sub:` frame survived would be the projection overclaiming
    about itself, which is the defect class this axis exists to surface."""

    hidden: tuple[str, ...] = field(default=())
    """The nodes `restrict` stopped DRAWING — selection's counterpart to `dropped`, which
    records the fold.

    It exists so filtering stays *a projection, never a redaction*: a view that
    silently omitted nodes would be a second, lossy bookkeeper, which is the two-bookkeepers rule
    one level out. Every key here is still one the checkpoint store answers to, so a viewer that
    hid the tool calls can always ask what they were."""

    unclaimed: tuple[str, ...] | None = field(default=None)
    """The keys whose INNERMOST term the source map could not read, which is what makes `dropped`
    checkable. `None` where the projection consulted no map, which `project` and the unrolled
    graph both do: an empty tuple would say the map reached everything.

    A projection substitutes only coordinates some registered variant claims, so a fold over keys
    the registry has never been told about drops NOTHING while still reporting the roles it was
    asked for: the label is accurate about the request and silent about the reach.

    **Innermost, because every op carries an arm and every arm binds any payload.** `op_key` puts
    `step;`/`ledger;`/`event;` on the front, each registered as a trailing splice, so "some variant
    decoded this" is true of every recorded key whatever sits behind it. `step;myown:3` is the case
    that matters: a namespace nobody registered, carrying an integer coordinate that declares no
    role, under an arm that decodes.

    Empty on a tape whose ops the substrate minted and whose payloads it registered, which is the
    ordinary case. Non-empty names a domain the map has not been told about."""

    @property
    def executions(self) -> int:
        """How many op executions this graph stands for — invariant across projections, which is
        what makes a folded view honest: it shows fewer nodes, never fewer facts."""
        return sum(node.count for node in self.nodes)


def from_keys(
    run_id: str,
    keys: Sequence[str],
    *,
    states: Mapping[str, str] | None = None,
    pending: Key | None = None,
    # `float | None` / `int | None`, matching this function's own `(None, None)` default and
    # `Node.cost`/`Node.duration_ns`. The narrower `tuple[float, int]` this carried was
    # unfalsifiable while both callers were hand-written literals; the first real producer
    # (`telemetry.measurements`) is a `ty` error against it, which is how the lie surfaced.
    telemetry: Mapping[str, tuple[float | None, int | None]] | None = None,
) -> RunGraph:
    """The UNROLLED graph: one node per recorded op, edges in commit order.

    `keys` is what a reader returns (`effective.checkpoints.keys(...)`) — already
    engine-normalized, so this works identically on either engine. Every element of it has a
    **producer**: a row some
    engine actually wrote. `states` optionally overrides one of those nodes' state.

    `pending` is the one node that has no producer — **where the run is now**, the await it is
    suspended on, which neither engine checkpoints while it is pending. A caller
    derives it from the park reader (`effective.parked.pending_key(park)`) and hands it in as a
    plain key, so this module stays a pure function of strings and never learns what a
    `ParkedTask` is. It is appended after the last recorded key and wired with an edge from it, so
    the *whole* difference between a finished run's graph and a parked run's is one trailing node.

    **A separate parameter rather than a `states` entry, because `states` cannot express this.**
    It maps a key that is *already in* `keys` to a state; an entry for a key that is not there is
    silently ignored (pinned in `test_graphview.py`). Making it work would mean letting `states`
    introduce nodes, at which point `keys` would stop meaning "what a reader returned": the same
    record, two projections, one silently assumed to be the other. Its state still comes from
    the same seam, defaulting to `PARKED`: pass `states={key: …}` to say something else about it.

    Commit order is the edge relation, with one caveat: across concurrent gather branches it is a
    race, so an edge between two *different* branches means only "these committed in this order
    in this run". Within a branch, and outside
    gathers, it is program order."""
    # `pending` arrives as a `Key` (an identity `pending_key` composed) and is flattened HERE,
    # once, on the way into the projection — the same exit `checkpoints.keys()` is for the
    # recorded half. Everything below this line is text, because a projection does real string
    # work on names: stripping branch prefixes, folding occurrence suffixes, laying out labels.
    pending_text = None if pending is None else pending.display()
    ordered = (*keys, pending_text) if pending_text is not None else tuple(keys)
    # the pending node's state defaults to PARKED; an explicit `states` entry still wins
    resolved = {**({pending_text: PARKED} if pending_text is not None else {}), **(states or {})}
    nodes = tuple(
        Node(
            key=key,
            kind=kind_of(key),
            occurrence=split_occurrence(key)[1] or 1,
            path=branch_path(key),
            frames=split_frames(key)[0],
            order=index,
            state=resolved.get(key, COMMITTED),
            # `.get(key)` with NO default: a key absent from the mapping is unmeasured, and a
            # `telemetry=None` caller measures nothing at all. Both must stay distinguishable
            # from a measured zero (see `Node.cost`).
            cost=(telemetry or {}).get(key, (None, None))[0],
            duration_ns=(telemetry or {}).get(key, (None, None))[1],
        )
        for index, key in enumerate(ordered)
    )
    edges = tuple(Edge(a, b) for a, b in pairwise(ordered))
    return RunGraph(run_id=run_id, nodes=nodes, edges=edges)


def fold_cycles(
    graph: RunGraph,
    *,
    drop: tuple[type[Role], ...] = (Index,),
    keymap: KeyMap | None = None,
) -> RunGraph:
    """Project the unrolled DAG down onto the PROGRAM's shape: drop occurrence, branch and scope.

    **As classic as projection gets**: it is π on the key's coordinate tuple,
    dropping two columns and grouping what remains — the relational sense and the geometric sense
    are the same operation here, and the implementation is the group-by that implies. It is
    straightforward for the reason he gave: the trace repeats a SHAPE, so folding it is just
    recognising the same shape twice.

    The unrolling coordinates go away and the program position stays, so `ask`, `ask#2`, `ask#3`
    become one node with `count=3`; `gather:0,0;fetch` / `gather:0,1;fetch` become one `fetch` with
    `count=2`; and `d:0;step;tool:judge` … `d:3;step;tool:judge` become one `judge` with `count=4`.
    Edges collapse the same way and carry how many times they were taken — which is what turns a
    300-node agent trace into a handful of boxes with numbers on the arrows.

    **`drop=` is the quotient, read from what each coordinate declared at its mint.** It answers
    the case a per-tag table could not: `govern:{gate},{run_id},pass-n={n}` is one term carrying a
    name, a run and an index, so a set keyed by tag has two answers and both lose something. It
    also says what it dropped: every node's key is a `ProjectedKey`, rendering a dropped
    coordinate as `*`, which no store accepts.

    **`keymap=` is where a CONSUMER supplies its own tags.** `build/key-registry.json` is built
    from `src/` and `examples/`, so a fixture minting `talk:{ident}` declares `Index` at its own
    mint and hands the fold a map built from its own file (`tests/_keymap.including`).

    **What makes a coordinate an `Index`**: its values index repetitions of the SAME code.
    `sub:planner` and `sub:critic` run different prompts and tools, so they are two program
    positions and declare `Name`; `talk:amber` and `talk:birch` run the identical lane, so they are
    one position executed twice, however the coordinate is spelled. Index-versus-name is a tell,
    not the test, which is why `Index` takes a `str`.

    **The result may be genuinely cyclic**, and that is the point: a loop the trace had unrolled
    into a chain (`ask → act → ask#2 → act#2`) folds back into the loop the program actually
    contains (`ask → act → ask`). A cycle in the program becomes a chain in the trace; this is
    that sentence read right-to-left.

    Self-edges survive deliberately: a step that immediately repeats (`ask → ask#2`) folds to
    `ask → ask`, which is the honest picture of a retry loop.

    **State folds by `STATE_PRIORITY`, not by first-node-wins** — otherwise a run parked at its
    third `review` shows as committed in the very view this projection exists to make legible."""
    resolved = keymap if keymap is not None else _source_map()
    projected = {node.key: _role_projection(node.key, drop, resolved) for node in graph.nodes}
    return regroup(
        graph,
        lambda key: projected[key].display(),
        dropped=("occurrence", *sorted(cls.role for cls in drop)),
        unclaimed=tuple(key for key, p in projected.items() if not p.claimed),
    )


@cache
def _source_map() -> KeyMap:
    """The registry `drop=` reads, loaded once. A caller with its own tags passes `keymap=`."""
    return KeyMap.load()


def _role_projection(key: str, drop: tuple[type[Role], ...], keymap: KeyMap) -> ProjectedKey:
    """A key with every coordinate whose declared ROLE is in `drop` replaced by `*`.

    The answer comes from what each coordinate declared at its mint rather than from its tag,
    which is what lets one term mixing roles (`govern:{gate},{run_id},pass-n={n}`) be answered at
    all.

    The occurrence suffix is split off rather than dropped by a role, because `#N` is the grammar's
    own qualifier on a whole key and no coordinate declares it.

    The `ProjectedKey` rather than its text, because `claimed` is what tells a caller whether the
    map reached this key at all, and the fold reports that as `RunGraph.unclaimed`."""
    return keymap.project(split_occurrence(key)[0], drop=drop)


def regroup(
    graph: RunGraph,
    position: Callable[[str], str],
    *,
    dropped: tuple[str, ...],
    unclaimed: tuple[str, ...] | None = None,
) -> RunGraph:
    """Group a graph's nodes by `position` and sum what they carried — the aggregation half of
    **every** projection here, extracted so a new quotient supplies only its own equivalence.

    `position` IS the quotient: keys mapping to the same string are one node. Everything below is
    the arithmetic that must hold whatever the quotient is — counts sum, members accumulate, state
    takes the highest rank, edges collapse and carry how often they were taken."""
    folded = {node.key: position(node.key) for node in graph.nodes}
    executions: Counter[str] = Counter()
    # Folded telemetry sums only what was MEASURED, and stays `None` when nothing under the fold
    # was: a `Counter` would have turned "no member was measured" back into `0`, reintroducing at
    # the fold exactly the lie `Node.cost` stopped telling at the leaf.
    cost: dict[str, float] = {}
    duration: dict[str, int] = {}
    for node in graph.nodes:
        executions[folded[node.key]] += node.count
        if node.cost is not None:
            cost[folded[node.key]] = cost.get(folded[node.key], 0.0) + node.cost
        if node.duration_ns is not None:
            duration[folded[node.key]] = duration.get(folded[node.key], 0) + node.duration_ns
    members: dict[str, list[str]] = {}
    state: dict[str, str] = {}
    # `order` folds as the MINIMUM, so a folded node sorts where its earliest execution ran. Every
    # other field here sums, ranks or accumulates; this one is the only fold that has to look
    # backwards, which is why it is not derivable from the group's representative.
    order: dict[str, int] = {}
    for node in graph.nodes:
        members.setdefault(folded[node.key], []).extend(node.members or (node.key,))
        name = folded[node.key]
        if name not in state or _rank(node.state) > _rank(state[name]):
            state[name] = node.state
        order[name] = min(order.get(name, node.order), node.order)
    seen: dict[str, Node] = {}
    for node in graph.nodes:
        name = folded[node.key]
        if name not in seen:
            # EVERY field, and the enumeration is the point. This site rebuilt 7 of 9 and let
            # `occurrence` and `path` fall back to their defaults, so a projection silently
            # answered about coordinates it had not computed — and any field added to `Node`
            # inherited the same fate. `frames` and `order` would have been the third and fourth.
            # `dataclasses.fields(Node)` is what to check against when the type grows again.
            #
            # `occurrence` and `path` are recomputed from the FOLDED key rather than carried from
            # the representative, because the quotient may have dropped exactly those coordinates
            # — `fold_cycles` declares that it does. Reading them off the name the node now has
            # keeps the node agreeing with its own key whatever the quotient removed.
            seen[name] = Node(
                key=name,
                kind=node.kind,
                occurrence=split_occurrence(name)[1] or 1,
                path=branch_path(name),
                frames=split_frames(name)[0],
                order=order[name],
                count=executions[name],
                state=state[name],
                cost=cost.get(name),
                duration_ns=duration.get(name),
                members=tuple(members[name]),
            )
    edge_counts = Counter(
        (folded[edge.src], folded[edge.dst])
        for edge in graph.edges
        if edge.src in folded and edge.dst in folded
    )
    edges = tuple(Edge(src, dst, count) for (src, dst), count in edge_counts.items())
    return RunGraph(
        run_id=graph.run_id,
        nodes=tuple(seen.values()),
        edges=edges,
        cyclic=has_cycle(seen, edges),
        dropped=dropped,
        unclaimed=unclaimed,
    )


# --- π, with the axes DISCOVERED rather than declared -----------------------------------------

type Column = tuple[int, str, int]
"""One coordinate position in the tape's space: `(term index, tag, coordinate index)`.

A key is a POINT (one value per column) and the tape is a set of points, so *a view is a choice
of which coordinates to keep*. The columns are more than `(position, occurrence, branch)`: a
lane's `talk:{ident}`, the `depth=` inside a grant's event name, a ledger `event_id` and a cart's
`item:{sku}` are columns too. The space is as wide as the grammar, so the columns are read off the
parse."""


def _point(key: str) -> tuple[tuple[str, ...], dict[Column, str]] | None:
    """A key as its tag sequence and one value per coordinate column, or `None` if it is not in
    the language. The occurrence suffix comes off first — it is an axis by definition."""
    try:
        terms = parse(split_occurrence(key)[0]).terms
    except KeySyntaxError:
        return None
    point: dict[Column, str] = {}
    for t, term in enumerate(terms):
        for c, coordinate in enumerate(term.coordinates):
            point[(t, term.tag, c)] = coordinate.path
    return tuple(term.tag for term in terms), point


def _determines(rows: Sequence[Mapping[Column, str]], a: Column, b: Column) -> bool:
    """`a -> b`: no two rows agree on `a` and differ on `b` — functional dependency, read off
    the tape rather than declared about it."""
    seen: dict[str, str] = {}
    for row in rows:
        if a not in row or b not in row:
            return False
        if seen.setdefault(row[a], row[b]) != row[b]:
            return False
    return True


def axes(graph: RunGraph) -> dict[tuple[str, ...], frozenset[Column]]:
    """The unrolling columns this tape actually has — the quotient, DISCOVERED, per tag sequence.

    Three moves, and no tag name appears in any of them:

    - **Seed.** A column whose atoms are all `Kind.INTEGER` is a counter the program walked. That
      is the `Index` role read structurally instead of declared.
    - **Closure under bijection.** Drop any column that DETERMINES and IS DETERMINED BY one
      already dropped. Two mutually determining columns are one axis wearing two names, and
      collapsing one while keeping the other reduces nothing, which is the shape that made a
      lane's tag need a ruling at all: its ident is in bijection with its gather branch.
      A column merely *determined by* a dropped one survives, which is what keeps
      `step;score:systems` and `step;score:story` two nodes: branch determines the rubric, but a
      rubric does not determine the branch.
    - **A constant is not an axis.** Nothing unrolls, and two constant columns are trivially
      mutually determining — so without this every column collapses whenever a tape has one row.

    The closure is confluent, so the quotient does not depend on the order columns are visited.

    **This is a projection of the TAPE, so it is data-dependent by construction**: one program
    folds differently across runs, because a cart with one line has no axis to fold. That is
    right for a view whose job is to make *this* run legible and wrong for cross-version
    alignment, which is why `fold_cycles` stays exactly as it is and keeps that job."""
    grouped: dict[tuple[str, ...], list[dict[Column, str]]] = {}
    for node in graph.nodes:
        if (point := _point(node.key)) is not None:
            grouped.setdefault(point[0], []).append(point[1])
    found: dict[tuple[str, ...], frozenset[Column]] = {}
    for tags, rows in grouped.items():
        columns = sorted({column for row in rows for column in row})
        varies = {c: len({row[c] for row in rows if c in row}) > 1 for c in columns}
        dropped = {
            c
            for c in columns
            if varies[c] and all(atom_kind(row[c]) is Kind.INTEGER for row in rows if c in row)
        }
        while True:
            more = {
                c
                for c in columns
                if c not in dropped
                # No `varies[c]` here, and its absence is load-bearing: every dropped column
                # varies (the seed requires it, and the closure only adds via bijection with one
                # that does), so a column in bijection with one of them varies too. A guard that
                # cannot fail reads as protection and provides none — mutation-checked.
                and any(_determines(rows, c, d) and _determines(rows, d, c) for d in dropped)
            }
            if not more:
                break
            dropped |= more
        found[tags] = frozenset(dropped)
    return found


def _projected(key: str, found: Mapping[tuple[str, ...], frozenset[Column]]) -> str:
    """`key` with its unrolling coordinates elided — the label of the position it stands for.

    **Coordinates go; terms never do.** A term that loses every coordinate keeps its bare tag,
    which the grammar already reads as a qualifier. That is more verbose than `fold_cycles`'
    output and it is the reason there is no special case here: eliding `item:oat-milk` to `item`
    needs no rule about which terms are frames and which carry the identity, and the surviving
    tag says the node sits inside a collapsed scope instead of pretending it never did."""
    try:
        terms = parse(split_occurrence(key)[0]).terms
    except KeySyntaxError:
        return key
    drop = found.get(tuple(term.tag for term in terms), frozenset())
    rendered = []
    for t, term in enumerate(terms):
        # `Coordinate.render`, not `.path` — a coordinate's NAME is part of its identity
        # (`depth=1` is not `1`), so rendering from the path alone would let an optional
        # coordinate alias a bare one in the label. The dependency arithmetic above compares
        # paths deliberately: `depth=1` and `depth=2` differ in the value, which is the axis.
        kept = tuple(
            coordinate
            for c, coordinate in enumerate(term.coordinates)
            if (t, term.tag, c) not in drop
        )
        # Rebuild the TERM and let the grammar render it, rather than joining tags with `;`.
        # A foreign term that wraps our key joins its payload with `:` — the bytes the engine
        # wrote — and a hand-rolled join emits `$awaitEvent;review:m1`, which parses as a
        # DIFFERENT valid key, so nothing downstream can notice.
        rendered.append(
            Term(term.tag, kept, foreign=term.foreign, wraps_payload=term.wraps_payload)
        )
    return ParsedKey(tuple(rendered)).render()


def project(graph: RunGraph) -> RunGraph:
    """π over the tape's own coordinate space — `fold_cycles`' sibling, with nothing declared.

    `fold_cycles` reads a role DECLARED at the mint, which is nominal typing: a coordinate is an
    axis because its producer said so. What that cannot reach is a coordinate nobody declared, a
    depth echoed inside an event name, a ledger id minted by a domain. This asks the tape instead.

    Both survive on purpose. `fold_cycles` is declared and therefore stable
    across runs, which is what alignment and equivalence need; this one is sharper and moves with
    its data, which is what a dashboard wants."""
    found = axes(graph)  # once, not per key — the quotient is a property of the whole tape
    return regroup(graph, lambda key: _projected(key, found), dropped=("discovered-axes",))


# --- SELECTION: choose the ROWS, where π chose the columns ----------------------------------


def restrict(graph: RunGraph, where: Callable[[Node], bool]) -> RunGraph:
    """Selection over the graph's nodes — keep what a viewer asked to see, and CONTRACT the rest.

    The relational sibling of `project`, and the two are orthogonal on purpose: π collapses
    coordinate *columns*, selection chooses *rows*. As selection it composes with the fold, and
    either order is meaningful (`restrict(project(g), …)` filters the
    program's shape, `project(restrict(g, …))` folds what survived).

    **Edges contract; they are never deleted.** Hiding the tool calls between two ledger rows must
    still draw the edge between them, or the view falls into disconnected islands and is worse
    than the unfiltered one. So a kept node reaches out through the hidden region and lands on
    whatever kept nodes it can reach.

    A contracted edge carries the **flow along its path** — the minimum edge count on the way,
    summed over the distinct paths that arrive. That is exact when the hidden region is a chain or
    a fan (the two shapes a run actually produces: a sequence of steps, or a branch), and an
    approximation when it is a diamond, where the same traffic can be counted along two routes.
    Stated rather than hidden, because a count nobody can reproduce is worse than an absent one.

    `where` is a predicate the CALLER brings rather than a property a node has, which is why there
    is no `Node.salient` field: a parked await is the interesting node when
    you are answering it and noise when you are auditing the record."""
    kept = {node.key: node for node in graph.nodes if where(node)}
    if len(kept) == len(graph.nodes):
        return graph
    outgoing: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        outgoing.setdefault(edge.src, []).append(edge)

    flow: Counter[tuple[str, str]] = Counter()
    for source in kept:
        # Walk out through the HIDDEN region only; a kept node ends the walk (it becomes the
        # destination) rather than being walked through, so `a -> b -> c` with all three kept
        # stays two edges instead of quietly gaining a third.
        stack: list[tuple[str, int]] = [(source, 0)]
        seen: set[str] = {source}
        while stack:
            at, carried = stack.pop()
            for edge in outgoing.get(at, ()):
                reaching = edge.count if carried == 0 else min(carried, edge.count)
                if edge.dst in kept:
                    flow[(source, edge.dst)] += reaching
                elif edge.dst not in seen:
                    seen.add(edge.dst)
                    stack.append((edge.dst, reaching))
    edges = tuple(Edge(src, dst, count) for (src, dst), count in flow.items())
    return RunGraph(
        run_id=graph.run_id,
        nodes=tuple(kept.values()),
        edges=edges,
        cyclic=has_cycle(kept, edges),
        dropped=graph.dropped,
        hidden=tuple(node.key for node in graph.nodes if node.key not in kept),
        unclaimed=graph.unclaimed,  # selection chooses rows; it consults no map of its own
    )


def has_cycle(nodes: Iterable[str], edges: Iterable[Edge]) -> bool:
    """Whether a directed graph contains a cycle — Kahn's algorithm, peel the sources.

    A self-edge check would miss the interesting case, `ask -> act -> ask`: a two-node loop with
    no self edge at all. The unrolled graph always answers False here, since the occurrence index
    makes every repeat a distinct node; the folded one answers True exactly when the program
    really does loop."""
    remaining = {node: 0 for node in nodes}
    outgoing: dict[str, list[str]] = {node: [] for node in remaining}
    for edge in edges:
        if edge.src not in remaining or edge.dst not in remaining:
            continue
        outgoing[edge.src].append(edge.dst)
        remaining[edge.dst] += 1
    ready = [node for node, degree in remaining.items() if degree == 0]
    peeled = 0
    while ready:
        node = ready.pop()
        peeled += 1
        for successor in outgoing[node]:
            remaining[successor] -= 1
            if remaining[successor] == 0:
                ready.append(successor)
    return peeled < len(remaining)


_SHAPES: Mapping[str, tuple[str, str]] = {
    "step": ("[", "]"),
    "ledger": ("[(", ")]"),  # a cylinder: it writes to the canonical record
    "artifact": ("[/", "/]"),
    "await": ("{{", "}}"),  # a hexagon: the run can stop here
    "sleep": ("(", ")"),
}


def _ident(index: int) -> str:
    return f"n{index}"


def _label(text: str) -> str:
    """Escape a node label for Mermaid.

    Keys carry author text in their TERMINAL position, so a quote or a newline in an event id
    reaches the renderer. Unescaped, both break the diagram silently: `["he said "hi""]` is a
    parse error.

    Mermaid's flowchart documentation supports HTML character names as entity codes, so `#quot;`
    is doc-supported. `to_mermaid` also wraps every label in quotes (`n0["…"]`), the docs'
    recommended handling, and a raw `"` would terminate that wrapper.

    The render itself is unverified: it would need mermaid.js in-tree, a pinned and sandboxed npm
    toolchain (`infra/elkjs` is the precedent). `test_a_label_carrying_author_text_...` pins the
    property that would break the parse: the emitter never puts a bare `"` inside the wrapper."""
    return text.replace('"', "#quot;").replace("\n", "<br/>").replace("\r", "")


@dataclass(frozen=True)
class Collision:
    """One `event_id` written by more than one placed writer in a single run — a **placed-writer
    collision**.

    `writers` are the unrolled op keys that wrote it, so the report names the culprits rather than
    only the symptom — a folded node's `members` is exactly the drill-down the fold promises."""

    event_id: str
    writers: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.writers)


def ledger_collisions(graph: RunGraph) -> tuple[Collision, ...]:
    """Where the two bookkeepers disagree: ledger appends the tape separates and the canonical
    record cannot.

    **A collision IS a fold, which is why this is a group-by and not a module.** The defect is *one
    `event_id`, two placed writers*, and what makes a writer placed is exactly the coordinates a
    projection drops — occurrence, branch, and the enclosing frames. So two ledger keys sharing a
    `bare_name` is not a heuristic for the defect, it is the definition of it. Nobody had asked the
    view this question.

    **It does NOT reuse `fold_cycles`, and that is the load-bearing choice here.** The fold answers
    *what program position is this*, so it keeps a `sub:{name}` frame — two subagents are two
    positions. This asks *is this the same identity*, and two subagents appending one `event_id` is
    exactly the defect: the canonical record has no `sub:` coordinate to separate them with, so one
    of the rows is lost. Grouping by `bare_name`, which looks past every frame down to the op tag,
    is what makes that reportable. The two normalizations answer different questions and are
    meant to disagree here.

    The scopes line up, and not by luck. A tape is per TASK, and the store-side refusal's
    predicate is per task, so the cross-generation idempotency this substrate documents as a
    FEATURE
    (`handlers/absurd.py:1185-1190`: one row for a message triaged in generation 0 and again in 3)
    lives across tasks and is structurally invisible here. This cannot false-positive on it.

    What it is for, until the refusal lands: `run_fork` seals over a lineage that silently lost a
    row (measured), and a dashboard reading only the checkpoint bookkeeper draws that run as
    correct. This is the read-path half (no schema change, both engines) that lets a surface say
    *the record disagrees with the tape* instead of rendering a confident wrong picture. It is a
    DETECTOR, not a guard: it reports after the fact, and only the store-side refusal prevents the
    loss.

    Takes the UNFOLDED graph (`from_keys(...)`); a folded one has already discarded the members
    this reports as culprits."""
    writers: dict[str, list[str]] = {}
    for node in graph.nodes:
        if node.kind == "ledger" and node.state == COMMITTED:
            writers.setdefault(bare_name(split_occurrence(node.key)[0]), []).append(node.key)
    return tuple(
        Collision(event_id=name.removeprefix(LEDGER_TAG), writers=tuple(keys))
        for name, keys in writers.items()
        if len(keys) > 1
    )


def to_mermaid(graph: RunGraph, *, direction: str = "TD", title: str | None = None) -> str:
    """Render a graph as a Mermaid flowchart: the portable target.

    Mermaid because it is TEXT: an agent reading a projection gets the same artifact a human looks
    at, and it survives into a report or a commit message unchanged.

    **What renders it is markdown-on-GitHub, and nothing else in this repo**:
    `cards/render_html.py` ships a Vega bootstrap and no Mermaid one, and there is no `Diagram`
    cell in the IR. Serving this to a browser needs a mermaid.js bootstrap.

    Node labels carry the count when a folded node stands for more than one execution, and edge
    labels carry the traversal count when an edge was taken more than once."""
    ids = {node.key: _ident(i) for i, node in enumerate(graph.nodes)}
    lines = [f"graph {direction}"]
    if title:
        lines.insert(0, f"---\ntitle: {title}\n---")
    for node in graph.nodes:
        open_, close = _SHAPES.get(node.kind, ("[", "]"))
        label = _label(node.key if node.count == 1 else f"{node.key} x{node.count}")
        if node.state != COMMITTED:
            label = f"{label}<br/>({node.state})"
        # Each half is rendered only if MEASURED — mirroring `graphlayout.svg`, and now
        # load-bearing rather than cosmetic: an unmeasured node (`None`) must not render as
        # `$0.0000`, which is the claim `Node.cost` exists to avoid making.
        #
        # `is not None`, NOT truthiness, and the comment above is why the difference matters. A
        # measured-and-free call is `0.0` (`Usage.as_attributes()` always emits
        # `effective.cost.usd`, so an unpriced model produces exactly that), and truthiness
        # rendered it identically to `None`, collapsing "this cost nothing" into "nobody measured
        # this". That is the same claim `Node.cost` stopped being a `float` to avoid, arriving at
        # the last step before a human reads it. Measured: both renderings were byte-identical, and
        # changing both sites reddened nothing across 138 tests, so neither behaviour was pinned in
        # either direction.
        badge = []
        if node.cost is not None:
            badge.append(format_cost(node.cost))
        if node.duration_ns is not None:
            badge.append(f"{node.duration_ns / 1e6:.0f}ms")
        if badge:
            label = f"{label}<br/>{' · '.join(badge)}"
        lines.append(f'  {ids[node.key]}{open_}"{label}"{close}')
    for edge in graph.edges:
        if edge.src not in ids or edge.dst not in ids:
            continue
        arrow = f' -->|"x{edge.count}"| ' if edge.count > 1 else " --> "
        lines.append(f"  {ids[edge.src]}{arrow}{ids[edge.dst]}")
    return "\n".join(lines)


_SEQUENCE_TEXT = str.maketrans(
    {";": "#59;", "#": "#35;", "<": "#60;", ">": "#62;", "\n": " ", "\r": ""}
)
"""Mermaid ends a sequence statement at `;`, reads `#` as an entity's start and `<` as markup,
and a key can carry each, so each renders as its entity."""


def to_sequence(graph: RunGraph) -> str:
    """Render a run as a Mermaid sequence diagram: each branch's ops, in the order they committed.

    The participants are the workflow's main line, one per gather or race branch, and the world
    the ops reach. Each node is drawn on the branch it ran in, labeled with its key past the
    gather and race frames (`past_frames`):

    | node                        | drawn as                                                     |
    |-----------------------------|--------------------------------------------------------------|
    | an op                       | a message from the branch to the world                       |
    | an await                    | a note on the branch, then the world's reply if it committed |
    | a race's choice or endings  | a note on the branch that ran the race                       |

    Within a branch the messages are program order. Across branches commit order is one run's
    interleaving (`from_keys`), so the participants are ordered by coordinate, and by name where
    coordinates tie."""
    nodes = sorted(graph.nodes, key=lambda node: node.order)
    drawn = frozenset(branch_frames(node.key) for node in nodes)
    branches = sorted(
        {_branch(node.key, drawn) for node in nodes} - {()},
        key=lambda frames: (_coordinates(frames), frames),
    )
    lanes = {(): "run"} | {frames: "b" + str(i) for i, frames in enumerate(branches, 1)}
    lines = [_statement(t"sequenceDiagram"), _statement(t"  participant run as workflow")]
    for frames in branches:
        lane, name = lanes[frames], " / ".join(frames)
        lines.append(_statement(t"  participant {lane} as {name}"))
    lines.append(_statement(t"  participant world"))
    for node in nodes:
        branch = _branch(node.key, drawn)
        inner = split_frames(node.key)[0][len(branch_frames(node.key)) :]
        lane, label = lanes[branch], past_frames(node.key)
        if node.kind == "await":
            lines.append(_statement(t"  Note over {lane}: awaits {label}"))
            if node.state == COMMITTED:
                lines.append(_statement(t"  world-->>{lane}: {label}"))
        elif inner and _is_race(race := inner[-1]):
            lines.append(_statement(t"  Note over {lane}: {race} {label}"))
        else:
            lines.append(_statement(t"  {lane}->>world: {label}"))
    return "\n".join(lines)


def _branch(key: str, drawn: frozenset[tuple[str, ...]] = frozenset()) -> tuple[str, ...]:
    """The branch frames `key` ran under. A branch's await carries them at the head of its
    address, inside the await arm; an authored name can spell a branch frame too, so frames
    found deeper in the address count only when another node ran under them (`drawn`). The
    address alone cannot tell a scoped branch's await from an authored name that spells the same
    frames, so either can land on the wrong line: `place-awaits-by-provenance-task`."""
    if frames := branch_frames(key):
        return frames
    if not key.startswith(AWAIT_TAG):
        return ()
    frames = branch_frames(key.removeprefix(AWAIT_TAG))
    return frames if frames and (is_branch_frame(frames[0]) or frames in drawn) else ()


def _statement(template: Template) -> str:
    """One statement of a sequence diagram: the static text as written, and each hole as text with
    `;`, `#`, `<`, `>` and line breaks rendered as entities."""
    parts = []
    for item in template:
        match item:
            case str() as text:
                parts.append(text)
            case Interpolation(value=value):
                parts.append(str(value).translate(_SEQUENCE_TEXT))
            case unreachable:
                assert_never(unreachable)
    return "".join(parts)


def _coordinates(frames: tuple[str, ...]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Branch frames as `(tag, integers)` pairs, so `gather:0,10` sorts after `gather:0,2`."""
    placed = []
    for frame in frames:
        tag, _, coordinates = frame.partition(TAG_SEPARATOR)
        numbers = coordinates.split(ARITY_SEPARATOR)
        placed.append((tag, tuple(int(n) for n in numbers if n.isdigit())))
    return tuple(placed)


def _is_race(frame: str) -> bool:
    """A `race:{r}` frame, the one a race's own choice and endings sit under."""
    tag, _, coordinates = frame.partition(TAG_SEPARATOR)
    return tag == RACE_ARM and ARITY_SEPARATOR not in coordinates


def identity(node: Node) -> str:
    """A node's key with its frames peeled — what a tree draws at the leaf.

    Derived rather than stored, because the key already determines it — a third field would be a
    second spelling of one fact and could disagree with the other two.

    It asks the partition rather than stripping the frames as a prefix: a wrapping foreign term
    rides with the identity while the frames behind it do not, so the frames are not always a
    literal prefix, and a strip that misses returns the whole key as the leaf.

    The `#N` stays on. It rides with the identity because a frame never carries one, and it is
    what tells three executions of one step apart in the unrolled view — the only view that shows
    them separately."""
    return split_frames(node.key)[1]


def _badge(node: Node) -> str:
    """The measured half of a label, and only the measured half.

    Same rule as `to_mermaid`, and it is the `Node.cost` claim arriving at the last step before a
    human reads it: `is not None` rather than truthiness, so a measured-and-free `0.0` renders as
    `$0.0000` instead of vanishing into the same blank an unmeasured node gets."""
    parts = []
    if node.count > 1:
        parts.append(f"x{node.count}")
    if node.cost is not None:
        parts.append(format_cost(node.cost))
    if node.duration_ns is not None:
        parts.append(f"{node.duration_ns / 1e6:.0f}ms")
    if node.state != COMMITTED:
        parts.append(node.state)
    return f"  ({' · '.join(parts)})" if parts else ""


def to_text(graph: RunGraph, *, title: str | None = None) -> str:
    """Render a run as an indented Unicode tree — CONTAINMENT, where `to_mermaid` draws sequence.

    The second projection of the same value, and the one a terminal can show without a layout
    engine: `Node.frames` is a path, so the tape's scope frames nest exactly as the run did.
    A machine run draws its own trajectory — `d:N` around `state:NAME` around the steps — and
    a flat tape draws a flat list, which is the honest picture of a ReAct loop.

    **Hand-rendered, with no dependency, and that is a choice rather than an omission.** `rich`
    has a `Tree`, and reaching for it here would put a library that arrives as a dev-group
    transitive of an example (`code-agent` -> `typer` -> `rich`, imported nowhere in this repo)
    inside the lean-deps half of the public seam. `to_mermaid` is the precedent: a projection
    writes its own text. The box-drawing characters assume Unicode, which the design ruling
    already grants — *"we are not targeting old vt100s"*.

    **Each level sorts by `order`, which is the whole reason that field exists.** A tree groups by
    containment, so without it the frameless postamble of a machine run hoists to the top of its
    level and the picture claims an order the run did not have. A folded node sorts by the minimum
    over its members, so it sits where it FIRST ran.

    What a tree cannot show is interleaving: grouping by frame is what a tree IS, so two gather
    branches that alternated on the tape draw as two contiguous blocks. That is a fact about tree
    rendering rather than about this graph — `graph.nodes` is still in run order for a pane that
    wants to show it — and it matters because a gather's branches can interleave at all.
    Whether they do is a property of the CTX rather than of the engine: a ctx advertising
    `concurrent_safe` races, and one without it runs branches in index order. SQLite ships the
    first and the Absurd worker ships the second, so naming the engine here would go false the
    moment either is rewired.
    """
    lines = [title if title is not None else graph.run_id]
    _draw(list(graph.nodes), depth=0, prefix="", lines=lines)
    return "\n".join(lines)


def _draw(nodes: list[Node], *, depth: int, prefix: str, lines: list[str]) -> None:
    """One level of the tree: the nodes whose frame path ends here, and the groups that go deeper.

    Leaves and groups are sorted TOGETHER by run position — a group's position being the earliest
    node under it — so a frame that opened before a sibling leaf ran draws above it. Sorting them
    separately would put every leaf before every group and silently reorder the run.
    """
    groups: dict[str, list[Node]] = {}
    entries: list[tuple[int, str, Node | None]] = []
    for node in nodes:
        if len(node.frames) == depth:
            entries.append((node.order, identity(node) + _badge(node), node))
        else:
            groups.setdefault(node.frames[depth], []).append(node)
    for frame, members in groups.items():
        entries.append((min(member.order for member in members), frame, None))
    entries.sort(key=lambda entry: (entry[0], entry[1]))

    for index, (_, label, node) in enumerate(entries):
        last = index == len(entries) - 1
        lines.append(f"{prefix}{'└─ ' if last else '├─ '}{label}")
        if node is None:
            _draw(
                groups[label],
                depth=depth + 1,
                prefix=prefix + ("   " if last else "│  "),
                lines=lines,
            )


def summarize(graphs: Iterable[RunGraph]) -> str:
    """A one-line size report per graph — what the legibility experiment measures."""
    return "\n".join(
        f"{g.run_id}: {len(g.nodes)} nodes, {len(g.edges)} edges, "
        f"{g.executions} executions{' (cyclic)' if g.cyclic else ''}"
        for g in graphs
    )


__all__ = [
    "COMMITTED",
    "LEDGER_TAG",
    "PARKED",
    "STATE_PRIORITY",
    "Collision",
    "Edge",
    "Node",
    "RunGraph",
    "axes",
    "bare_name",
    "branch_path",
    "fold_cycles",
    "from_keys",
    "has_cycle",
    "kind_of",
    "ledger_collisions",
    "project",
    "regroup",
    "restrict",
    "summarize",
    "to_mermaid",
    "to_text",
]
