"""The substrate's authority park names, as MINTERS — one row per namespace.

**Why a table rather than literals at each call site.** Several tests ask the same question of
different surfaces — *does this reader/page/query see a park in EVERY authority namespace, not just
`approve`?* — and each spelled its own park names by hand:

    _await_wf(Key.parse("govern:spend:r3:0"))        # a FLAT-grammar literal, long stale
    _await_wf(Key.parse(f"budget-grant:{run_id},0")) # an f-string building an identity

Both are the same defect in two costumes. `Key.parse(f"…")` is an **f-string composing a key**,
which this repo's standing posture makes a finding by default — the `Template` is the AST, so
there is no reason to flatten a name into text and parse it back. A hand-spelled literal drifts,
and the drift is INVISIBLE: a test that creates the park from the same literal it asserts is an
instrument agreeing with itself, so it stays green while the real minter produces something else
entirely.

**The rule is POSITIONAL, exactly like the f-string one**: parse flattened strings only when
testing the parser itself. `Key.parse("approve;…")` is correct in
`test_grammar.py`, where the parser IS the subject and a literal is the input under test. It is a
shortcut here, where the subject is a reader or a page and the key is scaffolding — the same call,
legitimate in one position and a finding in the other.

So: **compose through the declared `AuthorityTag`, and let the expected value come from the
minter.** A test that both mints and asserts through one function cannot drift, and adding a
namespace is one row here rather than an edit in every file that enumerates them.

The rows are also what makes `event_name`'s fence testable at all: it refuses a
reserved-authority `Key` carrying no `Scope` — the mark of a PARSED key rather than a composed one
— so `Key.parse("approve;…")` cannot stand in for a substrate park. That refusal is the point,
and these minters are how a test says "I am the substrate" honestly.
"""

from collections.abc import Callable

from effective.budget import (
    GENERATION_GRANT,
    budget_grant_name,
    chain_grant_name,
    depth_grant_name,
    round_grant_name,
)
from effective.domain import CallTool
from effective.govern import GateState
from effective.keys import Key, Segment, compose_key
from effective.ops import CHAIN_GENERATION, Step
from effective.permission import APPROVE, approval_name

type Minter = Callable[..., Key]
"""`(run_id, *, generation) -> Key`, spelled loosely because the rows are lambdas.

**`generation` is REQUIRED of every row, including the rows that ignore it**, and that is the
census doing its job rather than a courtesy. A `respawn` chain keeps `run_id` stable across
generations while each generation is a fresh task, so a namespace whose minter cannot ACCEPT a
generation is a namespace whose name cannot separate one. Making the parameter mandatory means
the signature surfaces that hole for every minter."""


def production_approve_park(run_id: str, *, generation: int) -> Key:
    """The `approve` census row — composed by `permission.approval_name` ITSELF.

    The census asks whether a namespace can tell two askers apart, so its rows have to be the
    names production mints. `approve_park` below cannot serve: it holds a copy of the template,
    and a copy answers about itself. Measured, not supposed — stripping the generation coordinate
    out of `permission` left the copy-composed row green while `govern`'s reddened.

    Production reads the generation from the ambient rather than taking it as an argument, because
    that is how it reaches a park name in a real chain: `respawn` sets `CHAIN_GENERATION` and
    every walk re-derives it. Setting the ContextVar is therefore the faithful way to ask for
    generation N, not a workaround for a missing parameter.
    """
    op = Step(name=run_id, op=CallTool(name=run_id, args={}, result_schema=str))
    token = CHAIN_GENERATION.set(generation)
    try:
        return approval_name(op)
    finally:
        CHAIN_GENERATION.reset(token)


def approve_park(op_key: Key, *, generation: int = 0) -> Key:
    """The human tier's park over a specific op — the shape `permission.approval_name` mints.

    **This is a COPY of production's template, and a copy is the defect this module's docstring is
    about one level up** — an instrument that spells the shape it is checking agrees with itself
    and stays green while production moves. It could not simply call `approval_name`, which takes
    a `WorkflowOp` and derives the op key from its PLACEMENT; six callers here hold an arbitrary
    `Key` (`ledger;processed:m1`) that no op would produce.

    So the agreement is pinned instead of assumed:
    `test_the_census_approve_minter_agrees_with_production` composes both ways over several
    generations and asserts they are equal. That is the same move as deriving a reader from its
    writer — the duplication stays, but it cannot drift silently, which is the only property that
    was ever load-bearing.

    `generation` defaults here and nowhere else: this minter's callers are asking "give me a park
    in the approve namespace", and six of them predate the coordinate entirely. The CENSUS row
    above passes it explicitly, which is where the requirement belongs.
    """
    # lint: terminal-hole — `op_key` is a `Key`, which the rule cannot see.
    return compose_key(t"{APPROVE}:generation={generation:default=0};{op_key:domain=any}")


AUTHORITY_PARKS: list[tuple[str, Minter]] = [
    # Through the minters at BOTH levels, which is this file's own thesis applied to itself: a
    # hand-composed `t"{APPROVE};{Segment(run_id)}"` would put a run id in TERM-HEAD position, a
    # shape production never mints (`permission.py` splices an op key there) and the grammar
    # forbids, since a term head names a namespace.
    # NOT `approve_park`: that is this module's own copy of the shape, and a census row composed
    # through the copy measures the copy. Stripping the generation coordinate from `permission`
    # leaves a copy-composed row GREEN while `govern`'s reddens, because the copy still carries
    # it. Production is one call away, so the row takes it.
    ("approve", production_approve_park),
    ("budget-grant", lambda run_id, *, generation: budget_grant_name(run_id, 0)),
    ("chain-grant", lambda run_id, *, generation: chain_grant_name(run_id, generation)),
    (
        "depth-grant",
        lambda run_id, *, generation: depth_grant_name(run_id, generation=generation, depth=0),
    ),
    (
        "generation-grant",
        # The one namespace the coverage pin FOUND missing when it was written — three coordinates
        # (run, generation, ask), composed inline because `combinators` mints it behind a closure.
        lambda run_id, *, generation: compose_key(
            t"{GENERATION_GRANT}:{Segment(run_id)},{generation},{1}"
        ),
    ),
    (
        "govern",
        lambda run_id, *, generation: (
            GateState(
                gate="spend", run_id=run_id, op_key="step;tool:x", generation=generation
            ).park_name
        ),
    ),
    (
        "round-grant",
        lambda run_id, *, generation: round_grant_name(run_id, generation=generation, rounds=1),
    ),
]
"""(namespace, minter) for every namespace a park can sit in — the parameterization for
"does this surface see them all?".

Derived from `RESERVED_AUTHORITY_TAGS` by hand and pinned against it below, so a namespace added
there without a row here fails rather than silently reducing coverage: the two `gate-state`/`fork`/
`hyp` tags are excluded WITH A REASON in that test rather than quietly missing.

The two rows that DROP `generation` are the two `Scope.ACCRUAL` ones, and nothing here says so:
the property test reads `Key.scope` off the minted key instead, so which namespaces owe generation
separation is derived from the `AuthorityTag` declarations rather than restated in a list one edit
away from disagreeing with them."""
