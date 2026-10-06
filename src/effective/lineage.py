"""Op-key sequences: canonicalize them, align them, diff them.

Two consumers want the same thing from opposite directions, so it is built once here.

**The fork** needs `compare(base, fork)`, a shared-prefix-aware diff over two lineages, because
without it a counterfactual produces a second run and no answer.

**The locally-in-distribution mapping** needs the `~_H` of Alex Zhang's "Language model harnesses
are compositional generalizers" (see `wiki/references.md`): two runs are equivalent when their
op-key sequences unify under scope renaming. Where his post-hoc analysis has to *approximate* that
with five string metrics over token trajectories, determinism plus injective `op_key` make it a
computable projection over recorded state: no model calls, no embeddings, no metric tuning.

The two share an **alignment core** and then want **different projections over different
bookkeepers**: a keys-only distance over *checkpoints* for the LID metric, a value/usage marginal
over the *ledger* lineage for the fork. This module owns the core and both projections.

The checkpoint reader (`Checkpoint`, `keys`, `ENGINE_INTERNAL`, `read_sqlite_task` and the rest)
lives in **`effective.checkpoints`**. This module is *analysis* over what that reader returns, so
the dependency runs `agent` -> `effective`, the allowed direction.

**Canonicalization is required for any comparison**: a fork pair diverges at `ledger;r-base:e1`
vs `ledger;r-fork-0:e1` at the *first ledger op*, because workflows interpolate the run id into
event ids, and `govern:{gate}:{run_id}:{pass}:{occ}:{op}`, `budget-grant:{run_id}:{n}` and
run-scoped await names all carry it too. Without scrubbing, two
structurally identical runs compare as different everywhere, and a fork's divergence index is
meaningless. Scrub tokens are supplied by the CALLER rather than guessed: only the caller knows
which strings are scopes rather than content, and a guesser that scrubbed a *meaningful* substring
would silently equate runs that genuinely differ.

Reads recorded state and imports no domain package.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, assert_never

from effective.counterfactual import FORK_SEALED_KIND, FORKED_KIND, fork_unscoped

# --- canonicalization: what counts as the same key across two runs -----------------------

RUN = "{run}"
"""The placeholder a scrubbed run id becomes. A single stable token, so a canonical key stays
greppable and two runs' keys are literally equal rather than merely 'equivalent'."""


def canonical(
    names: Iterable[str], *, scrub: Sequence[str] | Mapping[str, str] = ()
) -> tuple[str, ...]:
    """Rewrite run-scoped keys into their scope-free form.

    `scrub` names the tokens that are *scope* rather than content — run ids, task ids, message
    ids. A sequence maps every token to `{run}`; a mapping lets a caller distinguish scopes it
    wants to keep apart (`{"r-base": RUN, "trial-7": "{task}"}`). Longest token first, so a run id
    that is a prefix of another cannot half-scrub it."""
    # Both arms of the declared union are named: `Mapping()` and `Sequence()` are both usable
    # class patterns, so the two input forms say themselves.
    match scrub:
        case Mapping():
            table = dict(scrub)
        case Sequence():
            table = {t: RUN for t in scrub}
        case unreachable:
            assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    ordered = sorted(table, key=len, reverse=True)

    def rewrite(name: str) -> str:
        for token in ordered:
            name = name.replace(token, table[token])
        return name

    return tuple(rewrite(n) for n in names)


# --- the alignment core ------------------------------------------------------------------


@dataclass(frozen=True)
class Alignment:
    """How two op-key sequences relate — the shared core both projections read.

    `shared_prefix` is the count of leading keys that match, which is the number a FORK cares
    about: a well-formed counterfactual shares its base's prefix exactly up to the substitution,
    so `shared_prefix == at` is the fork's own correctness check. `first_divergence` is `None`
    when one sequence is a prefix of the other (a truncated or extended run, not a divergent one).
    """

    a: tuple[str, ...]
    b: tuple[str, ...]
    shared_prefix: int
    first_divergence: int | None

    @property
    def identical(self) -> bool:
        return self.a == self.b

    @property
    def diverged(self) -> bool:
        return self.first_divergence is not None


def align(a: Iterable[str], b: Iterable[str]) -> Alignment:
    """The alignment core. Pure, total, and cheap — no model calls, no embeddings."""
    left, right = tuple(a), tuple(b)
    shared = 0
    for x, y in zip(left, right, strict=False):
        if x != y:
            break
        shared += 1
    diverged = shared < min(len(left), len(right))
    return Alignment(left, right, shared, shared if diverged else None)


def compare(
    a: Iterable[str], b: Iterable[str], *, scrub: Sequence[str] | Mapping[str, str] = ()
) -> Alignment:
    """`align` over canonicalized keys — the form a fork or a cross-run comparison wants.

    Scrub the run ids (and any other scope tokens) or the comparison is meaningless: a fork pair
    diverges at its first ledger op purely because the event id embeds the run id."""
    return align(canonical(a, scrub=scrub), canonical(b, scrub=scrub))


def equivalent(
    a: Iterable[str], b: Iterable[str], *, scrub: Sequence[str] | Mapping[str, str] = ()
) -> bool:
    """`~_H`: do these two runs unify under scope renaming?

    Exact, not estimated — which is the whole claim. Zhang approximates this relation with five
    string metrics over token trajectories because his harness cannot observe it; here it falls
    out of the injective-`op_key` invariant."""
    return compare(a, b, scrub=scrub).identical


# --- projection 1: the keys distance (the LID metric) ------------------------------------


def key_distance(alignment: Alignment) -> int:
    """Edit distance over op keys — how far apart two runs are *structurally*.

    Zhang's Figure 8 over canonical keys instead of estimated token distance. `SequenceMatcher`
    gives the matching blocks; the distance is the number of keys outside them. Exact, symmetric,
    and 0 exactly when the sequences are identical (so `key_distance == 0` iff `equivalent`)."""
    matcher = SequenceMatcher(a=alignment.a, b=alignment.b, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return (len(alignment.a) - matched) + (len(alignment.b) - matched)


# --- projection 2: the value/usage marginal over the ledger lineage ----------------------
#
# `key_distance` (projection 1) reads the CHECKPOINT bookkeeper for the LID metric. The fork's own
# deliverable reads the other bookkeeper — the ledger lineage — and asks a different question: what
# COMMITTED differently ("approve -> commits $42; reject -> ledgers a rejection").


@dataclass(frozen=True)
class Marginal:
    """The fork's deliverable: two lineages, identical up to the fork point, divergent after.

    `shared_prefix` counts the leading events both lineages committed alike (a well-formed
    counterfactual shares its base's prefix). `base_tail` / `fork_tail` are the
    divergent remainders — the actual marginal: what the base committed vs what the fork would
    have. A value or cost summary layers on top of the tails; this keeps the diff structural and
    lets the caller decide what to total."""

    shared_prefix: int
    base_tail: tuple[Mapping[str, Any], ...]
    fork_tail: tuple[Mapping[str, Any], ...]

    @property
    def diverged(self) -> bool:
        return bool(self.base_tail or self.fork_tail)


def fork_marginal(
    base: Sequence[Mapping[str, Any]],
    fork: Sequence[Mapping[str, Any]],
    *,
    child_run_id: str,
    forked_at_event: str,
    key: Callable[[Mapping[str, Any]], str] = lambda row: str(row.get("event_id", "")),
) -> Marginal:
    """The fork's marginal by **ADDRESS**: a caller slices both lineages at `forked_at_event`
    directly.

    `marginal` is the DISCOVERY form: it aligns two arbitrary runs by canonical identity and finds
    where they part. That does not work on a real `run_fork` lineage, and the reason is structural
    rather than a bug in either function:

    1. the fork's **genesis** row sits at index 0 with no counterpart in the base, so
       alignment-by-identity diverges at the first row;
    2. the shared prefix's ledger rows are **seeded**, so `_record_ledger` never runs for them and
       they are ABSENT from the fork lineage entirely — "store the divergence, not a copy of
       history", which is correct and which discovery-alignment was never taught about.

    Net effect before this: `shared_prefix == 0` and the whole base read as divergent. The fork's
    own deliverable did not work end-to-end, while a hand-built fixture in the tests pinned a shape
    `run_fork` cannot emit.

    Here the fork point is KNOWN, so nothing is discovered. The base's rows through
    `forked_at_event` are shared **by construction** (they were seeded, are referenced by the
    genesis, and were never re-committed), the fork's bookkeeping rows (`forked` genesis,
    `fork_sealed` seal) are dropped, and what remains on each side is the tail. Raises if
    `forked_at_event` is not in the base — that means the genesis names an address the base lineage
    does not contain — a provenance error worth surfacing rather than silently aligning at 0.
    """
    base_ids = [key(row) for row in base]
    try:
        boundary = base_ids.index(forked_at_event) + 1
    except ValueError:
        raise ValueError(
            f"forked_at_event {forked_at_event!r} is not in the base lineage "
            f"({len(base)} rows: {base_ids[:6]}{'…' if len(base_ids) > 6 else ''}) — the genesis "
            f"names an address this base does not contain, so the two are not a fork pair."
        ) from None
    shared_ids = set(base_ids[:boundary])
    fork_tail = tuple(
        row
        for row in fork
        # drop the fork's own bookkeeping (genesis, seal) — they have no base counterpart …
        if str(row.get("kind")) not in (FORKED_KIND, FORK_SEALED_KIND)
        # … and drop any row that re-commits a SHARED-prefix event. Exact membership, not a prefix
        # test: `extracted:m11` must not be swallowed by `extracted:m1`. A well-formed fork emits
        # none of these (the prefix is seeded, so `_record_ledger` never runs); one appearing
        # means
        # the fork copied history, which S-1's boundary arm now refuses at the source.
        and fork_unscoped(key(row), child_run_id) not in shared_ids
    )
    return Marginal(
        shared_prefix=boundary,
        base_tail=tuple(base[boundary:]),
        fork_tail=fork_tail,
    )


def marginal(
    base: Sequence[Mapping[str, Any]],
    fork: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], str] = lambda row: str(row.get("event_id", "")),
    scrub: Sequence[str] | Mapping[str, str] = (),
) -> Marginal:
    """Diff two ledger lineages past their shared prefix — projection 2, the fork's answer.

    The lineages are aligned by their **canonical event identity** (`key`, run-id `scrub`bed), so a
    fork's run-scoped event ids line up with the base's rather than diverging at the first: the
    same run-id scrubbing projection 1 needs. The tails are the ORIGINAL rows (unscrubbed),
    because the marginal is about their *content*: the committed values and their diff.

    **Alignment is by IDENTITY, not payload** — a contract worth stating. A decision fork
    substitutes a value AT an event that keeps its id (a `reviewed:*` event, approve in one run,
    reject in the other), so that event ALIGNS and the divergence surfaces at its *consequence*
    (base commits, fork ledgers a rejection), not at the substitution. This is the
    DISCOVERY form, for two arbitrary runs. When the fork point is already recorded (the genesis'
    `forked_at_event`) use **`fork_marginal`** instead — the ADDRESS form, which slices at
    the known boundary and knows about the genesis/seal rows and the seeded prefix. Discovery
    alignment cannot work on a real `run_fork` lineage; see `fork_marginal`'s docstring."""
    a = align(
        canonical((key(r) for r in base), scrub=scrub),
        canonical((key(r) for r in fork), scrub=scrub),
    )
    return Marginal(
        shared_prefix=a.shared_prefix,
        base_tail=tuple(base[a.shared_prefix :]),
        fork_tail=tuple(fork[a.shared_prefix :]),
    )


__all__ = [
    "RUN",
    "Alignment",
    "Marginal",
    "align",
    "canonical",
    "compare",
    "equivalent",
    "fork_marginal",
    "key_distance",
    "marginal",
]
