"""The banked agent TAPE — what makes one current, and how a machine says so.

A tape test walks a tape a real model produced and asks the transcript property of
it (`effective.tape.transcript_violations`). That fixture is only as good as the tape underneath
it, and the tape is gitignored — so it is whole where a live run happened and absent everywhere
else. This module is the tape's own description, read by three callers that have to agree:

- a tape writer and gate, which live outside this repository;
- the tape test, which skips when the local tape cannot support the walk;
- a human bumping `CURRENT_ERA` when the loop or the key grammar moves, which is the whole
  maintenance procedure.

**Why banking an INPUT is safe here, where banking an OUTPUT would not be.** The walk asserts a
PROPERTY every valid trajectory must satisfy, not equality with a blessed artifact. A freshly
banked tape that violates the property still fails. Compare an approval test, where re-blessing
silently redefines what correct means — the failure mode where a bug is captured as the new golden
output and nobody notices. So re-banking is a question about coverage freshness (does this tape
still exercise interesting paths?) rather than about correctness drift, which is a much weaker
obligation and is why the cadence can be casual.

**The stamp is the git hash**, which is better than inventing one: exact, free,
and already the thing a bisect speaks. It is not decoration — the span sidecar this design
rejected held 17 runs from two arms in one file, and without a code stamp a violation could not be
attributed to a historical bug rather than a live one.

**THE STAMP MUST CARRY MORE THAN THE TAPE DOES, and that is measured rather than assumed.**
`engine.run` spawns with `{"run_id", "seed", "goal"}` only; `max_iters` and `interrupt` are
closure state, and `command` defaults inside the live workflow. A replay that guessed
them would diverge from the tape it is replaying, so they ride on the stamp.

**A VACUOUS TAPE IS NOT A TAPE.** A give-up run — the model answers at turn 0 without acting —
has no transition for the property to check, so the walk returns no violations because it never
looked. Measured on the tape sitting in `build/` at the time this was written: one witnessed turn,
zero violations, entirely clean and entirely uninformative. So `status` demands turns AND actions,
and the banker verifies by replaying what it just wrote rather than trusting that a live run is
interesting.
"""

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from effective.tape import TapeVerdict

REPO = Path(__file__).resolve().parents[2]
BENCHES = REPO / "benches"
TAPES = BENCHES / "tapes"

STAMP_NAME = "STAMP.json"
"""Written at the ROOT of an era tree by the tape writer. Its presence is what distinguishes a
tree this repo banked from a directory someone happened to create with the right name."""

TAPE_NAME = "tape.db"
"""**NOT `task.db`, and the difference is load-bearing.** `agent.bank.all_stores` is
`rglob("task.db")` over the local corpus directory, so a tape there named that way would be swept
silently into the CONTRAST corpus's store census and shift a count the corpus gate checks. The two
corpora share a gitignored root and must not share a filename. This is the enforcer's-domain rule
pointed at our own sweep: its grammar is its domain, and it would happily claim a file that is not
its business."""

SIDECAR_NAME = "tape.spans.jsonl"
"""The span sidecar banked BESIDE the tape, and it is not decoration.

The tape cannot witness itself: replay reads the checkpoints, so it agrees with them by
construction. The premise that makes the whole walk meaningful — *the prompt is a function of the
tape* — therefore cannot be checked against the tape. The sidecar is an INDEPENDENT record of the
same run, written at a different seam while it was live, so `tape.wire_divergences` can ask
whether replay rebuilds what was actually sent. It also carries the cost and duration a
`(key, state)` checkpoint structurally cannot hold.

`engine.run` writes it next to the db by default; banking without it discarded the only evidence
that could retire the premise, which is what happened on the first bank."""

CURRENT_ERA = "tape-4"
"""Tapes written by the CURRENT loop and key grammar live under this name. Eras are told apart by
PATH rather than by sniffing the keys, for the same reason `agent.bank` gives: the point of a
fixture is to state what it expects, and a classifier that read the keys would agree with any tape
by construction.

Bumping this constant IS the migration procedure — the old tree becomes legacy where it stands and
the gate banks a new one beside it. Unlike the contrast corpus there is no cross-version claim
here: an old tape is not replayable against a changed loop, and the walk needs replay to
reconstruct the prompts. So a stale era is simply dropped, not kept as a compat fixture."""

MIN_TURNS = 2
"""ReAct turns a tape must carry for the property to have anything to say. Two is the true
minimum: the invariant is about turn `i` seeing turn `i-1`, so one turn has no predecessor and
cannot witness anything."""

MIN_ACTIONS = 1
"""Tool calls a tape must carry. A trajectory that never acted has no action for a later turn to
be shown, so the action half of the property is vacuous even when the turn count is fine — which
is exactly the shape of a give-up run."""


def bank_refusal(verdict: TapeVerdict, actions: int) -> str | None:
    """Why a freshly-produced tape must not be banked, or `None` to bank it.

    **Two questions, and the banker only ever asked the first.** The census — enough turns, at
    least one action — asks *is this worth walking?*. The verdict's violations ask *did it pass?*.
    `bank()` computed the whole verdict and read `.witnessed` off it, discarding the violations, so
    a tape that FAILED the property was stamped and copied over the good one.

    That is not a small gap, because it is the banker's own claim about re-banking. The tape is
    described everywhere as an INPUT rather than a golden output, safe to re-bank because a fresh
    tape that violates the property still fails. It only fails if something checks, and this is
    that check.

    A pure function taking the verdict rather than a step inside `bank()`, because `bank()` needs a
    live model, a network and a container — anything spelled inside it ships unpinned, which is the
    same argument `skillsbench.measurements_record` was extracted on."""
    if verdict.witnessed < MIN_TURNS or actions < MIN_ACTIONS:
        return (
            f"this tape is vacuous for the walk ({verdict.witnessed} turns, {actions} actions; "
            f"need {MIN_TURNS}/{MIN_ACTIONS}). The model gave up rather than working the problem, "
            f"which happens. Re-run to try again; nothing was displaced."
        )
    if verdict.violations:
        detail = "\n".join(f"  {violation}" for violation in verdict.violations)
        return (
            f"the fresh tape VIOLATES the transcript property ({len(verdict.violations)} "
            f"violation(s) over {verdict.witnessed} witnessed turn(s)):\n{detail}\n\n"
            f"That is a defect in the loop, not in the run — banking it would install a fixture "
            f"that agrees with the bug. Nothing was displaced."
        )
    return None


BANK_GOAL = "fix the bug in mod.py so the test passes"
BANK_MODE = "whole-file"
BANK_MODEL = "gpt-5.6-luna"
BANK_WIRE = "responses"
"""The engine's own tuned defaults, and chosen for RELIABILITY rather than price. A bank that
produces a give-up run has produced nothing — the vacuity check rejects it and the half-cent is
spent for no tape. Measured in the telemetry arc: luna on `/v1/responses` finished 6/6 cleanly
where chat completions managed 3/6, and the nano tape left in `build/` is a give-up run that
censuses at one turn. Paying slightly more for a trajectory that actually acts is the right trade
for a gate that runs rarely."""
BANK_MAX_ITERS = 6
BANK_BUDGET_USD = 0.05
"""A ceiling, not an estimate: a live code-agent run measured ~$0.002. The gap is deliberate — the
ceiling exists to stop a runaway, not to predict the bill."""


@dataclass(frozen=True)
class Stamp:
    """What a bank records about itself, so a later run can tell whether it is still the one asked
    for. Every field is something a mismatch should re-bank over — plus the four replay inputs the
    tape itself does not carry."""

    era: str
    revision: str
    banked_at: str
    model: str
    wire: str
    mode: str
    max_iters: int
    command: str
    goal: str
    run_id: str
    turns: int
    actions: int
    spans: int


@dataclass(frozen=True)
class TapeStatus:
    """The gate's whole finding. `problems` empty means the current era is usable."""

    era: str
    root: Path
    stamp: Stamp | None
    tapes: int
    problems: tuple[str, ...]

    @property
    def usable(self) -> bool:
        return not self.problems


def era_root(era: str = CURRENT_ERA, tapes: Path = TAPES) -> Path:
    return tapes / era


def tape_path(era: str = CURRENT_ERA, tapes: Path = TAPES) -> Path:
    return era_root(era, tapes) / TAPE_NAME


def sidecar_path(era: str = CURRENT_ERA, tapes: Path = TAPES) -> Path:
    return era_root(era, tapes) / SIDECAR_NAME


def all_tapes(tapes: Path = TAPES) -> list[Path]:
    return sorted(tapes.rglob(TAPE_NAME)) if tapes.exists() else []


def read_stamp(root: Path) -> Stamp | None:
    """Absent or unreadable both read as "no stamp" — the caller's repair is the same either way,
    and a half-written stamp is exactly the case where refusing to guess is cheapest."""
    path = root / STAMP_NAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except OSError, json.JSONDecodeError:
        return None
    known = {f.name for f in fields(Stamp)}
    if not known <= data.keys():
        return None
    return Stamp(**{k: v for k, v in data.items() if k in known})


def write_stamp(root: Path, stamp: Stamp) -> Path:
    path = root / STAMP_NAME
    path.write_text(json.dumps(asdict(stamp), indent=1) + "\n")
    return path


def status(era: str = CURRENT_ERA, tapes: Path = TAPES) -> TapeStatus:
    """Decide whether this machine holds a tape the fixture can be pointed at.

    THE DOMAIN, written down because a gate's grammar is its domain and the sentence describing it
    will otherwise quietly claim more. CHECKED: the era tree exists, holds a `tape.db`, carries a
    stamp naming this era, and that stamp claims enough turns and actions for the property to be
    non-vacuous. NOT CHECKED: that the tape replays, that any key parses, or that the property
    holds — those are the tape test's job, and running it is what proves
    the tape. This gate exists only to keep it from skipping (or passing) vacuously.

    Note what the turn/action check is and is not. It reads the STAMP, which the banker wrote
    after replaying what it had just produced — so it is a recorded measurement, not a re-derived
    one. The fixture re-derives it independently; if the two ever disagree, the fixture is right
    and the tape wants re-banking."""
    root = era_root(era, tapes)
    stamp = read_stamp(root)
    found = [p for p in all_tapes(tapes) if era in p.parts]

    problems: list[str] = []
    if not root.exists():
        problems.append(f"no {era} tree at {root}")
    elif not (root / TAPE_NAME).exists():
        problems.append(f"no {TAPE_NAME} in {root}")
    elif not (root / SIDECAR_NAME).exists():
        # A tape without its sidecar is walkable but cannot be RECONCILED — the wire-fidelity and
        # unmeasured-step checks both need the independent record. Repairable, so it is a problem
        # the gate re-banks over rather than a lane that quietly narrows.
        problems.append(f"no {SIDECAR_NAME} in {root} — banked before the sidecar was kept")
    elif stamp is None:
        problems.append(
            f"{root / STAMP_NAME} missing or unreadable — not banked by the tape writer"
        )
    elif stamp.era != era:
        problems.append(f"stamp says era {stamp.era!r}, current era is {era!r}")
    elif stamp.turns < MIN_TURNS or stamp.actions < MIN_ACTIONS:
        problems.append(
            f"tape is vacuous for the walk: {stamp.turns} turn(s) and {stamp.actions} action(s), "
            f"below the {MIN_TURNS}/{MIN_ACTIONS} the property needs to witness anything"
        )

    return TapeStatus(era=era, root=root, stamp=stamp, tapes=len(found), problems=tuple(problems))
