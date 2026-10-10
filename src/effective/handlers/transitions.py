"""Every way the durable handler reads its engine and its clock, labeled by what a retry sees.

The record is an effect like any other: the engine interprets each ctx call against a store that
can answer, miss, park the task, or fail. A transition is one such call at one handler site and
one of its outcomes, and its label says whether a retry of the attempt meets it the same way.

| label    | the outcome                                                         |
|----------|---------------------------------------------------------------------|
| `record` | a function of what this task saved, read back alike on a retry      |
| `fresh`  | a domain call ran                                                   |
| `world`  | a value the record does not hold: a store error, a clock, a value   |
|          | another writer put on the tape, an attempt number                   |

A `world` transition carries what keeps the futile-retry rule sound across it.

| kept by    | why a retry still replays the attempt, or the attempt is counted         |
|------------|--------------------------------------------------------------------------|
| `counted`  | the handler counts it at `counted_at` on every path from it to a raise   |
| `settled`  | it decides only what the handler saves to the tape before acting on it,  |
|            | and `counted_at` counts it where nothing is saved                        |
| `suspends` | it parks its branch or its task, and a park decides no failure: the      |
|            | attempt parks, or raises a sibling's error, which that sibling's own     |
|            | transitions account for                                                  |
| `monotone` | a clock found an instant passed, which stays passed on a retry           |
| `elides`   | it only skips reads whose answer the record already fixes                |

A refusal the domain raised is `record`: a refusal is what a retry gets again, and a governor whose
answer can change between attempts waits on an event instead. An attempt that raises would raise
again on a retry when it took no `fresh` transition and the handler counted none of its `world`
ones. The table is data: the handler keeps the rule where it counts.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class Outcome(StrEnum):
    """How one ctx call or clock read ended."""

    HIT = "hit"  # the store held the answer
    MISS = "miss"  # the store held none, and the call ran its thunk or wrote
    THUNK_RAISED = "thunk_raised"  # the thunk the call ran raised
    REFUSED = "refused"  # the thunk the call ran raised a refusal, alone or in a group
    PARKED = "parked"  # the engine's suspend signal
    STORE_RAISED = "store_raised"  # the store failed, a lost claim included
    DUE = "due"  # a clock found its instant passed
    UNDUE = "undue"  # a clock found its instant ahead
    ANSWERED = "answered"  # a value that is no checkpoint: an attempt number, a reading


@dataclass(frozen=True)
class World:
    kept_by: Literal["counted", "settled", "suspends", "monotone", "elides"]
    counted_at: str | None = None  # the function that counts it, for `counted` and `settled`


type Label = Literal["record", "fresh"] | World

RECORD: Label = "record"
FRESH: Label = "fresh"
STORE = World("counted", "DurableHandler._store_failed")
SUSPENDS = World("suspends")
MONOTONE = World("monotone")
WAKE_RACE = World("counted", "DurableHandler._join")
RACE_CLOCK = World("settled", "DurableHandler._run_race")

type Site = str
type Source = str


def _checkpoint(fresh: Label) -> dict[Outcome, Label]:
    return {
        Outcome.HIT: RECORD,
        Outcome.MISS: fresh,
        Outcome.THUNK_RAISED: fresh,
        Outcome.REFUSED: RECORD,
        Outcome.STORE_RAISED: STORE,
    }


def _peek(miss: Label) -> dict[Outcome, Label]:
    return {Outcome.HIT: RECORD, Outcome.MISS: miss, Outcome.STORE_RAISED: STORE}


def _wait(**answered: Label) -> dict[Outcome, Label]:
    """A call that answers as `answered` names (`HIT` an event, `DUE` a clock), or parks."""
    return {
        **{Outcome[name]: label for name, label in answered.items()},
        Outcome.PARKED: SUSPENDS,
        Outcome.STORE_RAISED: STORE,
    }


# fmt: off
_BY_CALL: dict[tuple[Site, Source], dict[Outcome, Label]] = {
    ("DurableHandler._respawn",       "step"):        _checkpoint(FRESH),
    ("DurableHandler._checkpointed",  "step"):        _checkpoint(FRESH),
    ("DurableHandler._race_leaf",     "peek_step"):   _peek(RECORD),
    ("DurableHandler._race_leaf",     "settle"):      _peek(RECORD),
    ("DurableHandler._guard",         "peek_step"):   _peek(RECORD),
    ("DurableHandler._guard",         "settle"):      _peek(RECORD),
    ("DurableHandler._run_race",      "peek_step"):   _peek(RECORD),
    ("DurableHandler._run_race",      "settle"):      _peek(RECORD),
    ("DurableHandler._settle_endings", "peek_step"):  _peek(RECORD),
    ("DurableHandler._settle_endings", "settle"):     _peek(RECORD),
    ("DurableHandler._branch_await",  "peek_event"):  _peek(SUSPENDS),
    ("DurableHandler._await",         "await_event"): _wait(HIT=RECORD),
    ("DurableHandler._await_absolute", "await_event"): _wait(HIT=RECORD),
    ("DurableHandler._await_bounded", "await_until"): _wait(HIT=RECORD, DUE=MONOTONE),
    ("DurableHandler._handle",        "sleep_until"): _wait(DUE=MONOTONE),
    ("DurableHandler._join",          "await_event"): _wait(HIT=WAKE_RACE),
    ("DurableHandler._join",          "sleep_until"): _wait(DUE=WAKE_RACE),
    ("DurableHandler._join",          "repark"):      {Outcome.ANSWERED: WAKE_RACE,
                                                       Outcome.PARKED: SUSPENDS,
                                                       Outcome.STORE_RAISED: STORE},
    ("_attempt_of",                   "attempt"):     {Outcome.ANSWERED: World("elides"),
                                                       Outcome.STORE_RAISED: STORE},
    ("DurableHandler._branch_sleep",  "clock"):       {Outcome.DUE: MONOTONE,
                                                       Outcome.UNDUE: SUSPENDS},
    ("Racing.expired",                "race_clock"):  {Outcome.ANSWERED: RACE_CLOCK},
    ("Racing.concurrently",           "race_clock"):  {Outcome.ANSWERED: RACE_CLOCK},
    ("Racing.stamp",                  "race_clock"):  {Outcome.ANSWERED: RACE_CLOCK},
}
# fmt: on

TRANSITIONS: Mapping[tuple[Site, Source, Outcome], Label] = {
    (site, source, outcome): label
    for (site, source), outcomes in _BY_CALL.items()
    for outcome, label in outcomes.items()
}
"""Each transition, keyed by the function that makes it (its qualified name in its module), what it
reads (a ctx member, `clock` or `race_clock`), and how it ended."""

FIXED: frozenset[str] = frozenset(
    (
        "prefix",
        "event_rename",
        "transplanted",
        "phase",
        "task_id",
        "_queue_name",
        "concurrent_safe",
        "speaks_sdk_text",
    )
)
"""Ctx attributes that are the same on every attempt at the same position in the walk."""
