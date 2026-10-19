"""What a banked agent TAPE must hold before it is worth keeping.

The transcript walk takes a tape a real model produced and asks the transcript property of it
(`effective.tape.transcript_violations`). The walk asserts a PROPERTY every valid trajectory must
satisfy, so a freshly banked tape that violates it still fails, and re-banking is a question about
coverage rather than about correctness drift.

A give-up run, where the model answers at turn 0 without acting, has no transition for the
property to check: the walk returns no violations because it never looked. So a banker asks two
questions of a fresh tape before keeping it, and `bank_refusal` asks both.
"""

from effective.tape import TapeVerdict

MIN_TURNS = 2
"""ReAct turns a tape must carry for the property to have anything to say. Two is the true
minimum: the invariant is about turn `i` seeing turn `i-1`, so one turn has no predecessor and
cannot witness anything."""

MIN_ACTIONS = 1
"""Tool calls a tape must carry. A trajectory that never acted has no action for a later turn to
be shown, so the action half of the property is vacuous even when the turn count is fine: the
shape of a give-up run."""


def bank_refusal(verdict: TapeVerdict, actions: int) -> str | None:
    """Why a freshly produced tape must not be banked, or `None` to bank it.

    | question                         | refused when                                  |
    |----------------------------------|-----------------------------------------------|
    | is the tape worth walking?       | fewer than `MIN_TURNS` turns or `MIN_ACTIONS` |
    | does the tape pass the property? | the verdict carries a violation               |

    A pure function of the verdict, so it is pinned without the live model, network and container
    a bank needs."""
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
