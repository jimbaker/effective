"""The coding-side assembly: what this embodiment supplies to the generic machine.

`effective.machine` walks whatever states it is handed, and this is where the coding machine's
own answers to its parameters live: `coding_read`, the fold from a ReAct loop's recorded results
to the `Evidence` a judge reads and the postamble commits, and `coding_bind`, the workspace each
tool call is handed.

Which states reach the append-only ledger is NOT declared here. It is `StateSpec.canonical`, per
state, read back off the tape by `canonical_violations` — a declaration with something that can
disagree with it, which a set literal beside the call sites was not."""

from typing import Any

from effective.machine.evidence import CommandRun
from effective.machine.spec import Ctx, Evidence
from effective.react import ToolLog, Trajectory


def coding_bind(ctx: Ctx[Any], log: ToolLog) -> dict[str, Any]:
    """The `agent_worker` binding for the table this package ships: the tree a tool call acts on,
    which is this visit's last recorded tree, else the tree the visit started from.

    Every tree it hands out came from the `Ctx` or from a recorded op result, so a worker resumed
    on a fresh process binds the same tree the crashed one did, and a domain serving the table
    keeps nothing between calls."""
    changed = log.last(dict[str, str])
    return {"tree": dict(ctx.tree if changed is None else changed)}


def coding_read(trajectory: Trajectory, log: ToolLog) -> Evidence:
    """The `agent_worker` reader for the table this package ships: typed results to `Evidence`.

    Two folds, and both are by the tool's DECLARED result type rather than by sniffing a value:
    the last tree-valued result becomes `Evidence.tree`, the last `CommandRun` becomes
    `Evidence.measured`. Declaration over sniffing matters because `read_file` and `check` both
    return `str`, and a fold that guessed from runtime shape would pick whichever ran last.

    **This is what carries a canonical state's work to the commit.** The trampoline advances its
    workspace only from what a worker returns (`Evidence.tree`), because a workflow may only
    learn about a handler-side edit through a recorded op result — read a shared workspace instead
    and a replay, where no tool runs, re-derives a different artifact id. Every entry in the log
    came from a `step` result, so this fold has that property by construction.

    `None` where nothing of that kind ran, which is the honest answer and the one the mechanical
    judges are built to complain about by name: `None` says "not measured", where an empty
    `CommandRun` would say "measured, and clean"."""
    return Evidence(
        summary=trajectory.answer,
        detail=f"{len(trajectory.steps)} steps, stopped: {trajectory.stop_reason}"
        + (f"; tools: {', '.join(log.names)}" if log.names else ""),
        measured=log.last(CommandRun),
        tree=log.last(dict[str, str]),
    )


__all__ = ["coding_bind", "coding_read"]
# NOT re-exported here: `CODING_TOOLS`, which lives with the runners that serve it. Re-exporting
# it would make this module import `effective.coding.runners`, and that is the one edge the
# package's own split forbids — the yielding half must stay importable without `gate`, `runners`
# or `edits`, which carry jedi, ast-grep and a ruff config resolved from the filesystem.
#
# `lint.SEAM_FORBIDDEN` names the PROTECTED half rather than listing the yielding half, so the
# yielding half is whatever is not protected, including a module nobody has written yet.
