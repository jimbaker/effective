"""`RunView`: one run, shaped once, rendered many ways.

The **shaping** layer between a reader and a renderer. The readers (`checkpoints`, `parked`,
`graphview`, `telemetry`) are reusable and every surface imports the same ones. What happens next
is the child join, the cost sum, the parked-to-state mapping and the frame split: each is a small
decision that belongs to no reader, and without this layer each pane makes it independently.

**That gap is where drift lives**, and the repo has instances of the shape:

| one side                                          | disagrees with                             |
|---------------------------------------------------|--------------------------------------------|
| `telemetry` mints `f"{prefix}.{i}.message.role"`  | a reader hard-codes `llm.input_messages.…` |
| `bridge_*._decode_state` sniffs `{result, usage}` | `metered_call` owns that shape             |

A one-line fix applied where a maintainer would naturally make it, in the pane in front of them,
leaves two renderers disagreeing silently, with no error on either side.

**Flat nodes with relations.** `Node.frames` is a path, so containment is a filter over flat nodes
rather than a spine that has to be chosen: a tree pane groups by it, a table pane ignores it, and a
drill-down into a fold is `members`. A materialized `ViewNode` remains reachable from here as an
addition; flat wins on build cost, on the markdown floor below, and because expand-on-demand is a
filter. Both shapes represent an interleaved tape (with `order` on every node a tree round-trips
exactly); a tree groups by containment, so showing interleaving is a pane's choice.

**A pure function of values a caller already read.** Nothing here opens a database, so the whole
shaping layer is testable with a list of strings, the way `graphview` is.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from effective.cards.spec import Action
from effective.domain import SPAWN_TOOL
from effective.graphview import Node, RunGraph, format_cost, to_text
from effective.keys.grammar import KeySyntaxError, parse, split_occurrence
from effective.parked import ParkedTask

ANSWER_CMD = "answer"
"""The `Action.cmd` a park's action carries — what `effective.parked.answer` does.

`Action` comes from `effective.cards.spec` rather than being minted here, and the reuse is the
point: `cmd`/`target`/`risk` is already the typed write-back vocabulary a Shiny board and an
MCP host consume, and a second `Action` in a second module would be two spellings of one rule —
the failure this module was written to prevent, arriving in the module that prevents it."""


def _spawned_child(key: str) -> str | None:
    """The child run id in a `tool:spawn,{child}` key, or `None` if this is not one.

    PARSED rather than matched on text. A `startswith` against the spawn key's spelling would
    work today and is the nominal-where-structural shape this repo keeps finding as a defect: it
    anchors at the head of the string, so it stops matching the moment the op sits inside a gather
    frame — which is exactly where a fan-out puts it. Walking terms sees the same op wherever it
    is nested, and the coordinate it reads is the one `fork` composed.

    Total over strings: a key the grammar refuses is not a spawn, which is the honest answer for
    a reader that must stay open over the banked corpus."""
    try:
        parsed = parse(split_occurrence(key)[0])
    except KeySyntaxError:
        return None
    for term in parsed.terms:
        if term.tag == "tool" and term.arity == 2 and term.coordinates[0].path == SPAWN_TOOL:
            return term.coordinates[1].path
    return None


@dataclass(frozen=True)
class RunView:
    """One run's shape, joined once: the tape, plus what the tape alone cannot say.

    `nodes` carries the projection a caller chose — unrolled, `project`ed or `fold_cycles`d — so
    this type is neutral about which quotient a pane wants. The other fields are joins the tape
    cannot answer on its own: subagents come from spawn keys, committed and changed files from the
    canonical ledger, and actions from the park reader.
    """

    run_id: str
    nodes: tuple[Node, ...] = ()
    children: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    changed: tuple[str, ...] | None = None
    """The committed files the run changed, or `None` unless every commit row says which."""
    actions: tuple[Action, ...] = ()
    results: Mapping[str, Any] = field(default_factory=dict, hash=False)
    """What each node RECORDED, keyed by node key — the fourth join, and the one a driver missed
    most.

    A run view that draws a step and cannot say what it returned sends its reader to the database.
    Measured: a de-essaying machine's gate ran, failed, and bounced the machine back a state, and
    the only way to learn *which* check failed was `sqlite3` against the store — while the answer
    sat in the checkpoint the tree was already drawing.

    **A left-outer join, and the misses are structural rather than accidental.** A node under
    `project` or `fold_cycles` is a REPRESENTATIVE of several recorded ops, so its key names a
    program position rather than one row; the pending node has no producer at all. Both are absent
    here rather than defaulted, so a pane can say "not recorded" instead of showing one branch's
    result under a folded label."""

    @property
    def cost(self) -> float | None:
        """Total dollars, or `None` when nothing under this run was measured.

        `None` rather than `0.0` for the reason `Node.cost` is not a `float`: summing unmeasured
        nodes into a zero reports "this run was free" for "nobody measured it". A run with one
        measured-and-free node totals `0.0` and is a different answer."""
        measured = [node.cost for node in self.nodes if node.cost is not None]
        return sum(measured) if measured else None


def run_view(
    graph: RunGraph,
    *,
    ledger: Iterable[Mapping[str, Any]] = (),
    parked: Iterable[ParkedTask] = (),
    recorded: Mapping[str, Any] | None = None,
) -> RunView:
    """Join a projected graph with the two records beside it.

    `ledger` is the canonical rows as stored — whole payload dicts, which is what both engines
    hold — and `parked` is what a park reader returned. Both default to empty, so a pane that only
    has the tape gets a `RunView` rather than having to invent the other halves.

    **The action carries the name the reader READ.** A wake registration carries coordinates no
    caller can reconstruct — the enclosing frames, and `Key.occurrence`'s `#N` for a second ask —
    so `Action.target` is `park.wake_event` verbatim and never a name composed here. That is the
    same rule `effective.parked.answer` enforces one layer down, and stating it in both places is
    deliberate: this is where a renderer would otherwise be tempted to prettify.
    """
    children = tuple(
        dict.fromkeys(
            child for node in graph.nodes if (child := _spawned_child(node.key)) is not None
        )
    )
    commits = [row for row in ledger if row.get("kind") == "machine-committed"]
    files = tuple(dict.fromkeys(name for row in commits for name in row.get("files") or ()))
    stated = [row["changed"] for row in commits if "changed" in row]
    changed = (
        tuple(dict.fromkeys(name for names in stated for name in names))
        if stated and len(stated) == len(commits)
        else None
    )
    actions = tuple(
        Action(
            cmd=ANSWER_CMD,
            label=f"answer {park.task_name}",
            target=park.wake_event,
            risk="medium",
        )
        for park in parked
    )
    keys = {node.key for node in graph.nodes}
    return RunView(
        run_id=graph.run_id,
        nodes=graph.nodes,
        children=children,
        files=files,
        changed=changed,
        actions=actions,
        # Narrowed to the nodes actually drawn, so a pane holds one run's results and not the
        # whole tape — and so a projected view does not carry rows nothing on screen can address.
        results={k: v for k, v in (recorded or {}).items() if k in keys},
    )


type FilesHeading = Literal["**Changed:**", "**Committed:**"]


def file_summary(view: RunView) -> tuple[FilesHeading, tuple[str, ...]] | None:
    """The files line a markdown renderer draws, as its heading and its names.

    | the commit rows                          | the line                       |
    |------------------------------------------|--------------------------------|
    | all state what changed                   | `**Changed:**`, those files    |
    | commit files, and one or more do not say | `**Committed:**`, every file   |
    | commit nothing, or do not exist          | `None`                         |
    """
    match view.changed:
        case None:
            return ("**Committed:**", view.files) if view.files else None
        case changed:
            return ("**Changed:**", changed) if view.files or changed else None


def files_line(view: RunView) -> str | None:
    """`file_summary` as the markdown line both renderers draw; `none` for an empty change."""
    if (summary := file_summary(view)) is None:
        return None
    heading, names = summary
    listed = ", ".join(f"`{name}`" for name in names) or "none"
    return f"{heading} {listed}"


def to_markdown(view: RunView) -> str:
    """`RunView` -> Markdown. The always-works floor, no host and no layout engine.

    Modelled on `cards.render_markdown`, which calls itself *"the always-works Tier-1 fallback"*,
    and it earns the name the same way: Textual renders markdown natively, so does Shiny, and a
    report or an MCP fallback takes it unchanged. One renderer, four targets.

    It is also the Bobby-Tables rule on the view axis — **flatten at the renderer, never at the
    boundary**. The tree arrives as a `RunGraph` and the actions as typed `Action`s, and both are
    still legible in the flattest target: `cmd` and `target` render beside the label rather than
    being collapsed into prose, so the write-back survives the flatten.

    What markdown cannot draw is the laid-out graph, and that degrades rather than disappearing:
    the tree is the same containment the SVG shows, in text.
    """
    lines = [f"## {view.run_id}"]
    if view.cost is not None:
        lines.append(f"\n**Cost:** {format_cost(view.cost)}  ·  **Ops:** {len(view.nodes)}")
    else:
        lines.append(f"\n**Ops:** {len(view.nodes)}  ·  cost unmeasured")

    lines.append("\n```\n" + to_text(RunGraph(view.run_id, view.nodes)) + "\n```")

    if view.children:
        lines.append("\n**Subagents:** " + ", ".join(f"`{child}`" for child in view.children))
    if (line := files_line(view)) is not None:
        lines.append("\n" + line)
    if view.actions:
        rendered = ", ".join(
            f"**{action.label}** (`{action.cmd}` → {action.target})" for action in view.actions
        )
        lines.append(f"\n**Actions:** {rendered}")
    return "\n".join(lines) + "\n"


__all__ = ["ANSWER_CMD", "RunView", "run_view", "to_markdown"]
