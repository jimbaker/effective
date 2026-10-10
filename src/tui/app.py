"""The Textual app: an output pane, a status line and an input pane over a durable run.

Active, not a display mode of a shell — the same shape a coding agent's terminal has, because that
is what a run view is for: you watch a run and you answer it. What makes it safe to be active is
that the two writes it can make are the two the substrate already sanctioned. Answering a park
goes through `effective.parked.answer`, which settles the name the reader READ; replaying for the
decisions pane goes through `ViewingCtx`, which cannot write at all.

**Reads run on thread workers.** Every read here is blocking I/O or CPU — SQLite, then a pure
projection — and Textual's own guidance is that a blocking API belongs on a thread worker rather
than the event loop. Each refresh is numbered on the UI thread, and a picture from an earlier
number is dropped, so a newer read supersedes an in-flight one; `exclusive` only cancels a worker
whose result nothing awaits.

**Nothing here shapes data.** The tree comes from `graphview.to_text`, the decisions pane from
`runview.to_markdown`, and both are values a Shiny board or an MCP host takes unchanged. If a
future pane needs a shape this package has to compute, that computation belongs in `effective`.
"""

import json
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar
from uuid import UUID

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Markdown, Static, Tree
from textual.widgets.tree import TreeNode

from effective import runread
from effective.engines.sqlite import SqliteApp
from effective.graphview import (
    Node,
    RunGraph,
    fold_cycles,
    format_cost,
    identity,
    project,
    to_text,
)
from effective.parked import ParkedTask, answer
from effective.runs import RunStatus
from effective.runview import RunView, files_line, to_markdown

PROJECTIONS = ("unrolled", "project", "fold")
"""The three views of one tape, named as the substrate names them.

`project` discovers its axes from the tape and `fold_cycles` reads a declared list, and both
survive on purpose: the discovered one is sharper and moves with its data, which is what a
dashboard wants; the declared one is stable across runs, which is what comparing two runs needs.
A UI that offered only one would be picking for the reader."""


def _project(graph: RunGraph, mode: str) -> RunGraph:
    """Apply the named projection. Total over the tuple above; anything else is a caller bug."""
    match mode:
        case "project":
            return project(graph)
        case "fold":
            return fold_cycles(graph)
        case _:
            return graph


class RunViewApp(App[None]):
    """One store, one run at a time, four panes.

    `db_path` is a SQLite store. `checkpoints` and `parked` ship both engine halves, so widening
    this to a DSN is a constructor change for those two. It is not done here because `runs()` is
    SQLite-only, and its Absurd half is open work.
    """

    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #tree { width: 40%; border-right: solid $accent; }
    #right { width: 1fr; }
    #detail { height: 1fr; }
    #parks { height: auto; max-height: 40%; border-top: solid $accent; }
    #status { height: auto; padding: 0 1; color: $text-muted; }
    """

    BINDINGS: ClassVar[list[Binding]] = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("p", "cycle_projection", "Projection"),
        Binding("m", "toggle_markdown", "Markdown"),
        # `:` opens the command line, vim's convention, and it exists because the two obvious
        # arrangements are each broken in one direction. Leave the `Input` focused and every
        # single-key binding is swallowed as text; focus the tree instead and the command line is
        # unreachable — `Tab` lands on the parks table, so typing `open 2` fires `p` and cycles
        # the projection instead. Both were measured by driving the app under tmux. One key to
        # enter, `escape` to leave, and neither mode eats the other's keys.
        Binding(":", "focus_input", "Command"),
    ]

    def __init__(self, db_path: str | Path, task_id: UUID | None = None) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.task_id = task_id
        self.projection = PROJECTIONS[0]
        self.view: RunView | None = None
        self.parks: tuple[ParkedTask, ...] = ()
        self.status: RunStatus | None = None
        self.selected: Node | None = None
        self._as_markdown = False
        self._refresh_seq = 0

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            yield Tree("run", id="tree")
            with Vertical(id="right"):
                yield Markdown("", id="detail")
                yield DataTable(id="parks")
        yield Static("", id="status")
        yield Input(placeholder=": to type  ·  answer <json>  ·  open <n>  ·  help", id="input")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#parks", DataTable)
        table.add_columns("parked task", "waiting on")
        if self.task_id is None and (found := runread.runs(self.db_path)):
            self.task_id = found[0].task_id
        # FOCUS THE TREE, not the input. Textual focuses the first focusable widget on mount, and
        # with an `Input` in the layout that meant every single-key binding was swallowed as text:
        # `p` typed a "p" instead of cycling the projection. Found by driving the app rather than
        # by reading it — the bindings are declared correctly and the Footer even advertised them.
        # Tab reaches the command line when a reader wants to type.
        self.query_one("#tree", Tree).focus()
        self.refresh_view()

    # --- the read path -------------------------------------------------------------------------

    def refresh_view(self) -> None:
        """Read the selected run on a thread and redraw it. Refused off the UI thread, since the
        refresh number is read and written here.

        Each call takes the next refresh number, and `_render` drops a picture carrying an earlier
        one, so a pane shows the run it has selected. `exclusive` cancels an earlier read's
        worker, which stops nothing a thread is already doing; the number is what stops its
        picture."""
        if threading.get_ident() != self._thread_id:
            raise RuntimeError("refresh_view runs on the UI thread; use call_from_thread")
        self._refresh_seq += 1
        if self.task_id is not None:
            self._read(self._refresh_seq, self.task_id, self.projection)

    @work(exclusive=True, thread=True, group="read")
    def _read(self, seq: int, task_id: UUID, projection: str) -> None:
        """One read, from arguments rather than from fields the UI thread writes."""
        view = runread.view(self.db_path, task_id, project=lambda g: _project(g, projection))
        parks = tuple(runread.parks(self.db_path, task_id))
        status = runread.status(self.db_path, task_id)
        self._to_ui(self._render, seq, view, parks, status)

    def _to_ui(self, callback: Callable[..., object], *args: Any, **kwargs: Any) -> None:
        """Run `callback` on the UI thread from a worker, or drop it once the app is stopping: a
        write already made stays made, and no pane is left to show anything.

        Checked again on the UI thread, because shutdown stops the app before it closes the
        screens, and a delivery queued in between would draw into widgets being removed."""

        def deliver() -> None:
            if self.is_running:
                callback(*args, **kwargs)

        try:
            self.call_from_thread(deliver)
        except RuntimeError:
            if self.is_running:
                raise

    def _render(
        self,
        seq: int,
        view: RunView,
        parks: tuple[ParkedTask, ...],
        status: RunStatus | None,
    ) -> None:
        if seq != self._refresh_seq:
            return
        self.view, self.parks, self.status = view, parks, status
        self._draw_tree(view)
        self._draw_detail(view)
        table = self.query_one("#parks", DataTable)
        table.clear()
        for park in parks:
            table.add_row(park.task_name, park.wake_event)
        cost = "cost unmeasured" if view.cost is None else format_cost(view.cost)
        # The run's STATE leads, because it decides whether the rest of the line is a snapshot or
        # a final picture — and because without it a failed run reads as an empty one.
        #
        # NOT `[{state}]`. A `Static.update` string is parsed as Textual CONTENT MARKUP, so square
        # brackets are a style tag: `[failed]` rendered as a *span* named "failed" wrapping the
        # rest of the line, and the word itself disappeared from the visible text. The fix that
        # made a failed run legible was therefore invisible, in a way `assert "[failed]" in
        # str(render())` reports as a plain missing string. Measured in the pilot.
        state = "?" if status is None else status.state.value
        self.query_one("#status", Static).update(
            f"{state}  ·  {view.run_id}  ·  {len(view.nodes)} ops  ·  {cost}  ·  "
            f"{self.projection}  ·  {len(parks)} parked"
        )

    def _draw_tree(self, view: RunView) -> None:
        """Rebuild the tree from `Node.frames`, each level in run order.

        The node's `Node` rides in `data`, which is what makes selecting one a lookup rather than
        a re-parse of its label — the reason `Tree` was the right widget rather than a `Log`.

        Grouping by frame is what a tree IS, so this cannot show interleaving: two gather branches
        that alternated on the tape draw as two blocks. A gather's branches interleave when its CTX
        advertises `concurrent_safe`, which is a property of the wiring rather than of the engine,
        so that is a real picture to be missing and the status line's op count plus the ORDER of
        `view.nodes` is where a reader recovers it.
        """
        tree = self.query_one("#tree", Tree)
        tree.reset(view.run_id)
        levels: dict[tuple[str, ...], TreeNode[Any]] = {(): tree.root}
        for node in sorted(view.nodes, key=lambda n: n.order):
            parent = tree.root
            for depth in range(len(node.frames)):
                path = node.frames[: depth + 1]
                if path not in levels:
                    levels[path] = parent.add(node.frames[depth], expand=True)
                parent = levels[path]
            parent.add_leaf(_label(node), data=node)
        tree.root.expand()

    def _draw_detail(self, view: RunView) -> None:
        pane = self.query_one("#detail", Markdown)
        pane.update(to_markdown(view) if self._as_markdown else _summary(view, self.selected))

    def on_tree_node_selected(self, event: Tree.NodeSelected[Any]) -> None:
        """Selecting a leaf shows what that op RECORDED.

        The reason `Tree` was the right widget rather than a `Log`: the `Node` rides in `data`, so
        this is a lookup instead of a re-parse of a label. It is also the drop-out that cost the
        most on the first real drive — a gate ran, failed, bounced the machine back a state, and
        the only way to learn which check failed was to open the database, while the answer sat in
        the checkpoint the tree was already drawing."""
        self.selected = event.node.data
        if self.view is not None:
            self._draw_detail(self.view)

    # --- actions -------------------------------------------------------------------------------

    def action_focus_input(self) -> None:
        """Enter command mode. `escape` (below) is the way back."""
        self.query_one("#input", Input).focus()

    def on_key(self, event) -> None:
        """`escape` leaves command mode, so a reader is never stuck typing.

        Handled here rather than as a `Binding` because a focused `Input` consumes key events
        before the app's bindings see them — which is the whole reason `:` exists — so the return
        journey has to be caught at the same level the outbound one is."""
        if event.key == "escape" and self.focused is self.query_one("#input", Input):
            self.query_one("#tree", Tree).focus()
            event.stop()

    def action_refresh(self) -> None:
        self.refresh_view()

    def action_cycle_projection(self) -> None:
        index = PROJECTIONS.index(self.projection)
        self.projection = PROJECTIONS[(index + 1) % len(PROJECTIONS)]
        self.refresh_view()

    def action_toggle_markdown(self) -> None:
        self._as_markdown = not self._as_markdown
        if self.view is not None:
            self._draw_detail(self.view)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """The command line. Deliberately small, and `answer` is the only write.

        Answering settles the park's OWN name — `self.parks[…].wake_event`, never a name composed
        here — because a wake registration carries enclosing frames and an occurrence suffix that
        no caller can reconstruct. It marks the task claimable and does NOT resume it: a drain
        still has to run, and this app is not one.
        """
        # Back to navigation after every command, the way `:` behaves in vim. Staying in the
        # input is the other reasonable choice and it is worse here: the mode becomes STICKY, so
        # a second `:` is typed as text rather than re-entering — measured under tmux, where
        # `:answer {...}` after an earlier command submitted the colon and was correctly rejected
        # as an unknown command. One rule ("`:` always enters, a command always leaves") beats a
        # mode a reader has to track.
        self.query_one("#tree", Tree).focus()
        accepted = self._command(event.value.strip().split(maxsplit=1))
        # CLEARED ONLY WHEN THE COMMAND ACTED. Clearing first cost three answers on the second
        # drive: a driver answering faster than the drain advances finds `parks` momentarily
        # empty, the command is refused, and the payload it refused is already gone — so the only
        # trace is a toast that scrolls away. Retyping a long JSON answer is the most expensive
        # thing this surface can ask for, and it was asking for it on its own rejections.
        if accepted:
            event.input.value = ""

    def _command(self, parts: list[str]) -> bool:
        """Run one command. `True` when it acted, which is what decides whether the input clears.

        A rejection is not a failure of the driver's typing — `nothing is parked` usually means
        *not yet*, so the payload is worth keeping."""
        match parts:
            # The third of three layers: the refresh number drops a stale picture, `_open` empties
            # the park list, and this refuses a park of another run that reaches it any other way.
            case ["answer", payload] if self.parks and self.parks[0].task_id == self.task_id:
                self.answer_park(self.parks[0], payload)
                return True
            case ["answer", _] | ["answer"]:
                self.notify(
                    "nothing is parked yet — the answer is kept, press enter again after `r`",
                    severity="warning",
                )
                return False
            case ["open", which]:
                return self._open(which)
            case ["help"] | []:
                self.notify(
                    "answer <json> · open <n> · : command · esc back · "
                    "r refresh · p projection · m markdown"
                )
                return True
            case [unknown, *_]:
                self.notify(f"unknown command {unknown!r}", severity="warning")
                return False
        # `ty` cannot see that a list is empty or non-empty and that both are matched above, so
        # the match is total in fact and open in type. A trailing refusal rather than an
        # `assert_never`: this alphabet is `list[str]`, not a closed union, so there is no
        # exhaustiveness to assert — and refusing is the right answer for a shape nobody matched.
        return False

    def _open(self, which: str) -> bool:
        found = runread.runs(self.db_path)
        try:
            self.task_id = found[int(which)].task_id
        except ValueError, IndexError:
            self.notify(f"no run {which!r} (0..{len(found) - 1})", severity="warning")
            return False
        # The park list belongs to the run we just LEFT until the refresh lands, and `answer`
        # reads `self.parks[0]`. Clearing it here makes the window refuse rather than answer the
        # wrong run's park — the same "refused rather than routed" rule the transition follows.
        self.parks = ()
        self.selected = None
        self.refresh_view()
        return True

    @work(thread=True)
    def answer_park(self, park: ParkedTask, payload: str) -> None:
        """Append the event a parked run is waiting on, then re-read.

        NOT `exclusive`: a write must not be cancelled by a later refresh. `SqliteApp` here opens
        the store to emit and nothing else — it registers no task, so it cannot claim one, which
        is what keeps this surface structurally unable to reach the incident that made the drain
        and the client separate things.
        """
        app = SqliteApp(str(self.db_path))
        try:
            answer(app, park, json.loads(payload))
        except json.JSONDecodeError as exc:
            self._to_ui(self.notify, f"payload is not JSON: {exc}", severity="error")
            return
        finally:
            app.close()
        self._to_ui(self.notify, f"answered {park.wake_event}")
        self._to_ui(self.refresh_view)


def _label(node: Node) -> str:
    """A leaf's text: its identity, then only what was MEASURED.

    `is not None` rather than truthiness on cost, the same rule `to_mermaid` and `to_text` follow —
    a measured-and-free call is `0.0` and must not render as an unmeasured blank."""
    text = identity(node)
    badges = []
    if node.count > 1:
        badges.append(f"x{node.count}")
    if node.cost is not None:
        badges.append(format_cost(node.cost))
    if node.state != "committed":
        badges.append(node.state)
    return f"{text}  ({' · '.join(badges)})" if badges else text


def _render_record(value: Any) -> str:
    """A recorded result, laid out so a reader can actually read it.

    `json.dumps` escapes every newline, so a multi-line field arrives as one long line and the
    pane truncates it. That defeated the first thing it was pointed at: the prose machine's READ
    state runs a caller query so a human can inventory against evidence, and the evidence rendered
    as `"output": "src/effective/keys.py:160  class Key:\\n  711 syntactic candidate(s) for` and
    stopped. A long or multi-line field gets its own block; everything else stays on one line."""
    if not isinstance(value, dict):
        return "```json\n" + json.dumps(value, indent=2, default=str) + "\n```"
    blocks = []
    for name, field_value in value.items():
        # A guarded `match` rather than an `isinstance` chain: the alphabet is "any JSON value",
        # which no union closes, so `case _` is the honest bottom and `--totality` is right to
        # refuse the chain that read the same way without saying so.
        match field_value:
            case str() as text if "\n" in text or len(text) > 70:
                blocks.append(f"**{name}:**\n\n```\n{text}\n```")
            case _:
                blocks.append(f"**{name}:** `{json.dumps(field_value, default=str)}`")
    return "\n\n".join(blocks)


def _summary(view: RunView, selected: Node | None = None) -> str:
    """The default detail pane: the tree as text, plus what the selected op recorded.

    `not recorded` rather than nothing when a selected node has no row, because the misses are
    structural: a node under a projection represents several ops and the pending node has no
    producer. Saying so is the difference between "this op returned nothing" and "this label does
    not address one op"."""
    body = to_text(RunGraph(view.run_id, view.nodes))
    extra = ""
    if selected is not None:
        recorded = view.results.get(selected.key)
        rendered = "*not recorded* — a folded or pending node addresses no single op"
        if recorded is not None:
            rendered = _render_record(recorded)
        extra += f"\n\n**`{selected.key}`**\n\n{rendered}"
    if view.children:
        extra += "\n\n**Subagents:** " + ", ".join(f"`{c}`" for c in view.children)
    if (line := files_line(view)) is not None:
        extra += "\n\n" + line
    return f"```\n{body}\n```{extra}"


def main(argv: list[str] | None = None) -> int:
    """`python -m tui <store.db> [task-id]`."""
    args = sys.argv[1:] if argv is None else argv
    if not args:
        print("usage: python -m tui <store.db> [task-id]", file=sys.stderr)
        return 2
    task = UUID(args[1]) if len(args) > 1 else None
    RunViewApp(args[0], task).run()
    return 0
