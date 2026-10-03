"""The shaping layer (`effective.runview`) — the joins between a reader and a renderer.

Pure values throughout: `run_view` opens nothing, so every case here is a list of key strings plus
the two records that sit beside a tape. That is deliberate and is the same property `graphview`
has — if the shaping needed a database, it could not be the thing four surfaces share.

What these assert is mostly *drift resistance* rather than behaviour. The module exists because
the readers are reusable and the shaping between reader and render is not, and the repo has three
recorded instances of a writer and a reader disagreeing about a shape neither owned.
"""

from uuid import UUID

import pytest

from effective.cards.spec import Action
from effective.graphview import from_keys
from effective.keys import Index
from effective.parked import ParkedTask
from effective.runview import ANSWER_CMD, RunView, file_summary, run_view, to_markdown

MACHINE_TAPE = [
    "d:0;state:test;step;tool:run_suite",
    "d:1;state:draft;step;tool:read_file",
    "d:1;state:draft;step;tool:apply_fix",
    "artifact:application/json,sha256-d47e",
    "ledger;machine:r-9;commit",
]


def test_a_spawn_is_found_INSIDE_a_frame_where_a_prefix_match_would_miss_it():
    """The join that a text match gets wrong exactly where it matters.

    A `startswith` against the spawn key's spelling passes every test you would write from the
    top-level case and fails on the shape a real fan-out produces, because a `gather` frame
    precedes the arm and the string no longer starts where the pattern anchors. Both spawns below
    sit under a branch, which is the only place a fan-out puts them.

    This is the nominal-where-structural defect the repo keeps finding, so the test is written
    to fail for a text matcher rather than to confirm the parser.
    """
    graph = from_keys(
        "r",
        [
            "gather:0,0;step;tool:spawn,child-a",
            "gather:0,1;step;tool:spawn,child-b",
            "step;tool:read_file",
        ],
    )
    assert run_view(graph).children == ("child-a", "child-b")


def test_a_run_with_no_spawns_and_no_ledger_still_shapes():
    """Both joins default to empty, so a pane holding only the tape gets a `RunView`.

    Worth pinning because the alternative — requiring a ledger and a park list — would push every
    caller into inventing the halves it does not have, which is how a reader ends up with a
    plausible empty value instead of an honest absent one.
    """
    view = run_view(from_keys("r-9", MACHINE_TAPE))
    assert view.run_id == "r-9"
    assert len(view.nodes) == 5
    assert view.children == ()
    assert view.files == ()
    assert view.actions == ()


def test_committed_files_come_from_the_CANONICAL_record_not_the_tape():
    """Two bookkeepers, and this join reads the right one.

    A `ledger;` key on the tape proves the op was recorded; it does not carry what was committed.
    The file list lives in the `machine-committed` row's payload, which is the append-only record,
    and only rows of that kind contribute — a run's other ledger rows are not a file list.
    """
    ledger = [
        {"kind": "machine-committed", "files": ["mod.py", "test_mod.py"], "event_id": "c"},
        # A DIFFERENT kind that also carries `files`. Without this row the kind filter is
        # untestable — every other row lacks the field, so `row.get("files") or ()` returns the
        # same answer whether the filter runs or not, and a mutation deleting it stayed green.
        # `extra="allow"` on `LedgerRow` means any row may carry any field, so this is the shape
        # to defend against rather than a contrived one.
        {"kind": "machine-finished", "files": ["not-a-commit.py"], "event_id": "f"},
    ]
    view = run_view(from_keys("r-9", MACHINE_TAPE), ledger=ledger)
    assert view.files == ("mod.py", "test_mod.py")


STATING = {"kind": "machine-committed", "files": ["a.py", "b.py"], "changed": ["b.py"]}
SILENT = {"kind": "machine-committed", "files": ["a.py"]}
FINISHED = {"kind": "machine-finished", "changed": ["not-a-commit.py"]}
CHANGED = {
    "a row that states it": ([STATING, FINISHED], ("b.py",)),
    "a row that does not say": ([SILENT, FINISHED], None),
    "a row that changed nothing": ([{**STATING, "changed": []}], ()),
    "a silent row beside a stating one": ([SILENT, STATING], None),
    "two stating rows": ([STATING, {**SILENT, "changed": ["a.py"]}], ("b.py", "a.py")),
}


@pytest.mark.parametrize(("ledger", "changed"), CHANGED.values(), ids=CHANGED.keys())
def test_a_view_says_which_committed_files_changed_and_when_it_cannot(ledger, changed):
    """`None` rather than an empty tuple when a commit row does not say, which would claim nothing
    changed; one silent row makes the whole run's answer unknown."""
    assert run_view(from_keys("r-9", MACHINE_TAPE), ledger=ledger).changed == changed


def test_an_action_carries_the_park_name_the_reader_READ():
    """Never a name composed here, and the reason is not tidiness.

    A wake registration carries coordinates no caller can reconstruct — the enclosing frames, and
    `Key.occurrence`'s `#N` at a second ask — so a composed name settles a different park or none.
    `Action.target` is `wake_event` verbatim, frames and all, which is what `parked.answer`
    consumes one layer down.
    """
    park = ParkedTask(
        task_id=UUID(int=1),
        task_name="review",
        wake_event="gather:0,1;review:m1#2",
        state="parked",
    )
    (action,) = run_view(from_keys("r-9", MACHINE_TAPE), parked=[park]).actions
    assert action == Action(
        cmd=ANSWER_CMD, label="answer review", target="gather:0,1;review:m1#2", risk="medium"
    )


def test_an_unmeasured_run_reports_no_cost_rather_than_zero():
    """`None` and `0.0` are different answers, and the fold must not collapse them.

    A run where nobody measured anything is not a free run. The second half is the case that
    makes the distinction real: one measured-and-free node totals `0.0`, which a truthiness test
    would have reported as unmeasured.
    """
    tape = ["step;tool:a", "step;tool:b"]
    assert run_view(from_keys("r", tape)).cost is None
    free = run_view(from_keys("r", tape, telemetry={"step;tool:a": (0.0, None)}))
    assert free.cost == 0.0


def test_markdown_keeps_the_write_back_legible_in_the_flattest_target():
    """Flatten at the renderer, never at the boundary — the Bobby-Tables rule on the view axis.

    Markdown is the floor: no host, no JS, no layout engine. What it must NOT do is dissolve the
    typed action into prose, because the same value serves a Shiny button and an MCP action. So
    `cmd` and `target` survive into the text, and the tree degrades to the same containment the
    SVG would draw.
    """
    park = ParkedTask(
        task_id=UUID(int=1), task_name="review", wake_event="review:m1", state="parked"
    )
    ledger = [
        {"kind": "machine-committed", "files": ["mod.py"], "changed": ["mod.py"], "event_id": "c"}
    ]
    graph = from_keys("r-9", MACHINE_TAPE, telemetry={MACHINE_TAPE[0]: (0.0012, None)})
    rendered = to_markdown(run_view(graph, ledger=ledger, parked=[park]))

    assert rendered.startswith("## r-9\n")
    assert "**Cost:** $0.0012" in rendered
    assert "└─ state:test" in rendered  # the tree, not a bullet list
    assert "**Changed:** `mod.py`" in rendered
    assert "(`answer` → review:m1)" in rendered  # cmd and target, not "click to answer"


def test_the_view_is_neutral_about_which_projection_it_was_handed():
    """`nodes` is whatever quotient the caller chose, so one type serves every pane.

    A dashboard wants `project` (axes discovered from the tape); cross-run alignment wants
    `fold_cycles` (axes declared, so two runs are comparable). Shaping must not pick for them.
    """
    from effective.graphview import fold_cycles, project

    graph = from_keys("r-9", MACHINE_TAPE)
    for projection in (graph, project(graph), fold_cycles(graph, drop=(Index,))):
        view = run_view(projection)
        assert isinstance(view, RunView)
        assert view.nodes == projection.nodes


FILE_LINES = {
    "a row that states what changed": (
        [{"kind": "machine-committed", "files": ["a.py", "b.py"], "changed": ["b.py"]}],
        "**Changed:** `b.py`",
    ),
    "a row that changed nothing": (
        [{"kind": "machine-committed", "files": ["a.py"], "changed": []}],
        "**Changed:** none",
    ),
    "a row that does not say": (
        [{"kind": "machine-committed", "files": ["a.py", "b.py"]}],
        "**Committed:** `a.py`, `b.py`",
    ),
    "a removal of the only file": (
        [{"kind": "machine-committed", "files": [], "changed": ["a.py"]}],
        "**Changed:** `a.py`",
    ),
}


@pytest.mark.parametrize(("ledger", "line"), FILE_LINES.values(), ids=FILE_LINES.keys())
def test_markdown_names_what_changed_and_says_when_it_cannot(ledger, line):
    rendered = to_markdown(run_view(from_keys("r-9", MACHINE_TAPE), ledger=ledger))
    assert f"\n{line}\n" in rendered
    assert rendered.count("**Changed:**") + rendered.count("**Committed:**") == 1


@pytest.mark.parametrize(
    "ledger",
    [
        [],
        [{"kind": "machine-committed", "files": []}],
        [{"kind": "machine-committed", "files": [], "changed": []}],
    ],
    ids=["no commit", "an empty commit", "an empty commit that says so"],
)
def test_a_run_that_committed_nothing_draws_no_files_line(ledger):
    view = run_view(from_keys("r-9", MACHINE_TAPE), ledger=ledger)
    assert file_summary(view) is None
    assert "Changed" not in to_markdown(view)
    assert "Committed" not in to_markdown(view)
