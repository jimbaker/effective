"""The building page's projected blocks agree with fresh runs of the startup workflows."""

import re
import sys
from pathlib import Path

import pytest

from effective.graphview import from_keys, to_sequence

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import project_startup_runs as projected  # noqa: E402

pytestmark = pytest.mark.journey


def test_every_projected_block_agrees_with_a_fresh_run():
    assert projected.main(["--check"]) == 0


def test_a_block_a_run_would_not_draw_fails_the_check(tmp_path, monkeypatch):
    page = tmp_path / "page.md"
    text = projected.PAGE.read_text()
    assert "step#59;tool:remediate" in text
    page.write_text(text.replace("step#59;tool:remediate", "step#59;tool:rollback"))
    monkeypatch.setattr(projected, "PAGE", page)
    assert projected.main(["--check"]) == 1


def _without(text: str, line: str) -> str:
    assert line in text
    return text.replace(line, "", 1)


def _moved_judgment(text: str) -> str:
    """The incident's judgment drawn before the evidence gather it reads."""
    judged = "  run->>world: step#59;judge:cause\n"
    first_branch = "  b1->>world: step#59;tool:deploys\n"
    assert first_branch in text
    return _without(text, judged).replace(first_branch, judged + first_branch, 1)


def _reversed_edge(text: str) -> str:
    assert "  n6 --> n7\n" in text
    return text.replace("  n6 --> n7\n", "  n7 --> n6\n", 1)


def _swapped_pages(text: str) -> str:
    """Page 0's read drawn under page 2's branch, and page 2's under page 0's."""
    for line in ("│  └─ rec:0\n", "│  └─ rec:2\n"):
        assert line in text
    swap = {"│  └─ rec:0\n": "│  └─ rec:2\n", "│  └─ rec:2\n": "│  └─ rec:0\n"}
    return re.sub(r"│  └─ rec:[02]\n", lambda m: swap[m[0]], text)


def _read_after_consolidating(text: str) -> str:
    """A page read drawn after the consolidation that merges every page's themes."""
    read = "  b3->>world: rec:2#59;step:observe\n"
    consolidated = "  run->>world: d:1#59;step:consolidate\n"
    assert consolidated in text
    return _without(text, read).replace(consolidated, consolidated + read, 1)


def _filed_before_judged(text: str) -> str:
    """One routing branch drawn filing its theme before judging its kind."""
    judged = "├─ gather:2,0\n│  ├─ step;judge:kind\n│  └─ step;tool:file\n"
    assert judged in text
    return text.replace(judged, "├─ gather:2,0\n│  ├─ step;tool:file\n│  └─ step;judge:kind\n", 1)


@pytest.mark.parametrize(
    "mutate",
    [
        _moved_judgment,
        _reversed_edge,
        _swapped_pages,
        _read_after_consolidating,
        _filed_before_judged,
    ],
)
def test_a_block_drawing_another_program_fails_the_check(tmp_path, monkeypatch, mutate):
    page = tmp_path / "page.md"
    page.write_text(mutate(projected.PAGE.read_text()))
    monkeypatch.setattr(projected, "PAGE", page)
    assert projected.main(["--check"]) == 1


def test_every_view_agrees_across_interleavings_that_keep_each_branch_in_order():
    keys = list(projected.timeline("voice"))
    routing = [k for k in keys if k.startswith("gather:2,")]
    assert len(routing) == 6
    reordered = [
        "gather:2,2;step;judge:kind",
        "gather:2,1;step;judge:kind",
        "gather:2,2;step;tool:file",
        "gather:2,1;step;tool:file",
        "gather:2,0;step;judge:kind",
        "gather:2,0;step;tool:file",
    ]
    assert sorted(reordered) == sorted(routing)
    other = from_keys("voice", [k for k in keys if k not in routing] + reordered)
    for view in projected.FENCE:
        drawn, redrawn = projected.drawn("voice", view), projected.render(view, other)
        assert drawn != redrawn or view == "graph"
        assert projected.agreed(view, drawn) == projected.agreed(view, redrawn)


def test_two_interleavings_of_one_program_agree():
    keys = ["gather:0,0;step;tool:a", "gather:0,1;step;tool:b", "step;tool:c"]
    first = "```mermaid\n" + to_sequence(from_keys("r", keys)) + "\n```\n"
    second = "```mermaid\n" + to_sequence(from_keys("r", [keys[1], keys[0], keys[2]])) + "\n```\n"
    assert first != second
    assert projected.agreed("sequence", first) == projected.agreed("sequence", second)


def test_a_step_named_with_a_gather_term_keeps_its_edges():
    keys = ["step;authorize", "step;gather:metrics", "step;charge"]
    forward = projected.render("graph", from_keys("r", keys))
    backward = projected.render("graph", from_keys("r", keys[::-1]))
    assert projected.agreed("graph", forward) != projected.agreed("graph", backward)


def _tree(keys: list[str]) -> object:
    return projected.agreed("tree", projected.render("tree", from_keys("r", keys)))


def test_a_gathers_branches_may_swap_and_two_gathers_may_not():
    first = ["gather:0,0;step:a", "gather:0,1;step:b"]
    second = ["gather:1,0;step:c", "gather:1,1;step:d"]
    assert _tree(first + second) == _tree(first[::-1] + second[::-1])
    assert _tree(first + second) != _tree(second + first)


def test_a_second_root_is_part_of_the_tree():
    block = projected.render("tree", from_keys("r", ["step:a"]))
    grafted = block.replace("\n```", "\nFAKE ROOT\n└─ step;pay\n```")
    assert grafted != block
    assert projected.agreed("tree", grafted) != projected.agreed("tree", block)
