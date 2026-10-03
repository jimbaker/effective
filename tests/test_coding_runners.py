"""The runner table's dispatch: which tree the success predicate actually measures.

ROLE: unit. Pure decisions over `(workspace, args)`, with the subprocess stubbed; at stake is the
CHOICE of what to hand the subprocess.

The defect these pin: a dispatch that collapses three cases into two arms lets a `run_suite` call
with no arguments silently measure the handler-side workspace instead of the tree the workflow
named, and lets a malformed `tree` do the same by falling through the same `isinstance` into the
same arm. Two different mistakes wearing one answer.
"""

from collections.abc import Mapping
from typing import Any

import pytest

from effective.coding import runners
from effective.coding.runners import ToolError, Workspace, run_tool
from effective.machine.evidence import CommandRun

pytestmark = pytest.mark.unit

HANDLER_SIDE = {"mod.py": "# what the workspace happens to hold\n"}
WORKFLOW_SIDE = {"mod.py": "# what the workflow asked to measure\n"}


@pytest.fixture
def measured(monkeypatch) -> list[Mapping[str, str]]:
    """Records the tree the predicate was handed, without running pytest.

    Stubbing `run_suite` is the point rather than a shortcut: the decision under test is which
    tree reaches it, and letting the real one run would spend a subprocess per case to measure
    something no assertion here reads."""
    seen: list[Mapping[str, str]] = []

    def stub(tree: Mapping[str, str], **_: Any) -> CommandRun:
        seen.append(dict(tree))
        return CommandRun(exit_code=0)

    monkeypatch.setattr(runners, "run_suite", stub)
    return seen


@pytest.fixture
def ws() -> Workspace:
    return Workspace(tree=dict(HANDLER_SIDE))


def test_a_named_tree_is_the_one_measured(ws, measured):
    """The postamble's case. It passes the workflow-local tree explicitly so the recorded op
    carries what was measured, and a predicate that quietly measured something else would make
    the ledger's `passed` a claim about a different tree than the one it committed."""
    run_tool(ws, "run_suite", {"tree": dict(WORKFLOW_SIDE)})
    assert measured == [WORKFLOW_SIDE]


def test_NO_named_tree_measures_the_workspace_and_that_is_the_agreement(ws, measured):
    """A model asking "how are we doing?" mid-loop names no tree, and means the workspace it has
    been editing. Legitimate, and kept — but it is a distinct arm now rather than the place two
    unrelated inputs both land."""
    run_tool(ws, "run_suite", {})
    assert measured == [HANDLER_SIDE]


def test_a_MALFORMED_tree_refuses_instead_of_measuring_something_else(ws, measured):
    """A malformed `tree` must not land in the same arm as "no tree at all": a caller who asked
    for a specific measurement and got the argument wrong would be answered about a DIFFERENT
    tree, with a green exit code and nothing said.

    A refusal, not a fallback: the caller named a tree, so it had one in mind, and answering about
    another is worse than not answering."""
    with pytest.raises(ToolError, match="tree"):
        run_tool(ws, "run_suite", {"tree": "mod.py"})
    assert measured == [], "it measured something despite the malformed argument"


def test_an_explicit_None_reads_as_no_tree(ws, measured):
    """The boundary between the first two arms, stated so it cannot drift: `tree=None` and an
    absent `tree` are the same request, and neither is the malformed case."""
    run_tool(ws, "run_suite", {"tree": None})
    assert measured == [HANDLER_SIDE]


def test_an_empty_tree_is_a_real_request_not_a_missing_one(ws, measured):
    """`{}` is falsy, which is exactly how "measure nothing" turns into "measure everything" if
    the arm tests truthiness instead of shape. The predicate answers exit 5 on an empty tree —
    a real answer about the tree — and `CommandRun.green` already refuses to read that as
    passing."""
    run_tool(ws, "run_suite", {"tree": {}})
    assert measured == [{}]
