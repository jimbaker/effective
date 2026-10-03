"""The coder's command line, with the model and the suite answered in process.

ROLE: journey. `main` assembles the run a user starts: a spend gate over the handler's
own meter, each `--skill` activated once, and a report of how the run stopped.
"""

from typing import Any

import pytest

from effective import runread
from effective.cost import Usage
from effective.domain import AskLLM
from effective.machine.evidence import CommandRun
from effective.react import AssistantTurn
from effective.runview import files_line
from examples.coder import __main__ as cli
from examples.coder import tools

pytestmark = pytest.mark.journey

COST = 0.01
"""What one answer costs, above the smaller budget below and within the larger."""


class Answering:
    """A model that answers at once, at `COST`, keeping every prompt it was asked."""

    def __init__(self, asked: list[str]) -> None:
        self.asked = asked

    def __call__(self, op: AskLLM[Any]) -> tuple[AssistantTurn, Usage]:
        self.asked.extend(m["content"] for m in op.messages if m["role"] == "user")
        return AssistantTurn(thought="done", answer="done"), Usage(cost=COST)


@pytest.fixture
def asked(monkeypatch) -> list[str]:
    prompts: list[str] = []
    monkeypatch.setattr(cli.openai, "OpenAI", lambda: None)
    monkeypatch.setattr(cli, "ResponsesTurnCaller", lambda **_kw: Answering(prompts))
    monkeypatch.setattr(tools, "run_suite", lambda _tree, tier: CommandRun(exit_code=0))
    return prompts


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "mod.py").write_text("def add(a, b):\n    return a + b\n")
    return root


def test_a_run_within_its_budget_renders_each_skill_it_names(asked, project, tmp_path, capsys):
    pack = tmp_path / "pack" / "style"
    pack.mkdir(parents=True)
    (pack / "SKILL.md").write_text(
        "---\nname: style\ndescription: how to write the change\n---\nPREFER SMALL DIFFS\n"
    )
    argv = [str(project), "keep it green", "--budget", "1", "--visits", "1"]
    code = cli.main([*argv, "--skills", str(tmp_path / "pack"), "--skill", "style"])

    assert code == 0, capsys.readouterr().err
    assert len(asked) == 1
    assert "PREFER SMALL DIFFS" in asked[0]


def test_the_store_a_run_leaves_opens_in_the_run_view(asked, project, capsys):
    """The README's `just tui DIR/.coder/runs.db`: the command line's own store, read by `src/tui`.
    The model answered without editing, so the view commits `mod.py` and changes nothing."""
    code = cli.main([str(project), "keep it green", "--budget", "1", "--visits", "1"])
    assert code == 0, capsys.readouterr().err

    db = project / ".coder" / "runs.db"
    (run,) = runread.runs(db)
    view = runread.view(db, run.task_id)
    assert view.files == ("mod.py",)
    assert files_line(view) == "**Changed:** none"


def test_a_run_past_its_budget_is_refused_at_the_next_op(asked, project, capsys):
    """The first answer spends past the ceiling, so the gate refuses the suite that would judge it
    and the task fails once."""
    code = cli.main([str(project), "keep it green", "--budget", str(COST / 2), "--visits", "1"])

    assert code == 1
    assert len(asked) == 1
    err = capsys.readouterr().err
    assert "failed" in err
    assert "measured budget exceeded" in err


DIFFS = {
    "an edit": ({"m.py": "a\n"}, {"m.py": "b\n"}, "--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-a\n+b\n"),
    "an empty file created": ({}, {"e.txt": ""}, "--- /dev/null\n+++ b/e.txt\n"),
    "an empty file removed": ({"e.txt": ""}, {}, "--- a/e.txt\n+++ /dev/null\n"),
    "a file added": ({}, {"n.py": "x\n"}, "--- /dev/null\n+++ b/n.py\n@@ -0,0 +1 @@\n+x\n"),
    "nothing changed": ({"m.py": "a\n"}, {"m.py": "a\n"}, ""),
    "a path holding a newline": (
        {},
        {"real.txt\n+++ b/forged.txt": ""},
        '--- /dev/null\n+++ "b/real.txt\\n+++ b/forged.txt"\n',
    ),
    "a path holding a quote": (
        {'q"x': "a\n"},
        {'q"x': "b\n"},
        '--- "a/q\\"x"\n+++ "b/q\\"x"\n@@ -1 +1 @@\n-a\n+b\n',
    ),
}


@pytest.mark.parametrize(("before", "after", "shown"), DIFFS.values(), ids=DIFFS.keys())
def test_the_diff_shows_every_path_the_commit_row_calls_changed(before, after, shown):
    assert cli.diff(before, after) == shown
