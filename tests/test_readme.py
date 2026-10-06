"""Pins for the README and the docs pages: every Python block is lines from a tested example.

A block names its source in the comment above its fence, `<!-- source: examples/x.py -->`. A line
that is only `...` stands for lines elided; each run of lines between them is a contiguous slice of
that file, and the runs come in order.
"""

import contextlib
import io
import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
PAGES = [ROOT / "README.md", ROOT / "docs" / "intro.md", ROOT / "docs" / "first-workflow.md"]
BLOCK_RE = re.compile(r"<!-- source: (\S+) -->\n```python\n(.*?)```", re.S)


def blocks(page: Path) -> list[tuple[str, str]]:
    return BLOCK_RE.findall(page.read_text(encoding="utf-8"))


def runs(block: str) -> list[list[str]]:
    out: list[list[str]] = [[]]
    for line in block.splitlines():
        if line.strip() == "...":
            out.append([])
        else:
            out[-1].append(line.rstrip())
    return [run for run in out if run]


def slices_in_order(shown: list[list[str]], source: list[str]) -> bool:
    start = 0
    for run in shown:
        found = next(
            (
                i
                for i in range(start, len(source) - len(run) + 1)
                if source[i : i + len(run)] == run
            ),
            None,
        )
        if found is None:
            return False
        start = found + len(run)
    return True


def run(example: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.syspath_prepend(str(ROOT / "examples"))
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        runpy.run_path(str(ROOT / "examples" / example), run_name="__main__")
    return out.getvalue()


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_every_python_block_names_its_source(page: Path):
    fenced = page.read_text(encoding="utf-8").count("```python\n")
    assert fenced == len(blocks(page)) > 0


@pytest.mark.parametrize(
    ("page", "source", "block"),
    [(page, source, block) for page in PAGES for source, block in blocks(page)],
    ids=lambda v: v.name if isinstance(v, Path) else v if v.endswith(".py") else "",
)
def test_each_block_is_its_source_in_order(page: Path, source: str, block: str):
    lines = [line.rstrip() for line in (ROOT / source).read_text(encoding="utf-8").splitlines()]
    assert slices_in_order(runs(block), lines), (
        f"{page.name}: the block from {source} has drifted from it"
    )


def test_the_toy_loop_alternates_reason_and_act(monkeypatch: pytest.MonkeyPatch):
    printed = run("react_toy.py", monkeypatch).splitlines()
    assert [line.split("(", 1)[0] for line in printed] == ["Reason", "Act", "Reason"]
    assert "observation='sent a door code for 13:00 to 14:00'" in printed[2]


def test_the_audit_layer_sees_each_op_before_and_after(monkeypatch: pytest.MonkeyPatch):
    printed = run("hooks_as_layers.py", monkeypatch).splitlines()
    assert printed == [
        "before a call: setpoint",
        "after a call:  setpoint -> {'celsius': 22}",
        "before a call: tool:thermostat",
        "after a call:  tool:thermostat -> set to 22.0°C",
        "before a call: AppendLedgerRow",
        "after a call:  AppendLedgerRow -> "
        "{'event_id': 'setpoint:r1', 'kind': 'setpoint', 'celsius': 22.0}",
        "result: 22.0",
    ]


def test_a_dropped_line_is_drift():
    source = ["def f():", "    a = 1", "    b = 2", "    return a + b"]
    assert slices_in_order(runs("def f():\n    a = 1\n...\n    return a + b\n"), source)
    assert not slices_in_order(runs("def f():\n    b = 2\n"), source)
    assert not slices_in_order(runs("    return a + b\n...\ndef f():\n"), source)
