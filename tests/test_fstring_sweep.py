"""The f-string gate: what counts as a designated render backend."""

import pathlib
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import fstring_sweep  # noqa: E402

SOURCE = """
class Problem:
    def render(self, x):
        return f"at {x}"


class Other:
    def render(self, x):
        return f"other {x}"


def _render_name(op, target):
    def inner():
        return f"{op} {target}"
    return f"{op} {target}", inner
"""


def test_a_designation_names_one_function_by_its_qualified_name(tmp_path, monkeypatch):
    """Designating `Problem.render` exempts no other `render` in the file, and designating a
    function exempts no function nested inside it."""
    path = tmp_path / "telemetry.py"
    path.write_text(SOURCE)
    monkeypatch.setattr(
        fstring_sweep,
        "DESIGNATED",
        frozenset({(str(path), "Problem.render"), (str(path), "_render_name")}),
    )

    flagged = sorted(line for _, line, _, _ in fstring_sweep.findings([str(path)]))

    assert flagged == [9, 14]  # `Other.render`, and the function nested in `_render_name`


EXCEPTIONS = """
class Refused(Exception):
    pass


class RunRefused(Refused):
    def __init__(self, run_id):
        self.run_id = run_id

    def __str__(self):
        return f"run {self.run_id} was refused"


class Report:
    def __str__(self):
        return f"report {self.name}"


class Wrapped(Refused):
    def __str__(self):
        def inner():
            return f"inner {self.run_id}"
        return inner()
"""


def test_an_exception_says_itself_in_its_own_str_and_nowhere_else(tmp_path):
    """An exception class, through any depth of bases, may render its fields in `__str__`. A
    class that is not an exception may not, nor may a function nested in the `__str__`."""
    path = tmp_path / "errors.py"
    path.write_text(EXCEPTIONS)

    flagged = sorted(line for _, line, _, _ in fstring_sweep.findings([str(path)]))

    assert flagged == [16, 22]  # `Report.__str__`, and the function nested in `Wrapped.__str__`


def test_a_name_is_an_exception_only_when_every_class_of_it_is_one(tmp_path):
    """Two modules defining one name lend each other nothing, in either order, and a scanned class
    that shadows a builtin exception's name is read as itself."""
    (tmp_path / "a.py").write_text(
        "class Refused:\n    def __str__(self):\n        return f'x {self.v}'\n"
    )
    (tmp_path / "b.py").write_text("class Refused(Exception):\n    pass\n")
    (tmp_path / "c.py").write_text(
        "class TimeoutError:\n    def __str__(self):\n        return f'y {self.v}'\n"
    )
    a, b, c = (str(tmp_path / name) for name in ("a.py", "b.py", "c.py"))

    for order in ([a, b, c], [c, b, a]):
        flagged = sorted((path, line) for path, line, _, _ in fstring_sweep.findings(order))
        assert flagged == [(a, 3), (c, 3)], order


TWO_RAISES = """
def first(x):
    raise ValueError(f"no {x}")


def second(x):
    raise ValueError(f"no {x}")
"""

THREE_RAISES = """
def first(x):
    raise ValueError(f"no {x}")


def second(x):
    raise ValueError(f"no {x}")


def third(x):
    raise ValueError(f"no {x}")
"""

ONE_STRUCTURED = """
def first(x):
    raise Missing(x)


def second(x):
    raise ValueError(f"no {x}")
"""

ONE_CHANGED = """
def first(x):
    raise ValueError(f"no {x!r}")


def second(x):
    raise ValueError(f"no {x}")
"""


def _tree(tmp_path, monkeypatch, source: str) -> pathlib.Path:
    """A scratch tree holding one module and the baseline its sites count to."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    module = tmp_path / "src" / "errors.py"
    module.parent.mkdir()
    module.write_text(source)
    assert _sweep(monkeypatch, "--init-baseline") == 0
    return module


def _sweep(monkeypatch, *argv: str) -> int:
    monkeypatch.setattr(sys, "argv", ["fstring_sweep.py", *argv, "src"])
    return fstring_sweep.main()


def test_two_raises_sharing_a_message_are_counted_as_two(tmp_path, monkeypatch):
    """The record keeps how many sites share a key, so a third copy of a message fails the gate,
    and structuring one of the two leaves a count the gate refuses until it is lowered."""
    module = _tree(tmp_path, monkeypatch, TWO_RAISES)
    assert fstring_sweep.load_baseline() == {("src/errors.py", "no "): (fstring_sweep.RAISE, 2)}
    assert _sweep(monkeypatch, "--gate") == 0

    module.write_text(THREE_RAISES)
    assert _sweep(monkeypatch, "--gate") == 1

    module.write_text(ONE_STRUCTURED)
    assert _sweep(monkeypatch, "--gate") == 1
    assert _sweep(monkeypatch, "--init-baseline") == 0
    assert fstring_sweep.load_baseline() == {("src/errors.py", "no "): (fstring_sweep.RAISE, 1)}
    assert _sweep(monkeypatch, "--gate") == 0


def test_an_initialized_entry_keeps_the_reason_written_for_it(tmp_path, monkeypatch):
    """Rewriting the record counts the sites again and leaves each reviewed reason as written."""
    _tree(tmp_path, monkeypatch, 'def page(x):\n    return f"<p>{x}</p>"\n')
    baseline = tmp_path / fstring_sweep.BASELINE
    baseline.write_text(baseline.read_text().replace("html -> tdom.html", "html -> reviewed"))

    assert _sweep(monkeypatch, "--init-baseline") == 0

    assert fstring_sweep.load_baseline() == {("src/errors.py", "<p></p>"): ("html -> reviewed", 1)}


def _git(tmp_path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
    )


def test_a_raise_on_a_changed_line_fails_and_an_unchanged_one_passes(tmp_path, monkeypatch):
    """`--touched` refuses an f-string raise whose statement spans a line changed since the base,
    committed or not, and leaves every other raise to the counted record."""
    module = _tree(tmp_path, monkeypatch, TWO_RAISES)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "work")
    assert _sweep(monkeypatch, "--gate", "--touched", "main") == 0

    module.write_text(ONE_CHANGED)
    assert fstring_sweep.touched_raises(["src"], fstring_sweep.changed_lines("main") or {}) == [
        ("src/errors.py", 3)
    ]
    assert _sweep(monkeypatch, "--gate", "--touched", "main") == 1

    _git(tmp_path, "commit", "-q", "-am", "touch")
    assert _sweep(monkeypatch, "--gate", "--touched", "main") == 1
    assert _sweep(monkeypatch, "--gate") == 0  # the count alone cannot see the change


ONE_HOISTED = """
def first(x):
    message = f"no {x}"
    raise ValueError(message)


def second(x):
    raise ValueError(f"no {x}")
"""

THREE_LINES = """
def first(x):
    raise ValueError(
        f"no {x}",
        x,
    )
"""

ARGUMENT_DELETED = """
def first(x):
    raise ValueError(
        f"no {x}",
    )
"""


def _branch(tmp_path, monkeypatch, source: str) -> pathlib.Path:
    """A scratch repo whose `main` holds `source`, checked out on a branch `work`."""
    module = _tree(tmp_path, monkeypatch, source)
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "base")
    _git(tmp_path, "checkout", "-q", "-b", "work")
    return module


@pytest.mark.parametrize("config", ["diff.noprefix", "diff.mnemonicPrefix"])
def test_the_changed_lines_are_read_whatever_the_git_config(tmp_path, monkeypatch, config):
    """A user's diff configuration shapes the headers git prints, and the rule reads them alike."""
    module = _branch(tmp_path, monkeypatch, TWO_RAISES)
    _git(tmp_path, "config", config, "true")

    module.write_text(ONE_CHANGED)

    assert _sweep(monkeypatch, "--gate", "--touched", "main") == 1


def test_a_clone_without_the_base_reads_the_remote_and_a_missing_base_fails(tmp_path, monkeypatch):
    """A checkout holding only `origin/main` reads the changed lines from it, and a tree where
    neither resolves fails rather than checking nothing."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _branch(origin, monkeypatch, TWO_RAISES)
    (origin / "src" / "errors.py").write_text(ONE_CHANGED)
    _git(origin, "commit", "-q", "-am", "touch")
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", "--branch", "work", str(origin), str(clone))
    monkeypatch.chdir(clone)

    assert fstring_sweep.touched_raises(["src"], fstring_sweep.changed_lines("main") or {}) == [
        ("src/errors.py", 3)
    ]
    assert _sweep(monkeypatch, "--gate", "--touched", "main") == 1
    assert fstring_sweep.changed_lines("nowhere") is None
    _git(clone, "checkout", "-q", "main")
    (clone / "src" / "errors.py").write_text(TWO_RAISES)
    assert _sweep(monkeypatch, "--gate", "--touched", "nowhere") == 1


def test_a_message_hoisted_out_of_its_raise_is_refused(tmp_path, monkeypatch):
    """Building a raise's message on the line above it keeps the f-string the raise names, so the
    rule refuses the raise and the record refuses the site counted as a raise."""
    module = _branch(tmp_path, monkeypatch, TWO_RAISES)

    module.write_text(ONE_HOISTED)

    assert fstring_sweep.touched_raises(["src"], fstring_sweep.changed_lines("main") or {}) == [
        ("src/errors.py", 4)
    ]
    assert _sweep(monkeypatch, "--gate") == 1


def test_a_deletion_inside_a_raise_touches_it(tmp_path, monkeypatch):
    """Deleting an argument's line changes the raise as surely as editing one."""
    module = _branch(tmp_path, monkeypatch, THREE_LINES)

    module.write_text(ARGUMENT_DELETED)

    assert fstring_sweep.touched_raises(["src"], fstring_sweep.changed_lines("main") or {}) == [
        ("src/errors.py", 3)
    ]
