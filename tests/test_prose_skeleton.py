"""The five mutations that falsified `scripts/prose_skeleton.py`, pinned.

Written as a table because the instrument's whole claim is a DISCRIMINATION — prose-only edits
pass, everything else fails — and a claim of that shape is only tested by the cases it must
separate. Reading the code cannot distinguish "blanks prose" from "blanks every string", which is
what `str_value` catches.
"""

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prose_skeleton.py"


def _module():
    """Load the script as a module by PATH.

    `sys.path.insert` + `import prose_skeleton` works at run time and is an `unresolved-import` to
    `ty`, which resolves imports statically and cannot see a path mutated three lines earlier. An
    explicit loader says the same thing in a form both agree on."""
    spec = importlib.util.spec_from_file_location("prose_skeleton", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SOURCE = '''\
"""Module prose."""

import json
import sqlite3

MODE = "answer"
"""Attribute prose — the bare string after an assignment, which `ast.get_docstring` cannot see."""


def load(text: str) -> object:
    """Function prose."""
    return json.loads(text) if sqlite3 else None
'''


def _prose_only(text: str) -> str:
    return text.replace("Function prose.", "Function prose, reworded and longer.")


def _code_move(text: str) -> str:
    return text.replace("import json\nimport sqlite3", "import sqlite3\nimport json")


def _delete_prose(text: str) -> str:
    """The lossy pass: `just check` stays green if you delete every docstring in the repo."""
    return text.replace('    """Function prose."""\n', "")


def _str_value(text: str) -> str:
    """A string used as a VALUE is code. Blanking it would make the skeleton blind to a rename."""
    return text.replace('MODE = "answer"', 'MODE = "answer_TYPO"')


def _noop(text: str) -> str:
    return text


CASES = [
    # label, mutation, prose-only?, units expected to change
    ("prose_only", _prose_only, True, 1),
    ("noop", _noop, True, 0),
    ("code_move", _code_move, False, 0),
    ("delete_prose", _delete_prose, False, 1),
    ("str_value", _str_value, False, 0),
]


@pytest.fixture
def probe(tmp_path: Path) -> Path:
    path = tmp_path / "probe.py"
    path.write_text(SOURCE)
    return path


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=False
    )


@pytest.mark.parametrize(
    ("label", "mutate", "prose_only", "changed"), CASES, ids=[c[0] for c in CASES]
)
def test_only_a_prose_edit_passes_both_invariants(probe, label, mutate, prose_only, changed):
    snapshot = probe.parent / "before.json"
    assert _run("--save", str(snapshot), str(probe)).returncode == 0

    mutated = mutate(probe.read_text())
    assert (mutated != SOURCE) is (label != "noop"), f"{label}: the mutation did not apply"
    probe.write_text(mutated)

    result = _run("--check", str(snapshot), str(probe))
    assert result.returncode == (0 if prose_only else 1), result.stdout
    assert ("SKELETON CHANGED" in result.stdout) is not prose_only, result.stdout
    assert f"{changed} prose unit(s) changed" in result.stdout, result.stdout


def test_an_attribute_docstring_is_a_prose_unit(probe):
    """The census that missed these undercounted `src/` by 182 units and 14,699 words."""
    prose_skeleton = _module()
    names = prose_skeleton.units(probe)
    assert len(names) == 3, names  # module, the attribute docstring, and the function
    assert sorted(names) == ["<module>#0", "<module>#1", "load#0"], sorted(names)


def test_a_file_argument_is_not_silently_empty(tmp_path: Path, probe):
    """`rglob` matches nothing on a file path and returns in silence — the failure that put a
    false measured claim into four documents. Both report tools take a file; so does this."""
    result = _run(str(probe))
    assert "prose units" in result.stdout, result.stdout
    assert result.returncode == 0


def test_the_skeleton_ignores_prose_but_not_structure(probe):
    """The discrimination stated directly, without the CLI in the way."""
    prose_skeleton = _module()
    base = prose_skeleton.skeleton(probe)
    probe.write_text(_prose_only(SOURCE))
    assert prose_skeleton.skeleton(probe) == base
    probe.write_text(_code_move(SOURCE))
    assert prose_skeleton.skeleton(probe) != base
    assert ast.parse(SOURCE) is not None
