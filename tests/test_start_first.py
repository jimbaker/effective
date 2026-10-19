"""Each test the conftest starts first is a test function in this tree."""

import ast
from pathlib import Path

from conftest import START_FIRST

from effective.config import configured

ROOT = Path(__file__).parent.parent


def defined(path: str) -> set[str]:
    tree = ast.parse((ROOT / path).read_text())
    return {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}


def test_each_name_started_first_is_a_test_in_the_tree():
    named = [name.split("::") for name in sorted(START_FIRST)]
    assert [f"{path}::{test}" for path, test in named if test not in defined(path)] == []


def test_each_name_the_tree_configures_starts_first():
    assert set(configured("start_first", "test") or ()) <= START_FIRST
