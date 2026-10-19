"""`[tool.effective]` in the nearest `pyproject.toml`: the settings a tree declares for its lint,
its tests and its scripts. It imports no parser, so a test session reads it without the lint."""

import re
import tomllib
from pathlib import Path
from typing import Any, TypedDict

from pydantic import TypeAdapter, ValidationError


def config_root() -> tuple[Path, dict[str, Any]] | None:
    """The nearest `pyproject.toml` at or above the cwd with a `[tool.effective]` table, and that
    table. The search stops at the repository root, so a workspace member's own `pyproject.toml`
    does not hide the root's."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (pyproject := directory / "pyproject.toml").is_file():
            with pyproject.open("rb") as f:
                tool = tomllib.load(f).get("tool", {})
            if "effective" in tool:
                return directory, tool["effective"]
        if (directory / ".git").exists():
            return None
    return None


class _LazyImport(TypedDict):
    path: str
    module: str
    reason: str


_SHAPES: tuple[tuple[re.Pattern[str], TypeAdapter[Any]], ...] = (
    (
        re.compile(r"_srcs$|^extra_paths|^private_ops_nouns$|_exempt$|^app_tables$"),
        TypeAdapter(list[str]),
    ),
    (re.compile(r"^seam_forbidden$"), TypeAdapter(dict[str, list[str]])),
    (re.compile(r"^not_"), TypeAdapter(dict[str, str])),
    (re.compile(r"^lazy_import_allowed$"), TypeAdapter(list[_LazyImport])),
)
"""The shape each configured name must have, so a string where a list belongs raises rather than
extending nothing."""


def configured(name: str, table: str = "lint") -> Any:
    """`[tool.effective.<table>].<name>`, from `config_root`, checked against `_SHAPES`.

    A tree that holds more than this repository ships (more workflow modules, more packages a seam
    keeps out) declares the extension there, so every tree runs the same lint."""
    found = config_root()
    value = None if found is None else found[1].get(table, {}).get(name)
    if value is None:
        return None
    for pattern, shape in _SHAPES:
        if pattern.search(name):
            try:
                shape.validate_python(value, strict=True)
            except ValidationError as wrong:
                raise TypeError(f"[tool.effective.{table}].{name} has the wrong shape") from wrong
    return value
