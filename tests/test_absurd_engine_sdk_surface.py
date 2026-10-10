"""The private Absurd SDK surface `effective.engines.absurd` relies on exists at the pinned SDK.

The cases are read from the engine module's own source, so a new private touch is pinned the day
it is written. An SDK upgrade that moves one of these fails here, by name.
"""

import ast
import inspect
from pathlib import Path

import pytest

import effective.engines.absurd as engine

absurd_sdk = pytest.importorskip("absurd_sdk")

_TREE = ast.parse(Path(inspect.getfile(engine)).read_text())


def _own_attributes() -> set[str]:
    """Attributes the module's own classes assign on `self`."""
    owned: set[str] = set()
    for node in ast.walk(_TREE):
        match node:
            case ast.Assign(targets=targets):
                owned |= {attr for target in targets if (attr := _on_self(target))}
            case ast.AnnAssign(target=target) if attr := _on_self(target):
                owned.add(attr)
            case _:
                pass
    return owned


def _on_self(target: ast.expr) -> str | None:
    match target:
        case ast.Attribute(value=ast.Name(id="self"), attr=attr):
            return attr
        case _:
            return None


def _private_attributes() -> set[str]:
    """Single-underscore attributes read or written on an SDK object."""
    used: set[str] = set()
    for node in ast.walk(_TREE):
        match node:
            case ast.Attribute(attr=attr) if attr.startswith("_") and not attr.startswith("__"):
                used.add(attr)
            case _:
                pass
    return used - _own_attributes()


def _claim_keys() -> set[str]:
    """String keys subscripted on the SDK's claimed-task row (`ctx._task[...]`, or `claimed`)."""
    keys: set[str] = set()
    for node in ast.walk(_TREE):
        match node:
            case ast.Subscript(
                value=ast.Attribute(attr="_task") | ast.Name(id="claimed"),
                slice=ast.Constant(value=str() as key),
            ):
                keys.add(key)
            case _:
                pass
    return keys


def _private_imports() -> set[str]:
    names: set[str] = set()
    for node in ast.walk(_TREE):
        match node:
            case ast.ImportFrom(module="absurd_sdk", names=aliases):
                names |= {alias.name for alias in aliases if alias.name.startswith("_")}
            case _:
                pass
    return names


def test_the_scan_finds_the_private_surface() -> None:
    """Guards the tests below against passing over an empty scan."""
    assert {"_lookup_checkpoint", "_persist_checkpoint", "_task"} <= _private_attributes()
    assert {"task_id", "run_id", "attempt"} <= _claim_keys()
    assert "_CHECKPOINT_NOT_FOUND" in _private_imports()


@pytest.mark.parametrize("name", sorted(_private_attributes()))
def test_each_private_attribute_exists_on_the_sdk_task_context(name: str) -> None:
    sdk_ctx = absurd_sdk.TaskContext
    assert callable(getattr(sdk_ctx, name, None)) or name in sdk_ctx.__annotations__


@pytest.mark.parametrize("key", sorted(_claim_keys()))
def test_each_claim_key_is_a_field_of_the_claimed_task(key: str) -> None:
    assert key in absurd_sdk.ClaimedTask.__annotations__


@pytest.mark.parametrize("name", sorted(_private_imports()))
def test_each_private_import_exists_in_the_sdk(name: str) -> None:
    assert hasattr(absurd_sdk, name)


@pytest.mark.parametrize(
    ("method", "parameters"),
    [
        ("_lookup_checkpoint", ["self", "checkpoint_name"]),
        ("_persist_checkpoint", ["self", "checkpoint_name", "value"]),
    ],
)
def test_the_checkpoint_read_and_write_take_the_arguments_the_engine_passes(
    method: str, parameters: list[str]
) -> None:
    signature = inspect.signature(getattr(absurd_sdk.TaskContext, method))
    assert list(signature.parameters) == parameters
