"""The example face's key read on a real terminal: a bare Esc cancels, an arrow key does not.

A pipe or a `StringIO` would pass either way; the defect this pins lived in a text stream's
buffer taking a whole escape sequence, which only a tty shows."""

import importlib.util
import os
import pty
import sys
import tty
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).parent.parent / "examples" / "smol_durable.py"


@pytest.fixture
def face(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.syspath_prepend(str(EXAMPLE.parent))  # as running the example does
    spec = importlib.util.spec_from_file_location("smol_durable", EXAMPLE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    controller, terminal = pty.openpty()
    tty.setcbreak(terminal)
    stdin = os.fdopen(terminal, "r")
    monkeypatch.setattr(sys, "stdin", stdin)
    yield module, controller
    stdin.close()
    os.close(controller)


@pytest.mark.parametrize(("keys", "is_esc"), [(b"\x1b", True), (b"\x1b[A", False), (b"q", False)])
def test_only_a_bare_esc_reads_as_esc(face, keys: bytes, is_esc: bool):
    module, controller = face
    os.write(controller, keys)
    pressed = module._keypress()
    assert pressed == keys
    assert (pressed == module.ESC) is is_esc
