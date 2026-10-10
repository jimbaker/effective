"""A consumer on a former engine or handler path patches the module that defines what it uses."""

import subprocess
import sys
import textwrap

import pytest

FORMER = {
    "effective.sqlite": ("effective.engines.sqlite", "enable_wal"),
    "effective.handlers.absurd": ("effective.handlers.durable", "metered_call"),
}

PROBE = textwrap.dedent(
    """
    import importlib, sys

    former, defining, name = sys.argv[1:]
    old = importlib.import_module(former)
    parent, _, leaf = former.rpartition(".")
    marker = object()
    setattr(old, name, marker)
    new = importlib.import_module(defining)
    assert getattr(new, name) is marker, "a patch through the former path missed the module"
    assert getattr(importlib.import_module(parent), leaf) is new, "the parent names another module"
    """
)


@pytest.mark.parametrize(("former", "defining", "name"), [(f, *d) for f, d in FORMER.items()])
def test_a_patch_through_a_former_path_reaches_the_defining_module(former, defining, name):
    ran = subprocess.run(
        [sys.executable, "-c", PROBE, former, defining, name],
        capture_output=True,
        text=True,
        check=False,
    )
    assert ran.returncode == 0, ran.stderr
