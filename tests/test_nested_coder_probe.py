"""A coder inside a coder: recursion as a semantic probe.

ROLE: journey. `wiki/concepts/recursion-shapes.md` states the method: run a classic shape
through the substrate and a boundary that does not compose shows up as a disagreement, and the
defects it finds are rarely recursion bugs. Nesting `examples.coder` on either engine finds one in
neither recursion nor the coder: the machine's two ledger ids are composed from the run id and the
generation and nothing else, so a second run inside one task writes the first one's canonical
address.

`run_machine`'s own docstring says exactly that, down to the naming a nested machine should use
(`{outer}-{state}-{visit}`), and the refusal is correct: an `event_id` is the domain's public
address and a projection reads it, so the substrate refuses rather than silently rescoping. These
two tests run that claim, and the positive half records that everything else about the nesting
composes.
"""

from typing import Any
from uuid import uuid4

from _conformance import Fault

from effective.api import Effect, scoped
from effective.domain import AskLLM, CallTool, DomainOp
from effective.keys import Name, Run, compose_key
from effective.machine.evidence import CommandRun
from effective.react import AssistantTurn
from examples.coder.machine import coder
from examples.coder.tools import serve

SEED = {"mod.py": "def add(a, b):\n    return a - b\n"}


class Answering:
    """Every turn answers at once and the suite is green, so the shape is all that varies."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM():
                return AssistantTurn(thought="done", answer="ok")
            case CallTool(name="run_suite"):
                return CommandRun(exit_code=0)
            case CallTool():
                return serve(op)
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def nesting(inner_run_id: str):
    """A coder whose workflow runs another coder first, under its own scope."""

    def body(run_id: str) -> Effect[dict[str, Any]]:
        inner = yield from scoped(
            compose_key(t"sub:{Name('inner')}"),
            lambda: coder(inner_run_id or run_id, "the inner goal", SEED, visits=1, turns=2),
        )
        outer = yield from coder(run_id, "the outer goal", SEED, visits=1, turns=2)
        return {"inner": inner, "outer": outer}

    return body


def run_nested(backend, inner_run_id: str):
    run_id = f"n{uuid4().hex}"
    name = compose_key(t"coder-nested:{Run(run_id)}").stored()
    backend.register(
        name,
        nesting(inner_run_id and f"{run_id}{inner_run_id}"),
        None,
        Fault(),
        [],
        fresh=Answering,
    )
    return backend.run_until_result(backend.spawn(name, run_id))


def test_two_runs_in_one_task_under_one_run_id_are_refused(backend):
    """The machine is not re-entrant under one run id, and the refusal says which row was lost."""
    snap = run_nested(backend, "")

    assert snap.state == "failed", snap
    assert "PlacedWriterCollision" in repr(snap.failure)


def test_a_coder_inside_a_coder_composes_when_the_inner_run_is_named(backend):
    """Everything else about the nesting already composes: the op keys stay apart, each level
    binds its own tree, and each commits its own artifact."""
    snap = run_nested(backend, "-inner")

    assert snap.state == "completed", snap
    assert snap.result["inner"]["passed"]
    assert snap.result["outer"]["passed"]
    assert snap.result["inner"]["artifact_id"] == snap.result["outer"]["artifact_id"]
