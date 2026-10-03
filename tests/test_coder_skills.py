"""The coder with skills: activated once above the recursion, rendered at every level.

ROLE: journey. `combinators.hoisted` activates each skill before `fix` opens the
recursion, so every level renders the same recorded pin, and a pack that changes during the run
changes nothing a replay renders.
"""

import re
from functools import partial
from string.templatelib import Template
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition

from effective.api import Effect
from effective.combinators import fix, hoisted
from effective.domain import AskLLM, CallTool, DomainOp
from effective.keys import Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.trampoline import Placement
from effective.react import AssistantTurn
from effective.skills import SkillRegistry
from examples.coder.machine import Pins, coder
from examples.coder.tools import serving

pytestmark = pytest.mark.journey

SEED = {"mod.py": "def add(a, b):\n    return a - b\n"}
SKILLS = ("style",)
MARKER = re.compile(r"USE V\d")
SWEEP_OPS = 19
"""What one unarmed pass of the depth-2 recursion yields: the recursion's 18 and one activation.
A figure, recomputed by `Fault(position=…).count`."""


def pack(body: str) -> SkillRegistry:
    """A pack whose one skill is markdown, which is what a pin records."""
    return SkillRegistry.in_memory({"style": ("how to write the change", Template(body))})


class Moving:
    """A pack that moves from V1 to V2 once it has served a disclosure, whichever attempt asked.

    Each visit answers with every version marker its prompt carries, its child's cited answer
    included, so the root's answer holds one marker per level that rendered the skill."""

    def __init__(self) -> None:
        self.body = "USE V1"
        self.disclosed = 0

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages):
                seen = [found for m in messages for found in MARKER.findall(m["content"])]
                return AssistantTurn(thought="done", answer=" ".join(seen))
            case CallTool(name="run_suite"):
                return CommandRun(exit_code=0)
            case CallTool():
                served = serving(pack(self.body))(op)
                if op.name == "skill-disclose":
                    self.disclosed += 1
                    self.body = "USE V2"
                return served
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def open_coder(recur):
    def body(
        run_id: str, goal: str, depth: int, under: Placement | None, pins: Pins
    ) -> Effect[dict[str, Any]]:
        delegate = (
            (lambda placed: recur(run_id, f"step {depth}", depth - 1, placed, pins))
            if depth
            else None
        )
        return (
            yield from coder(
                run_id, goal, SEED, visits=1, turns=2, under=under, delegate=delegate, pins=pins
            )
        )

    return body


def skilled(run_id: str, depth: int) -> Effect[dict[str, Any]]:
    return (
        yield from hoisted(
            SKILLS, lambda pins: fix(open_coder)(run_id, "the root goal", depth, None, pins)
        )
    )


def run_skilled(backend, fault: Fault, depth: int) -> tuple[Any, Moving]:
    run_id = f"s{uuid4().hex}"
    name = compose_key(t"coder-skilled:{Run(run_id)}").stored()
    domain = Moving()
    backend.register(name, partial(skilled, depth=depth), domain, fault, [])
    return backend.run_until_result(backend.spawn(name, run_id)), domain


def test_one_activation_serves_every_level(backend):
    snap, domain = run_skilled(backend, Fault(), depth=3)
    assert snap.state == "completed", snap
    assert domain.disclosed == 1
    assert snap.result["summary"] == " ".join(["USE V1"] * 4), "every level rendered the one pin"


@pytest.mark.parametrize("position", [FaultPosition.BEFORE_OP, FaultPosition.AFTER_THUNK])
def test_every_level_renders_the_recorded_pin_across_a_crash_at_every_op(backend, position):
    """The pack moves after its first disclosure, so a replay renders the recorded pin. The one
    exception is a crash between the disclosure and its checkpoint, where the next attempt
    discloses again and every level renders that second pin."""
    unarmed = Fault(position=position)
    snap, _domain = run_skilled(backend, unarmed, depth=2)
    assert snap.state == "completed", snap
    assert unarmed.count == SWEEP_OPS, "the run changed shape; re-derive the bound"
    for k in range(1, SWEEP_OPS + 1):
        fault = Fault(k, position=position)
        snap, domain = run_skilled(backend, fault, depth=2)
        assert fault.armed is False, k
        assert snap.state == "completed", (k, snap)
        disclosed_twice = position is FaultPosition.AFTER_THUNK and k == 1
        assert domain.disclosed == 1 + disclosed_twice, k
        rendered = "USE V2" if disclosed_twice else "USE V1"
        assert snap.result["summary"] == " ".join([rendered] * 3), k
