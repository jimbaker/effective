"""The coder driven by a scripted model: one bug, its failing test, and the turns that fix it.

The tools, the gate and the judge are the real ones; the model's turns and the two container tools
are scripted, so a run needs no image and no network.
"""

from typing import Any

from effective.api import Effect
from effective.domain import AskLLM, CallTool, DomainOp
from effective.machine.evidence import CommandRun
from effective.react import AssistantTurn, ToolRequest
from examples.coder.machine import coder
from examples.coder.tools import TOOLS, serve

MODULE = "mod.py"
BUGGY = "def add(a, b):\n    return a - b\n"
FIXED = "def add(a, b):\n    return a + b\n"
TEST = "from mod import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
SEED = {MODULE: BUGGY, "test_mod.py": TEST}


LOOKED = "exit code 1"
STEP = "def add(a, b):\n    return a + b  # step one\n"


def acting(name: str, args: dict[str, Any]) -> AssistantTurn:
    return AssistantTurn(thought=name, tool=ToolRequest(name=name, args=args))


def edited(old: str, new: str) -> AssistantTurn:
    return acting("edit", {"path": MODULE, "edits": [{"old_text": old, "new_text": new}]})


def is_update(observation: str) -> bool:
    """Whether an observation reports a write, however the path is carried in it.

    Asked rather than compared, because the path is about to reach the model inside a data fence:
    a model chooses it and `tree_path` admits a newline in one."""
    return observation.startswith("Updated") and MODULE in observation


def scripted_run(exit_code: int, output: str) -> Any:
    """A scripted `bash` result in whatever shape the tool DECLARES, so this double cannot drift
    from the catalog the loop validates against."""
    return TOOLS["bash"].result(exit_code=exit_code, output=output)


def turn_for(observations: list[str]) -> AssistantTurn:
    """The scripted model: look, read, try an edit that does not lint, fix the bug, then edit the
    note that fix left behind, and answer.

    A function of the transcript, so a re-executed turn decides the same way. The last edit finds
    its old text only in what the previous edit produced, which is what makes the binding of the
    tree observable: a worker that bound the visit's starting tree would be refused there."""
    last = observations[-1] if observations else ""
    if not observations:
        return acting("bash", {"command": "python -m pytest -q"})
    if last.startswith(LOOKED):
        return acting("read", {"path": MODULE})
    if "a - b" in last:
        return edited("a - b", "a +")
    if last.startswith("[refused]"):
        return edited("a - b", "a + b  # step one")
    if is_update(last) and sum(1 for seen in observations if is_update(seen)) == 1:
        return edited("  # step one", "")
    return AssistantTurn(thought="done", answer="add adds")


class Deployment:
    """The coder's own tool server, with the model and the two container tools scripted."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages):
                return turn_for([m["content"] for m in messages if m["role"] == "tool"])
            case CallTool(name="bash", args={"tree": tree}):
                return scripted_run(1 if tree[MODULE] == BUGGY else 0, "1 failed")
            case CallTool(name="run_suite", args={"tree": tree}):
                return CommandRun(exit_code=1 if tree[MODULE] == BUGGY else 0)
            case CallTool():
                return serve(op)
            case _:
                raise AssertionError(f"unexpected op {op!r}")


def fixing(run_id: str) -> Effect[dict[str, Any]]:
    return (yield from coder(run_id, "make the test pass", SEED, visits=2, turns=8))
