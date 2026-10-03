"""A first Effective workflow: a house agent asks a model for a setpoint and sets the thermostat.

Run it with `uv run python examples/first_workflow.py`; `docs/first-workflow.md` walks through it.
"""

from typing import assert_never

from pydantic import BaseModel

from effective import (
    Effect,
    RecordingHandler,
    ReplayHandler,
    ReplayMismatch,
    Subject,
    append_ledger,
    ask_llm,
    call_tool,
    compose_key,
)
from effective.channels import Gated, Repair, render
from effective.ops import LedgerRow


class Setpoint(BaseModel):
    celsius: float


def set_temperature(request_id: str, request: str, ceiling: float = 30) -> Effect[float | Repair]:
    celsius = Gated(float, lambda c: 10 <= c <= ceiling, "celsius outside the allowed range")
    prompt = t"The occupant said: {request}\nSet the thermostat to {celsius}"
    setpoint = render(prompt, output=Setpoint)
    answer = yield from ask_llm("setpoint", setpoint.messages, dict)
    match setpoint.resolve(answer):
        case Repair() as repair:
            return repair
        case Setpoint(celsius=target):
            yield from call_tool("thermostat", {"celsius": target}, str)
            event_id = compose_key(t"setpoint:{Subject(request_id)}")
            yield from append_ledger(LedgerRow(event_id=event_id, kind="setpoint", celsius=target))
            return target
        case unreachable:
            assert_never(unreachable)


RESPONSES = {"setpoint": {"celsius": 22}, "tool:thermostat": "set to 22°C"}


def main() -> None:
    recorder = RecordingHandler(RESPONSES)
    target = recorder.run(lambda: set_temperature("r1", "make it warmer"))
    print("record:   ", target, "via", *(entry.key.stored() for entry in recorder.trace))

    replayed = ReplayHandler(recorder.trace).run(lambda: set_temperature("r1", "make it warmer"))
    print("replay:   ", replayed, "with no model and no thermostat")

    hot = RecordingHandler({"setpoint": {"celsius": 45}})
    refused = hot.run(lambda: set_temperature("r2", "make it much warmer"))
    print("guardrail:", refused, "via", *(entry.key.stored() for entry in hot.trace))

    try:
        ReplayHandler(recorder.trace).run(lambda: set_temperature("r1", "make it warmer", 20))
    except ReplayMismatch as drift:
        print("drift:     ReplayMismatch:", drift)


if __name__ == "__main__":
    main()
