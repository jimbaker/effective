"""A first Effective workflow: a house agent asks a model for a setpoint and sets the thermostat.

Run it with `uv run python examples/first_workflow.py`; `docs/first-workflow.md` walks through it.
"""

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, assert_never

from pydantic import BaseModel

from effective import (
    Effect,
    MeteredInterpreter,
    Subject,
    Usage,
    append_ledger,
    ask_llm,
    call_tool,
    compose_key,
)
from effective.channels import Gated, Repair, render
from effective.checkpoints import keys, read_sqlite_task
from effective.domain import AskLLM, CallTool
from effective.engines.sqlite import SqliteApp, SqliteLedger, TaskSnapshot
from effective.handlers.durable import DurableHandler
from effective.layers import OpLayer
from effective.ops import LedgerRow


class Setpoint(BaseModel):
    celsius: float


def set_temperature(request_id: str, request: str, ceiling: float = 30) -> Effect[float | Repair]:
    celsius = Gated(float, lambda c: 10 <= c <= ceiling, "celsius outside the allowed range")
    prompt = t"The occupant said: {request}\nSet the thermostat to {celsius}"
    setpoint = render(prompt, output=Setpoint)
    response = yield from ask_llm("setpoint", setpoint.messages, dict)
    match setpoint.resolve(response):
        case Repair() as repair:
            return repair
        case Setpoint(celsius=target):
            yield from call_tool("thermostat", {"celsius": target}, str)
            event_id = compose_key(t"setpoint:{Subject(request_id)}")
            yield from append_ledger(LedgerRow(event_id=event_id, kind="setpoint", celsius=target))
            return target
        case unreachable:
            assert_never(unreachable)


@dataclass
class House:
    """The world the ops reach: a model that always responds with `celsius`, and a thermostat
    that is offline for its first `outages` calls. `attempts` counts the times the task ran."""

    celsius: float = 22
    outages: int = 0
    asked: int = 0
    attempts: int = 0

    def model(self, op: AskLLM[Any]) -> tuple[dict[str, float], Usage]:
        self.asked += 1
        return {"celsius": self.celsius}, Usage()

    def thermostat(self, op: CallTool[Any]) -> str:
        if self.outages:
            self.outages -= 1
            raise ConnectionError("the thermostat is offline")
        return f"set to {op.args['celsius']}°C"


def run_durably(
    house: House, request: str, layers: tuple[OpLayer[Any], ...] = ()
) -> tuple[TaskSnapshot, tuple[str, ...]]:
    """Run `set_temperature` as a task on the embedded SQLite engine, in one temporary file.

    A failed attempt is retried, and the retry replays the steps the file already holds."""
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "house.db"
        app = SqliteApp(str(db))

        @app.register_task("set-temperature")
        def task(params: dict[str, str], ctx: Any) -> float | Repair:
            house.attempts += 1
            domain = MeteredInterpreter(llm=house.model, tools=house.thermostat)
            ledger = SqliteLedger(app.conn, params["request_id"], app.write_lock)
            handler = DurableHandler(ctx, domain, ledger=ledger, op_layers=layers)
            return handler.run(lambda: set_temperature(params["request_id"], params["request"]))

        try:
            run = app.spawn("set-temperature", {"request_id": "r1", "request": request})
            if (done := app.run_until_result(run)) is None:
                raise RuntimeError("the task parked, and set_temperature never waits")
            return done, keys(read_sqlite_task(db, run))
        finally:
            app.close()


def main() -> None:
    done, steps = run_durably(House(), "make it warmer")
    print("run:      ", done.result, "via", *steps)

    house = House(outages=1)
    done, _ = run_durably(house, "make it warmer")
    print(
        "resume:   ",
        done.result,
        f"after an outage: {house.attempts} attempts, and the model was asked {house.asked} time",
    )

    house = House(celsius=45)
    done, steps = run_durably(house, "make it much warmer")
    print("guardrail:", done.result, "via", *steps)


if __name__ == "__main__":
    main()
