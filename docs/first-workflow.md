# Your first workflow

One file, run three ways, with no model and no service. `examples/first_workflow.py` is a house
agent: it asks a model for a setpoint, sets the thermostat, and records what it did. It runs as a
durable task on the embedded SQLite engine, in a temporary file: once as it should, once through
a thermostat outage that the retry resumes, and once with a guardrail refusing a bad response.

## Run it

You need [uv](https://docs.astral.sh/uv/), which installs the pinned Python 3.14 for you. From the
repo root:

```bash
uv sync
uv run python examples/first_workflow.py
```

The first `uv sync` installs the whole development environment, so give it a minute or two. Then:

```text
run:       22.0 via step:setpoint step;tool:thermostat ledger;setpoint:r1
resume:    22.0 after an outage: 2 attempts, and the model was asked 1 time
guardrail: {'reason': 'celsius outside the allowed range'} via step:setpoint
```

`uv run pytest --no-cov tests/test_first_workflow.py` pins each of those lines, and this block.

## The workflow

<!-- source: examples/first_workflow.py -->
```python
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
from effective.handlers.absurd import DurableHandler
from effective.layers import OpLayer
from effective.ops import LedgerRow
from effective.sqlite import SqliteApp, SqliteLedger, TaskSnapshot
```

<!-- source: examples/first_workflow.py -->
```python
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
```

A workflow is a generator that does no I/O. Each `yield from` hands the handler a description of an
effect, and the handler decides what it means. `Effect[float | Repair]` is the type of such a
generator: it yields ops and returns a `float` or a `Repair`.

| line | what it does |
|---|---|
| `t"…"` | a PEP 750 template. `render` walks it: `{request}` is an input, rendered into the prompt; `{celsius}` is a `Gated` **output channel**, declaring a field the response must carry and a check it must pass. A channel is named by what is inside its braces, and the channels must match the fields of `output=` one to one: rename the variable to `temp`, or give `Setpoint` a field no channel fills, and `render` raises `ChannelMismatchError` |
| `ask_llm("setpoint", setpoint.messages, dict)` | one model step. The first argument is the step's **name**, the identity replay matches on. A name like `setpoint` or `set_point.v2` is one atom; the handler refuses one that is not, such as `water plan`, when it mints the step's key, and its error lists the forms an atom takes. The last is the type of the raw response. It is shorthand for `step("setpoint", AskLLM(setpoint.messages, dict))` |
| `setpoint.resolve(response)` | parses the response into a `Setpoint`, or returns a `Repair` whose `reason` says what was wrong: a missing field, a bad value, a failed check, or a response that is not a mapping of fields at all. The `reason` is the text a re-prompt would send back |
| `call_tool("thermostat", {"celsius": target}, str)` | one tool step: the tool's name, its arguments, and the type of its result. Its step is named `tool:` plus the tool, so it is shorthand for `step("tool:thermostat", CallTool(name="thermostat", result_schema=str, args={"celsius": target}))` |
| `append_ledger(LedgerRow(…))` | one row on the append-only ledger. `LedgerRow` is a pydantic model, and fields past `event_id` and `kind`, like `celsius`, ride on the row |
| `compose_key(t"setpoint:{Subject(request_id)}")` | the row's id. A bare `str` is refused because it could carry a separator and make two ids collide. A marker says what the coordinate is: `Subject` for the domain's own value, `Run` for a run id, `Name` for a position, `Index` for a counter over re-executions of ONE position, `Ordinal` for a counter over distinct positions of one kind. The last two read alike and a view treats them oppositely: it folds the `Index` away and keeps the `Ordinal`. Each marks one atom, such as `r1` or `bed-1`, and `compose_key` refuses what is not one, such as `a:b`, `007` or `2026-09-11`. A second coordinate follows a comma: `t"setpoint:{Name(house)},{Index(n)}"` |
| `case unreachable: assert_never(…)` | makes `ty`, the type checker the repo's gate runs, fail if `resolve` ever gains a third outcome |

Nothing between two yields may do I/O or read the clock or `random`.
`uv run python -m effective.lint path/to/your_workflow.py` catches the spellings it knows: a bare
`yield`, `time.time()`, `datetime.now()`, `random`, and imports of I/O modules such as `socket`. It
misses `open`, `print`, and some clock calls such as `time.perf_counter()`, so keep those out by
hand. It checks every function in the file, `main` included, so a driver that reads the clock
belongs in its own module. `just lint` runs the same check over the workflow files registered in
`WORKFLOW_ROLE_SRCS` (in `src/effective/lint.py`), and this example is one of them.

## Running it durably

<!-- source: examples/first_workflow.py -->
```python
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
```

`House` is the world the ops reach: a model that always gives the same response, and a thermostat
that can be offline. `MeteredInterpreter` connects the two to the op kinds: `llm` interprets an
`AskLLM` with the response and its token usage, and `tools` interprets a `CallTool`. `run_durably`
registers the workflow as a task on `SqliteApp`, spawns it, and runs it until it ends.

| output line | what happened |
|---|---|
| `run` | `DurableHandler` runs each op as a checkpointed step of the task, so the file records each result before the workflow sees it. The keys printed are those checkpoints: `step:setpoint`, `step;tool:thermostat`, and the ledger row's `ledger;setpoint:r1`, where `;` joins the terms of one key |
| `resume` | the thermostat is offline on the first attempt, so the attempt fails and the engine retries the task: two attempts. The retry runs the workflow from the top, the file serves `setpoint` its recorded result, and only the thermostat call runs again: the model was asked once. Nothing is captured from a frame; suspension is replay |
| `guardrail` | the model responds with 45°C, the channel's check fails, and the workflow returns the `Repair` before reaching the thermostat. A task's result is stored as JSON, so the `Repair` comes back as its fields |

The workflow never names its handler. `examples/testing_a_workflow.py` runs the same
`set_temperature` under the two test handlers, with canned results and then a replay that refuses
a changed program; `wiki/concepts/testing.md` says what each is for.

## What you imported

| names | module | what they are |
|---|---|---|
| `Effect` | `effective` | the type of a workflow: a generator that yields ops and returns a result |
| `ask_llm`, `call_tool`, `append_ledger`, `step` | `effective` | the author surface: typed wrappers you `yield from` |
| `compose_key`, `Subject`, `Run`, `Name`, `Index`, `Ordinal` | `effective` | identities built from t-strings, and what each coordinate means |
| `MeteredInterpreter`, `Usage` | `effective` | what interprets a model call and a tool call, and a model call's token usage |
| `Gated`, `Repair`, `render` | `effective.channels` | the data axis: typed prompt channels |
| `ChannelMismatchError` | `effective.channels` | what `render` raises when the channels and the fields of `output=` disagree |
| `LedgerRow` | `effective.ops` | the typed shape of a ledger append |
| `AskLLM`, `Judge`, `CallTool` | `effective.domain` | what a step carries; `ask_llm`, `judge` and `call_tool` build them for you |
| `DurableHandler` | `effective.handlers.absurd` | the handler that runs each op as a checkpointed step, on either engine |
| `SqliteApp`, `SqliteLedger`, `TaskSnapshot` | `effective.sqlite` | the embedded engine: one file, no service; its ledger; a task's ending |
| `keys`, `read_sqlite_task` | `effective.checkpoints` | a task's checkpoints, read back in commit order |
| `OpLayer` | `effective.layers` | the type of a layer, the hooks of `examples/hooks_as_layers.py` |

## Where next

| to | read |
|---|---|
| the model in ten minutes: generator, handler, tape, layers, combinators | [intro.md](intro.md) |
| the op set, the author surface, and why resume is replay | [effective-101.md](effective-101.md) §2, §3 and §5.3 |
| park for a human with `await_event`, resumed by replay | [effective-101.md](effective-101.md) §5.3, and `await_event` in `src/effective/api.py` |
| testing a workflow | `wiki/concepts/testing.md` |
