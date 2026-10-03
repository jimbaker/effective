# Your first workflow

One file, run four ways, with no model and no database. `examples/first_workflow.py` is a house
agent: it asks a model for a setpoint, sets the thermostat, and records what it did. Running it
records a run, replays it without the model, shows a guardrail refusing a bad answer, and shows
replay refusing a program whose steps changed.

## Run it

You need [uv](https://docs.astral.sh/uv/), which installs the pinned Python 3.14 for you. From the
repo root:

```bash
uv sync
uv run python examples/first_workflow.py
```

The first `uv sync` installs the whole development environment, so give it a minute or two. Then:

```text
record:    22.0 via step:setpoint step;tool:thermostat ledger;setpoint:r1
replay:    22.0 with no model and no thermostat
guardrail: Repair(reason='celsius outside the allowed range') via step:setpoint
drift:     ReplayMismatch: workflow ended after 1 ops but 3 were recorded
```

`uv run pytest --no-cov tests/test_first_workflow.py` pins each of those lines, and this block.

## The workflow

<!-- splice: examples/first_workflow.py::imports -->
```python
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
```

<!-- splice: examples/first_workflow.py::Setpoint,set_temperature -->
```python
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
```

A workflow is a generator that does no I/O. Each `yield from` hands the handler a description of an
effect, and the handler decides what it means. `Effect[float | Repair]` is the type of such a
generator: it yields ops and returns a `float` or a `Repair`.

| line | what it does |
|---|---|
| `t"…"` | a PEP 750 template. `render` walks it: `{request}` is an input, rendered into the prompt; `{celsius}` is a `Gated` **output channel**, declaring a field the answer must carry and a check it must pass. A channel is named by what is inside its braces, and the channels must match the fields of `output=` one to one: rename the variable to `temp`, or give `Setpoint` a field no channel fills, and `render` raises `ChannelMismatchError` |
| `ask_llm("setpoint", setpoint.messages, dict)` | one model step. The first argument is the step's **name**, the identity replay matches on. A name like `setpoint` or `set_point.v2` is one atom; the handler refuses one that is not, such as `water plan`, when it mints the step's key, and its error lists the forms an atom takes. The last is the type of the raw answer. It is shorthand for `step("setpoint", AskLLM(setpoint.messages, dict))` |
| `setpoint.resolve(answer)` | parses the answer into a `Setpoint`, or returns a `Repair` whose `reason` says what was wrong: a missing field, a bad value, a failed check, or an answer that is not a mapping of fields at all. The `reason` is the text a re-prompt would send back |
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

## Running it four ways

<!-- splice: examples/first_workflow.py::RESPONSES,main -->
```python
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
```

| output line | what happened |
|---|---|
| `record` | `RecordingHandler` answers each op from `RESPONSES`, keyed by the op's name: `setpoint` for the model step, and `tool:thermostat` for the tool. `run` takes a zero-argument function that builds the generator, hence each `lambda`. The trace prints each op's full key, where `;` joins the terms of one key: `step:setpoint`, `step;tool:thermostat`, and the ledger row's `ledger;setpoint:r1`. `recorder.ledger` holds the row itself |
| `replay` | `ReplayHandler` takes the trace and no responses. It re-executes the workflow and serves each op its recorded result, which is how a durable engine resumes a run after a crash. The workflow never names its handler, so swapping one needs no edit to it |
| `guardrail` | the model answers 45°C, the channel's check fails, and the workflow returns the `Repair` before reaching the thermostat. This handler has no canned thermostat answer, so reaching the tool would have raised |
| `drift` | lowering `ceiling` to 20 makes the same recorded answer fail the check, so the program stops after one op. Replay compares that with the three recorded ops and raises `ReplayMismatch`: the history no longer describes this program. Replay matches ops by name and order, so a changed prompt, tool argument, result type, or ledger field other than the row's id replays silently, served the recorded result |

## What you imported

| names | module | what they are |
|---|---|---|
| `Effect` | `effective` | the type of a workflow: a generator that yields ops and returns a result |
| `ask_llm`, `call_tool`, `append_ledger`, `step` | `effective` | the author surface: typed wrappers you `yield from` |
| `RecordingHandler`, `ReplayHandler`, `ReplayMismatch` | `effective` | two interpreters, and replay's refusal |
| `compose_key`, `Subject`, `Run`, `Name`, `Index`, `Ordinal` | `effective` | identities built from t-strings, and what each coordinate means |
| `Gated`, `Repair`, `render` | `effective.channels` | the data axis: typed prompt channels |
| `ChannelMismatchError` | `effective.channels` | what `render` raises when the channels and the fields of `output=` disagree |
| `LedgerRow` | `effective.ops` | the typed shape of a ledger append |
| `AskLLM`, `Judge`, `CallTool` | `effective.domain` | what a step carries; `ask_llm`, `judge` and `call_tool` build them for you |

## Where next

| to | read |
|---|---|
| the op set, the author surface, and why resume is replay | [effective-101.md](effective-101.md) §2, §3 and §5.3 |
| park for a human with `await_event`, resumed by replay | [effective-101.md](effective-101.md) §5.3, and `await_event` in `src/effective/api.py` |
