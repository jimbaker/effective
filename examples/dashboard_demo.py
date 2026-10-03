"""dashboard_demo: a seeded engine plus the run dashboard, in one process.

Run it: `just dashboard` (or `uv run examples/dashboard_demo.py`), then open the URL it prints.

`effective.dashboard` ships no entry point. A drain is `work_batch` run *in a process that holds
the task registry*, and the registry is workflows, which are domain code. So the substrate offers
`dashboard(engine)`, and a consumer such as this example supplies the workflows. The whole host is
the last four lines of `main`.

The engine is seeded with three runs, chosen so the page shows more than the happy path:

| run       | state                                                                          |
|-----------|--------------------------------------------------------------------------------|
| `review`  | parked on an approval: the amber hexagon you click. Answer it                  |
|           | (`{"decision": "approve"}` or `"reject"`) and the run resumes in the same      |
|           | request; the tail differs, so the graph's shape depends on the answer          |
| `settled` | finished: a graph with no pending node                                         |
| `batch`   | parked inside a `gather`. A gather reports ONE parked branch of N, and the     |
|           | page says so: parked branches wake one at a time, so answering this one yields |
|           | a *new* park. Answer it twice to watch that happen                             |

The database is fresh on every start unless you pass `--db`, so the starting state is
predictable: answer whatever you like and restart to get it back.
"""

import argparse
from pathlib import Path

import uvicorn

from effective.api import append_ledger, ask_llm, await_event, call_tool, gather
from effective.cost import MeteredInterpreter, Usage
from effective.dashboard import dashboard
from effective.handlers.absurd import DurableHandler
from effective.keys import Key, Run, compose_key
from effective.ops import LedgerRow
from effective.sqlite import SqliteApp, SqliteLedger


def review_wf(run_id: str):
    """Extract, record, then park for a human. The tail depends on the answer."""

    def workflow():
        yield from ask_llm("extract", [], dict)
        yield from append_ledger(LedgerRow(event_id=Key.parse("extracted"), kind="extracted"))
        approval = yield from await_event(compose_key(t"run-review:{Run(run_id)}"), dict)
        decision = approval.get("decision", "reject")
        yield from append_ledger(
            LedgerRow(event_id=Key.parse("reviewed"), kind="reviewed", decision=decision)
        )
        if decision == "approve":
            yield from append_ledger(LedgerRow(event_id=Key.parse("committed"), kind="committed"))
        return decision

    return workflow


def settled_wf(run_id: str):
    """No park anywhere — the contrast case."""

    def workflow():
        yield from ask_llm("extract", [], dict)
        yield from call_tool("classify", {}, int)
        yield from append_ledger(LedgerRow(event_id=Key.parse("settled"), kind="settled"))
        return "settled"

    return workflow


def batch_wf(run_id: str):
    """Three branches, two of them parked on different events — the L-1 shape."""

    def parked(i: int):
        def branch():
            return (yield from await_event(f"item{i}:{run_id}", dict))

        return branch

    def worked():
        def branch():
            return (yield from call_tool("classify", {}, int))

        return branch

    def workflow():
        return (yield from gather([parked(0), worked(), parked(2)]))

    return workflow


def _domain() -> MeteredInterpreter:
    """Canned, so the demo is deterministic and needs no API key."""
    return MeteredInterpreter(
        llm=lambda _op: ({"room": "kitchen", "celsius": "21.0"}, Usage()),
        tools=lambda _op: 1,
    )


def seed(engine: SqliteApp) -> None:
    """Register the three workflows and run each to its natural stopping point."""
    for name, factory in (("review", review_wf), ("settled", settled_wf), ("batch", batch_wf)):

        def task(params, ctx, _factory=factory):
            run_id = params["run_id"]
            ledger = SqliteLedger(engine.conn, run_id, engine.write_lock)
            return DurableHandler(ctx, _domain(), ledger=ledger).run(_factory(run_id))

        engine.register_task(name)(task)

    for name in ("review", "settled", "batch"):
        engine.run_until_result(engine.spawn(name, {"run_id": f"r-{name}"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--db",
        default=":memory:",
        help="engine database (default: in-memory, so the demo state is fresh every start)",
    )
    args = parser.parse_args()

    if args.db != ":memory:":
        Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    engine = SqliteApp(args.db)
    seed(engine)

    # `flush=True` because this is the whole point of the recipe: piped or captured, the URL has
    # to appear BEFORE uvicorn blocks, not when the buffer happens to drain.
    print(f"\n  effective dashboard → http://127.0.0.1:{args.port}\n", flush=True)
    uvicorn.run(dashboard(engine), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
