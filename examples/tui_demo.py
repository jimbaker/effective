"""tui_demo — a seeded store the terminal run view can be pointed at.

Run it: `just tui-demo`, which seeds `build/tui-demo.db` and prints the command to open it.

**Why this file exists at all.** `src/tui` ships no workflows, the same way `effective.dashboard`
ships no entry point and for the same reason: a run view is a *reader*, and something has to have
written a run first. That something is domain code, which is what an example is. `src/tui` holds
widgets; this holds three workflows.

**What you get** — three runs, chosen so the view tells the truth rather than only the happy path.

  1. `machine` — the shape `effective.coding` mints: `d:{n}` around `state:{name}` around the
     steps, parked on a review. This is the one route T exists for; the tree IS the trajectory,
     and the park is answerable from the input pane.
  2. `react` — a FLAT tape, finished. A ReAct loop carries no scope frames, so the tree draws a
     list — which is the honest picture, not a failure of the renderer. Press `p` and `project`
     recovers the structure the frames do not carry.
  3. `fanout` — two `gather` branches. The frames a gather mints, and the case a tree cannot draw:
     branches that interleaved on the tape group into two contiguous blocks, because grouping by
     containment is what a tree IS.

No model and no network: the domain answers from a table, so the store is byte-stable across runs
and an executable README can quote what it prints. The database is rebuilt on every start unless
you pass `--keep`, so the demo state is predictable — answer whatever you like and re-seed.
"""

import argparse
from pathlib import Path

from effective.api import append_ledger, await_event, call_tool, gather, scoped
from effective.cost import MeteredInterpreter, Usage
from effective.handlers.absurd import DurableHandler
from effective.keys import Index, Name, Run, compose_key
from effective.ops import LedgerRow
from effective.sqlite import SqliteApp, SqliteLedger

STATES = ("test", "draft", "review")
"""The states the demo machine visits, in order — one `state:` frame each.

Three rather than two so the tree has a level worth collapsing: `fold_cycles` drops `d:` (an
unrolling coordinate) and keeps `state:` (a naming one), and with a single state that distinction
is invisible."""


def machine_wf(run_id: str):
    """A coding-machine-shaped run: three visits, each a named state around two tools, then a park.

    The frames are minted exactly as `effective.coding` mints them — `d:{n}` outside,
    `state:{name}` inside — because the point of the demo is the real shape, not a shape that
    renders nicely. `effective/coding/__init__.py` says why both are needed: dropping either loses
    a distinction, and a re-entered state frame gets no occurrence suffix.
    """
    for visit, state in enumerate(STATES):
        yield from scoped(
            compose_key(t"d:{Index(visit)}"),
            lambda s=state, v=visit: _state(s, v),
        )
    yield from append_ledger(
        LedgerRow(
            event_id=compose_key(t"run-committed:{Run(run_id)}"),
            kind="machine-committed",
            files=["mod.py", "test_mod.py"],
        )
    )
    decision = yield from await_event(compose_key(t"run-review:{Run(run_id)}"), dict)
    return {"decision": decision}


def _state(name: str, visit: int):
    return (yield from scoped(compose_key(t"state:{Name(name)}"), lambda: _steps(name, visit)))


def _steps(name: str, visit: int):
    yield from call_tool("read_file", {"state": name, "visit": visit}, str)
    yield from call_tool("run_suite", {"state": name, "visit": visit}, str)
    return name


def react_wf(run_id: str):
    """A FLAT tape — a ReAct loop, which mints no scope frames at all.

    Here so the demo contains a run route T does NOT flatter. The tree draws a list because the
    tape is a list; `project` is what recovers the repeated position, and pressing `p` in the app
    is the whole demonstration.
    """
    for turn in range(4):
        yield from call_tool("think", {"turn": turn}, str)
        yield from call_tool("act", {"turn": turn}, str)
    yield from append_ledger(
        LedgerRow(event_id=compose_key(t"finished:{Run(run_id)}"), kind="react-finished")
    )
    return "done"


def fanout_wf(run_id: str):
    """Two concurrent branches under one `gather` — the frames a fan-out mints.

    The case worth having in a demo store: `gather:{g},{i};` is a frame like any other, so the
    tree nests it — and the tree therefore CANNOT show that the branches interleaved. On Absurd
    they interleave deterministically; on SQLite only as a race with overlapping branch work. The
    contiguous blocks the tree draws are a property of tree rendering, not of the run.
    """

    def branch(index: int):
        """`gather` takes zero-argument THUNKS returning generators.

        Not `GatherBranch` — that is the coordinate a handler assigns on the path to a park
        (`gather:{g},{i}`), which is a different thing wearing a similar name.
        """

        def thunk():
            value = yield from call_tool("fetch", {"branch": index}, str)
            yield from append_ledger(
                LedgerRow(
                    event_id=compose_key(t"fetched:{Run(run_id)},{Index(index)}"),
                    kind="fetched",
                )
            )
            return value

        return thunk

    results = yield from gather([branch(0), branch(1)])
    return list(results)


def _domain():
    """Answers from a table — no model, no network, so the store is byte-stable."""
    return MeteredInterpreter(
        llm=lambda _op: ({}, Usage()),
        tools=lambda op: f"{op.name}:{sorted(op.args.items())}",
    )


WORKFLOWS = (("machine", machine_wf), ("react", react_wf), ("fanout", fanout_wf))


def _register(app: SqliteApp) -> None:
    """Bind the three workflows to the engine — the REGISTRY, and the reason a drain is a process.

    A drain is not a button: it is `work_batch` *in a process that holds this*. The TUI holds no
    registry and registers nothing, so it structurally cannot claim a task — which is why
    answering a park there marks it claimable and stops. This function is the other half.
    """
    for name, workflow in WORKFLOWS:

        @app.register_task(name)
        def task(params, ctx, _wf=workflow):
            run = params["run_id"]
            return DurableHandler(
                ctx, _domain(), ledger=SqliteLedger(app.conn, run, app.write_lock)
            ).run(lambda: _wf(run))


def seed(db_path: Path) -> dict[str, str]:
    """Write the three runs. Returns `{name: task_id}` in the order they were spawned."""
    app = SqliteApp(str(db_path))
    _register(app)
    spawned: dict[str, str] = {}
    for name, _ in WORKFLOWS:
        task_id = app.spawn(name, {"run_id": f"r-{name}"})
        app.run_until_result(task_id)
        spawned[name] = str(task_id)
    app.close()
    return spawned


def drain(db_path: Path, batches: int = 8) -> int:
    """Claim and run whatever is ready — what the TUI deliberately will not do.

    `parked.answer` marks a task claimable and does NOT resume it: *"a drain must still run —
    nothing polls."* So after answering a park in the view, the run sits `ready` with its event
    recorded until something holding the registry claims it. Running that as a separate process is
    the drain/client split working, not a missing feature.
    """
    app = SqliteApp(str(db_path))
    _register(app)
    ran = 0
    try:
        for _ in range(batches):
            if not app.work_batch():
                break
            ran += 1
    finally:
        app.close()
    return ran


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="build/tui-demo.db", help="where to write the store")
    parser.add_argument(
        "--keep", action="store_true", help="reuse an existing store instead of rebuilding"
    )
    parser.add_argument(
        "--drain",
        action="store_true",
        help="do not seed; claim and run whatever is ready (after answering a park in the view)",
    )
    args = parser.parse_args()

    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if args.drain:
        print(f"drained {db_path}: {drain(db_path)} batch(es)")
        return 0
    if db_path.exists() and not args.keep:
        db_path.unlink()
    spawned = seed(db_path)

    print(f"seeded {db_path} with {len(spawned)} runs:")
    for name, task_id in spawned.items():
        print(f"  {name:8} {task_id}")
    print(f"\nopen it:  just tui {db_path}")
    print("          (0 = fanout, 1 = react, 2 = machine — newest first)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
