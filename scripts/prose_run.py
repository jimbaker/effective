"""Drive the de-essaying machine over one region, and serve its one mechanical gate.

    uv run python scripts/prose_run.py start src/effective/keys.py --subject keys-module
    uv run python scripts/prose_run.py drain --watch
    uv run python scripts/prose_run.py status

**Three commands because there are three PROCESSES, not because a CLI wants verbs.** `start`
spawns the run and takes the before-snapshot. `drain` claims and advances whatever is ready — the
half the terminal run view structurally cannot do, since it registers no task and
`SqliteApp._claim` refuses a name it did not register. `status` is a read, and exists so a driver
can tell "parked, waiting on me" from "ready, waiting on a drain" without opening a database.

**The gate tool records its own DOMAIN.** `CommandRun.output` names which checks actually ran, so
a green verdict on the tape says what it was green ON. The fast set omits `just check` (a
69-second serial suite is not a per-visit gate) and `--full` puts it back. A measurement whose
scope is not on the record licenses a claim over a domain nobody can see, and the fix is one line
of output rather than a rule someone remembers.
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from effective.domain import CallTool, DomainOp
from effective.engines.sqlite import SqliteApp, SqliteLedger
from effective.handlers.durable import DurableHandler
from effective.keys import Run
from effective.machine.evidence import CommandRun, Predicate
from effective.machine.trampoline import run_machine
from effective.prose import CALLERS_TOOL, VERIFY_TOOL, State, build_prose_specs, transition

TASK = "prose"
DEFAULT_DB = "build/prose.db"
SNAPSHOT = "build/prose-snapshot.json"

PROSE_PREDICATE = Predicate(VERIFY_TOOL, CommandRun)
"""The postamble measures the same thing VERIFY does.

Declared rather than defaulted: the default is the coding machine's `run_suite`, and a prose
deployment that does not serve it would reach its own commitment point and be asked for a pytest
runner — the leak `test_machine_second_embodiment` exists to catch."""


def _run(argv: list[str]) -> tuple[int, str]:
    done = subprocess.run(argv, capture_output=True, text=True, check=False)
    return done.returncode, (done.stdout + done.stderr).strip()


def gates(target: str, snapshot: str, *, full: bool) -> CommandRun:
    """The mechanical checks, folded into one record.

    ORDER IS THE CONTENT, and it mirrors `verdict_for_verify`. The skeleton runs first among the
    failing checks because "this edit moved code" is not a smaller version of "this line is too
    long" — one is undone, the other is reworded, and a fold that lost the difference would send a
    driver to tidy prose around a change it should be reverting.

    A missing snapshot is a `collection_error` rather than a failure: nothing has been said about
    the prose, which is exactly what BROKEN_ENV means."""
    if not Path(snapshot).exists():
        return CommandRun(exit_code=2, collection_error=f"no snapshot at {snapshot} — run `start`")

    ran: list[str] = []
    failures: list[str] = []

    code, out = _run([sys.executable, "scripts/prose_skeleton.py", "--check", snapshot, target])
    ran.append("skeleton+fingerprint")
    if code != 0:
        failures.append(f"skeleton: {out.splitlines()[0] if out else 'changed'}")

    for label, argv in (
        ("ruff-check", ["uv", "run", "ruff", "check", target]),
        ("ruff-format", ["uv", "run", "ruff", "format", "--check", target]),
        ("ty", ["uvx", "ty", "check", target]),
    ):
        if shutil.which(argv[0]) is None:
            return CommandRun(exit_code=2, collection_error=f"{argv[0]} is not on PATH")
        code, out = _run(argv)
        ran.append(label)
        if code != 0:
            failures.append(f"{label}: {out.splitlines()[0] if out else 'failed'}")

    if full:
        code, out = _run(["just", "check"])
        ran.append("just-check")
        if code != 0:
            failures.append(f"just-check: exit {code}")

    return CommandRun(
        exit_code=1 if failures else 0,
        failures=tuple(failures),
        output=f"gates run: {', '.join(ran)}",
    )


def callers(target: str, line: str) -> CommandRun:
    """Who uses the unit under the docstring — `whouses.py`, as a recorded op.

    The region's line is its OWN parameter and is not read out of `subject`, because a subject is
    a `Segment` and a `Segment` refuses `:` — the delimiter a `path:line` address needs. Packing
    the two into one string would have been the Bobby Tables shape on the identity axis, arriving
    in the machine whose keys that rule protects.

    A line that resolves to no `def` or `class` is a `collection_error` rather than an empty
    result: "nobody calls this" and "I asked the wrong question" are different answers, and only
    the second is the tool's fault."""
    if not line.isdigit():
        return CommandRun(
            exit_code=2, collection_error=f"region line {line!r} is not a line number"
        )
    code, out = _run([sys.executable, "scripts/whouses.py", f"{target}:{line}", "src", "tests"])
    if code != 0:
        return CommandRun(exit_code=2, collection_error=out.splitlines()[-1] if out else "failed")
    return CommandRun(exit_code=0, output=out)


class ProseDeployment:
    """What a de-essaying deployment serves — and nothing else.

    It refuses an unknown tool rather than answering harmlessly, for the reason the docs
    embodiment does: a harmless answer is how a substrate leak stays invisible."""

    def __init__(self, target: str, snapshot: str, line: str, *, full: bool) -> None:
        self.target, self.snapshot, self.line, self.full = target, snapshot, line, full

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        match op.name:
            case name if name == VERIFY_TOOL:
                return gates(self.target, self.snapshot, full=self.full)
            case name if name == CALLERS_TOOL:
                return callers(self.target, self.line)
            case unserved:
                raise KeyError(f"a prose deployment does not serve {unserved!r}")


def _register(app: SqliteApp, *, full: bool) -> None:
    @app.register_task(TASK)
    def task(params, ctx):
        run_id, subject, target = params["run_id"], params["subject"], params["target"]
        deployment = ProseDeployment(target, params["snapshot"], str(params["line"]), full=full)
        return DurableHandler(
            ctx, deployment, ledger=SqliteLedger(app.conn, run_id, app.write_lock)
        ).run(
            lambda: run_machine(
                Run(run_id),
                params["goal"],
                build_prose_specs(subject),
                transition,
                start=State.SELECT,
                budget=params.get("budget", 24),
                tree={target: params.get("digest", "")},
                predicate=PROSE_PREDICATE,
            )
        )


def start(args) -> int:
    Path(args.db).parent.mkdir(parents=True, exist_ok=True)
    code, out = _run(
        [sys.executable, "scripts/prose_skeleton.py", "--save", args.snapshot, args.target]
    )
    if code != 0:
        print(f"could not snapshot {args.target}: {out}", file=sys.stderr)
        return 1
    print(out)

    app = SqliteApp(args.db)
    _register(app, full=args.full)
    task_id = app.spawn(
        TASK,
        {
            "run_id": args.run_id,
            "subject": args.subject,
            "target": args.target,
            "snapshot": args.snapshot,
            "goal": args.goal,
            "line": args.line,
            "budget": args.budget,
            "digest": Path(args.target).read_text()[:200],
        },
    )
    app.run_until_result(task_id)
    app.close()
    print(f"spawned {TASK} {task_id} on {args.target} (subject {args.subject!r})")
    print(f"\nopen it:  just tui {args.db}")
    return 0


def drain(args) -> int:
    app = SqliteApp(args.db)
    _register(app, full=args.full)
    batches = 0
    try:
        while True:
            if app.work_batch():
                batches += 1
                print(f"drained batch {batches}", flush=True)
            elif args.watch:
                time.sleep(args.interval)
            else:
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        app.close()
    print(f"drained {args.db}: {batches} batch(es)")
    return 0


def status(args) -> int:
    """Parked-waiting-on-me against ready-waiting-on-a-drain — the one read a driver needs before
    deciding whether to answer or to drain."""
    from effective.runs import read_sqlite_runs

    # `read_sqlite_runs`, not the column: `tasks.waiting_event` outlives the park it names, so a
    # raw read tells a driver to answer a task that is sleeping, retrying or finished. The module
    # exists for this read and normalizes the engine's word into a `RunState` besides.
    for run in read_sqlite_runs(args.db):
        print(f"  {run.task_name:8} {run.state:10} {run.task_id}  {run.waiting_on or ''}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=DEFAULT_DB)
    parser.add_argument("--snapshot", default=SNAPSHOT)
    parser.add_argument("--full", action="store_true", help="add `just check` to the gate set")
    subs = parser.add_subparsers(dest="cmd", required=True)

    s = subs.add_parser("start")
    s.add_argument("target")
    s.add_argument("--subject", required=True, help="a Segment-safe region name")
    s.add_argument("--line", required=True, help="the def/class line the region covers")
    s.add_argument("--run-id", default=None)
    s.add_argument("--goal", default="de-essay this region")
    s.add_argument("--budget", type=int, default=24)
    s.set_defaults(func=start)

    d = subs.add_parser("drain")
    d.add_argument("--watch", action="store_true")
    d.add_argument("--interval", type=float, default=1.5)
    d.set_defaults(func=drain)

    subs.add_parser("status").set_defaults(func=status)

    args = parser.parse_args()
    if getattr(args, "run_id", None) is None and args.cmd == "start":
        args.run_id = f"prose-{args.subject}"
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
