"""Run the coder on a project directory.

    python -m examples.coder DIR "fix the failing test" [--model M] [--budget USD] [--write]
        [--skills PACK --skill NAME ...]

The directory's text files are copied in; the run works on the copy in a durable SQLite store at
`DIR/.coder/runs.db`, with spans beside it. A gate refuses every op once the run's measured spend
reaches `--budget`. Each `--skill` is activated once from the pack at `--skills` and rendered into
every visit. It prints how the run stopped, its cost, and a diff of what it committed. `--write`
copies the committed files back into DIR. `OPENAI_API_KEY` is read from the environment.
"""

import argparse
import difflib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import openai

from effective.budget import MeasuredBudget
from effective.budget import as_policy as budget_policy
from effective.checkpoints import read_sqlite_conn
from effective.coding.tier import tree_paths
from effective.combinators import hoisted
from effective.cost import Contract, MeteredInterpreter
from effective.engines.sqlite import SqliteApp, SqliteLedger, TaskSnapshot
from effective.govern import govern
from effective.handlers.durable import DurableHandler
from effective.interpreters.openai import ResponsesTurnCaller, profile_for
from effective.interpreters.tool_catalog import strict_function_tools
from effective.keys.grammar import parse
from effective.machine.trampoline import changed_paths
from effective.skills import SkillRegistry
from effective.telemetry import otlp_jsonl_sink, render_message, traced
from examples.coder.machine import coder
from examples.coder.prompt import system_prompt
from examples.coder.tools import TOOLS, serving, text

TEXT_SUFFIXES = frozenset({".py", ".txt", ".md", ".cfg", ".toml", ".ini", ".json", ".rst"})
MAX_FILE_BYTES = 64_000


def copy_in(root: Path) -> dict[str, str]:
    """The project's text files by relative path, skipping dot-paths, large files and bytes that
    are not UTF-8.

    A symlink is skipped whatever it points at, and every file has to resolve to somewhere under
    the project. Otherwise the copy is where isolation is lost: a link in the directory reads a
    host file into the tree, and the container the tools run in never sees it, because the file is
    already in the prompt."""
    inside = root.resolve()
    tree: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if path.is_symlink() or not path.is_file():
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        if not path.resolve().is_relative_to(inside):
            continue
        if path.suffix not in TEXT_SUFFIXES or path.stat().st_size > MAX_FILE_BYTES:
            continue
        try:
            tree[relative.as_posix()] = path.read_text()
        except UnicodeDecodeError:
            continue
    tree_paths(tree)
    return tree


def next_run_id(app: SqliteApp) -> str:
    """The run's name: its ordinal in this store. Every key it enters says what it names, as in
    `machine:3`."""
    (count,) = app.conn.execute("SELECT COUNT(*) FROM tasks").fetchone()
    return str(count + 1)


def committed_tree(app: SqliteApp, task_id: Any) -> dict[str, str]:
    """The tree the postamble stored: the run's one artifact checkpoint."""
    for checkpoint in read_sqlite_conn(app.conn, task_id, exclude=()):
        if parse(checkpoint.key.stored()).terms[0].tag == "artifact":
            return dict(checkpoint.state)
    raise LookupError("the run committed no artifact")


def diff(before: Mapping[str, str], after: Mapping[str, str]) -> str:
    """A unified diff of every changed path, with `/dev/null` for the side a path is absent from,
    so a file created or removed empty still shows its header."""
    return "".join(_file_diff(before, after, path) for path in changed_paths(before, after))


def _header(side: str, path: str) -> str:
    """A header's file name. A path holding a control character, a quote or a backslash is written
    as a JSON string, so a newline in a model-chosen path cannot start a second header."""
    name = side + path
    if any(ord(c) < 0x20 or c in '"\\' for c in path):
        return json.dumps(name, ensure_ascii=False)
    return name


def _file_diff(before: Mapping[str, str], after: Mapping[str, str], path: str) -> str:
    old = _header("a/", path) if path in before else "/dev/null"
    new = _header("b/", path) if path in after else "/dev/null"
    lines = difflib.unified_diff(
        before.get(path, "").splitlines(keepends=True),
        after.get(path, "").splitlines(keepends=True),
        old,
        new,
    )
    return "".join(lines) or text(t"--- {old}\n+++ {new}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m examples.coder", description=__doc__)
    parser.add_argument("dir", type=Path)
    parser.add_argument("goal")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--budget", type=float, default=0.05, help="USD ceiling for model calls")
    parser.add_argument("--visits", type=int, default=3)
    parser.add_argument("--turns", type=int, default=12, help="turns per visit")
    parser.add_argument("--write", action="store_true", help="copy the committed files back")
    parser.add_argument("--skills", type=Path, help="a skill pack: one SKILL.md per directory")
    parser.add_argument("--skill", action="append", default=[], help="a skill from the pack")
    args = parser.parse_args(argv)
    if args.skill and args.skills is None:
        parser.error("--skill names a skill in the pack --skills gives")
    pack = SkillRegistry.in_memory({}) if args.skills is None else SkillRegistry.load(args.skills)

    seed = copy_in(args.dir)
    store = args.dir / ".coder"
    store.mkdir(exist_ok=True)
    app = SqliteApp(str(store / "runs.db"))
    try:
        return run_in(app, args, seed, store, pack)
    finally:
        app.close()


def run_in(
    app: SqliteApp,
    args: argparse.Namespace,
    seed: Mapping[str, str],
    store: Path,
    pack: SkillRegistry,
) -> int:
    """Run the coder once in `app`'s store, report how it stopped, and write back on `--write`."""
    run_id = next_run_id(app)
    profile = profile_for(args.model)
    caller = ResponsesTurnCaller(
        client=openai.OpenAI(),
        system_prompt=system_prompt(),
        model=args.model,
        price=profile.price,
        extra={"reasoning_effort": profile.reasoning_effort},
        tools=strict_function_tools(TOOLS),
    )
    interpreter = MeteredInterpreter(
        llm=caller,
        tools=serving(pack),
        domain_layers=[traced(otlp_jsonl_sink(store / "spans.jsonl"), session_id=run_id)],
    )

    @app.register_task("coder")
    def task(params: dict[str, Any], ctx: Any) -> Any:
        ledger = SqliteLedger(app.conn, params["run_id"], app.write_lock)
        spend = MeasuredBudget(run_id=params["run_id"], overall=args.budget, on_exhaust="fail")
        gate = govern(budget_policy(spend), gate="spend", run_id=params["run_id"])
        handler = DurableHandler(
            ctx, interpreter, ledger=ledger, op_layers=(gate,), contract=Contract.V1
        )
        return handler.run(
            lambda: hoisted(
                args.skill,
                lambda pins: coder(
                    params["run_id"],
                    params["goal"],
                    params["tree"],
                    visits=args.visits,
                    turns=args.turns,
                    pins=pins,
                ),
            )
        )

    task_id = app.spawn("coder", {"run_id": run_id, "goal": args.goal, "tree": seed})
    snapshot = app.run_until_result(task_id)
    cost = interpreter.meter.cost
    match snapshot:
        case None:
            print(render_message(t"{run_id}: parked ${cost:.4f}"), file=sys.stderr)
            return 1
        case TaskSnapshot(state="completed"):
            pass
        case TaskSnapshot(state=state, failure=failure):
            print(render_message(t"{run_id}: {state} ${cost:.4f} {failure}"), file=sys.stderr)
            return 1
    result = snapshot.result
    print(render_message(t"{run_id}: {result} ${cost:.4f}"))
    tree = committed_tree(app, task_id)
    print(diff(seed, tree), end="")
    if args.write:
        for path, content in tree.items():
            if seed.get(path) != content:
                (args.dir / path).parent.mkdir(parents=True, exist_ok=True)
                (args.dir / path).write_text(content)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
