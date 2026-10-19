"""Find the real sleeps: waits that name a duration where they could name what they wait for.

Every wait in Effective waits for something: a delivered event, a time, another thread's work, or
a retry, and each has a mechanism that names it. A sleep names only a duration, so a test pays it
on every run and it is wrong under load. The gate fails on a file whose sleeps outnumber its
baseline count, and on a count nothing produces, so the record can only shrink.

| a reference to | flagged |
|---|---|
| `time.sleep` or `asyncio.sleep`, by attribute or imported name, called or passed | yes |
| a sleep called with the literal `0` | no: one turn of the scheduler, outside a loop |
| `.wait(t)` on an `Event()` built in the same expression | yes |
| `.get(timeout=t)` on a `Queue()` built in the same expression | yes |
| `select([], [], [], t)`, by attribute or imported name | yes |

Unscanned: a fresh object bound to a name and waited on later, `asyncio.wait_for` over a fresh
`Event`, a shell `sleep` inside a command string, and `loop.call_later`.

    uv run python scripts/sleep_sweep.py src tests scripts examples research
    uv run python scripts/sleep_sweep.py --gate src tests scripts examples research
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
from collections import Counter
from collections.abc import Iterator

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from scripts.fstring_sweep import shipping

BASELINE = pathlib.Path("scripts/sleep-baseline.txt")
PRIVATE_BASELINE = pathlib.Path("scripts/sleep-private-baseline.txt")
TIMED = {"time": "sleep", "asyncio": "sleep", "select": "select"}
WAITING = {"Event": "wait", "Queue": "get"}
SKIPPED = frozenset({".venv", "build", "node_modules", "__pycache__"})

Baseline = dict[str, tuple[int, str]]


def _functions(tree: ast.Module) -> dict[str, str]:
    """Each spelling of a timed function in `tree` (`time.sleep`, `nap`, `select`), mapped to the
    function's module."""
    modules: dict[str, str] = {}
    spelled: dict[str, str] = {}
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=aliases):
                modules |= {a.asname or a.name: a.name for a in aliases if a.name in TIMED}
            case ast.ImportFrom(module=str(module), names=aliases) if module in TIMED:
                spelled |= {a.asname or a.name: module for a in aliases if a.name == TIMED[module]}
    return spelled | {f"{alias}.{TIMED[module]}": module for alias, module in modules.items()}


def _spelling(node: ast.AST) -> str | None:
    match node:
        case ast.Name(id=name, ctx=ast.Load()):
            return name
        case ast.Attribute(value=ast.Name(id=name), attr=attr):
            return f"{name}.{attr}"
    return None


def _built(node: ast.expr, kind: str) -> bool:
    """Whether `node` constructs a `kind` (`Event()`, `threading.Event()`, `queue.Queue()`)."""
    match node:
        case ast.Call(func=ast.Name(id=name)) | ast.Call(func=ast.Attribute(attr=name)):
            return name == kind
    return False


def _timed_on_a_fresh_object(node: ast.AST) -> bool:
    match node:
        case ast.Call(func=ast.Attribute(value=value, attr=attr), args=args, keywords=keywords):
            timed = bool(args) or any(k.arg == "timeout" for k in keywords)
            return timed and any(attr == m and _built(value, k) for k, m in WAITING.items())
    return False


def _waits_on_nothing(call: ast.Call | None) -> bool:
    match call:
        case ast.Call(args=[ast.List(elts=[]), ast.List(elts=[]), ast.List(elts=[]), _]):
            return True
    return False


def _yields_a_turn(call: ast.Call | None, looping: set[ast.AST]) -> bool:
    match call:
        case ast.Call(args=[ast.Constant(value=0), *_]):
            return call not in looping
    return False


def sleeps(source: str) -> Iterator[int]:
    """The line of each real sleep in `source`."""
    tree = ast.parse(source)
    functions = _functions(tree)
    looping = {
        inner
        for loop in ast.walk(tree)
        if isinstance(loop, ast.While | ast.For | ast.AsyncFor)
        for inner in ast.walk(loop)
    }
    calls = {node.func: node for node in ast.walk(tree) if isinstance(node, ast.Call)}
    for node in (node for node in ast.walk(tree) if isinstance(node, ast.expr)):
        if _timed_on_a_fresh_object(node):
            yield node.lineno
        elif (spelling := _spelling(node)) is None or (module := functions.get(spelling)) is None:
            continue
        elif module == "select":
            if _waits_on_nothing(calls.get(node)):
                yield node.lineno
        elif not _yields_a_turn(calls.get(node), looping):
            yield node.lineno


def census(roots: list[str]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for root in roots:
        top = pathlib.Path(root)
        for path in [top] if top.is_file() else sorted(top.rglob("*.py")):
            if SKIPPED.isdisjoint(path.parts):
                counts[path.as_posix()] += sum(1 for _ in sleeps(path.read_text()))
    return +counts


def load_baseline() -> Baseline:
    entries: Baseline = {}
    for file in (BASELINE, PRIVATE_BASELINE):
        if file.exists():
            for line in file.read_text().splitlines():
                if line.strip() and not line.startswith("#"):
                    path, count, reason = (part.strip() for part in line.split(" :: ", 2))
                    entries[path] = (int(count), reason)
    return entries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--gate", action="store_true")
    args = ap.parse_args()
    live, baseline, ships = census(args.paths), load_baseline(), shipping()
    scanned = tuple(pathlib.Path(p).as_posix().rstrip("/") for p in args.paths)
    grown = {p: n for p, n in live.items() if n > baseline.get(p, (0, ""))[0]}
    shrunk = {
        p: n for p, (n, _) in baseline.items() if p.startswith(scanned) and n > live.get(p, 0)
    }
    for path, n in sorted(grown.items()):
        home = BASELINE if ships(path) else PRIVATE_BASELINE
        print(f"{path}: {n} sleeps, {baseline.get(path, (0, ''))[0]} in {home}")
    for path, n in sorted(shrunk.items()):
        print(f"{path}: {n} baselined, {live.get(path, 0)} found; lower the count")
    print(f"sleep-sweep: {sum(live.values())} sleeps in {len(live)} files, {len(grown)} grown,")
    print(f"  {len(shrunk)} shrunk; a shrunk entry is a count to lower to what was found")
    return 1 if args.gate and (grown or shrunk) else 0


if __name__ == "__main__":
    raise SystemExit(main())
