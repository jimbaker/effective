#!/usr/bin/env python
"""Run the RLM-vs-structured contrast pilot.

Resumable: a cell with a clean trial.json is skipped; errored cells re-run.
Reads OPENAI_API_KEY from the env or ./api.env — VALIDATED (the stale-env
gotcha). Stops when the cumulative campaign cost crosses --budget-total.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent.contrastbench import (
    ARMS,
    TrialResult,
    make_sem_tasks,
    make_tasks,
    probe_templates,
    run_trial,
    summarize,
)


def _load_key() -> str:
    from openai import OpenAI

    candidates: list[tuple[str, str]] = []
    if key := os.environ.get("OPENAI_API_KEY"):
        candidates.append(("env", key))
    api_env = Path(__file__).resolve().parent.parent / "api.env"
    if api_env.exists():
        for line in api_env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                candidates.append(("api.env", line.split("=", 1)[1].strip().strip('"')))
    for source, key in candidates:
        try:
            OpenAI(api_key=key).models.retrieve("gpt-5-nano")
        except Exception:
            print(f"OPENAI_API_KEY from {source} rejected; trying next source", file=sys.stderr)
            continue
        print(f"using OPENAI_API_KEY from {source}")
        return key
    sys.exit("no working OPENAI_API_KEY (checked env, api.env)")


def _execute(todo, done, args, make_client, out_dir: Path) -> None:
    """Run the pending cells in a pool; stop on the campaign budget ceiling."""
    spent = sum(t.cost_usd for t in done)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for task, arm, attempt in todo:
            fut = pool.submit(
                run_trial,
                task,
                arm,
                attempt,
                make_client=make_client,
                out_dir=out_dir,
                model=args.model,
                extra={"reasoning_effort": args.effort},
                budget_usd=args.budget_trial,
            )
            pending[fut] = (task.task_id, arm, attempt)
        for fut in as_completed(pending):
            tid, arm, attempt = pending[fut]
            try:
                trial = fut.result()
            except Exception as exc:
                print(f"  {tid}/{arm}#{attempt}: HARNESS ERROR {exc}", file=sys.stderr)
                continue
            done.append(trial)
            spent += trial.cost_usd
            mark = "PASS" if trial.reward >= 1.0 else ("ERR " if trial.error else "fail")
            print(
                f"  {tid}/{arm}#{attempt}: {mark} answer={trial.answer!r} "
                f"tok={trial.prompt_tokens}+{trial.completion_tokens} "
                f"${trial.cost_usd:.4f} {trial.elapsed_sec:.0f}s"
            )
            if spent > args.budget_total:
                print(f"campaign budget ${args.budget_total} crossed (${spent:.2f}); stopping")
                for f in pending:
                    f.cancel()
                return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--family",
        default="aggregation",
        choices=["aggregation", "semantic"],
        help="aggregation = contrast-1; semantic = contrast-2 (llm_query regime)",
    )
    ap.add_argument("--sizes", nargs="*", default=["S", "L"], choices=["S", "L"])
    ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--tasks", nargs="*", default=None, help="task_id filter")
    ap.add_argument("--attempts", type=int, default=2)
    ap.add_argument("--model", default="gpt-5-nano")
    ap.add_argument("--effort", default="low", help="reasoning_effort")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="benches/contrast/pilot")
    ap.add_argument("--budget-trial", type=float, default=0.10)
    ap.add_argument("--budget-total", type=float, default=5.0)
    args = ap.parse_args()

    key = _load_key()
    from openai import OpenAI

    def make_client():
        return OpenAI(api_key=key)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_tasks = make_sem_tasks(args.seed) if args.family == "semantic" else make_tasks(args.seed)
    tasks = [
        t
        for t in all_tasks
        if t.size in args.sizes and (args.tasks is None or t.task_id in args.tasks)
    ]
    if args.family == "semantic":
        # contrast-2 §2: the template pre-flight gates the campaign, pre-data.
        agreement, misses, usage = probe_templates(make_client(), model=args.model)
        print(f"template probe: agreement {agreement:.2%} (${usage.cost:.4f})")
        for truth, note, got in misses:
            print(f"  MISS [{truth} -> {got}] {note!r}")
        if agreement < 0.90:
            sys.exit("template agreement below the registered 0.90 gate — revise templates")
    cells = [
        (task, arm, attempt)
        for task in tasks
        for arm in args.arms
        for attempt in range(1, args.attempts + 1)
    ]

    done: list[TrialResult] = []
    todo = []
    for task, arm, attempt in cells:
        prior = out_dir / task.task_id / arm / f"attempt-{attempt}" / "trial.json"
        if prior.exists():
            data = json.loads(prior.read_text())
            if not data.get("error"):
                done.append(TrialResult(**data))
                continue
        todo.append((task, arm, attempt))
    print(f"{len(cells)} cells: {len(done)} already clean, {len(todo)} to run")

    _execute(todo, done, args, make_client, out_dir)

    # Rebuild the campaign file from disk, not from this invocation's cells — an
    # arm/task-filtered re-run must never clobber the other cells' rows.
    done = [
        TrialResult(**json.loads(p.read_text()))
        for p in sorted(out_dir.glob("*/*/attempt-*/trial.json"))
    ]
    (out_dir / "trials.json").write_text(json.dumps([asdict(t) for t in done], indent=1))
    spent = sum(t.cost_usd for t in done)
    print(f"\ncampaign cost: ${spent:.3f}  trials: {len(done)}")
    for row in summarize(done):
        print(
            f"{row['size']}/{row['arm']:<16} n={row['n']:<3} acc={row['accuracy']:.2f} "
            f"ptok={row['mean_prompt_tokens']:>9.0f} ctok={row['mean_completion_tokens']:>6.0f} "
            f"cost=${row['mean_cost_usd']:.4f} turns={row['mean_turns']:.1f} "
            f"sub={row['mean_subcalls']:.1f} "
            f"replay={row['replays_ok']}/{row['replays_run']} err={row['errors']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
