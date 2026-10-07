"""Sweep the `[sqlite]` conformance lane over shuffle seeds and report which ones fail.

A DIAGNOSTIC, not a gate — nothing in `just check` runs it, and **a green sweep is weak
evidence**. Read that sentence before using this to decide anything.

It is kept for the end-to-end view: whether the *conformance tests* still flake, as opposed
to whether the seam they sit on is sound. For the seam itself, the instrument with teeth is
`tests/test_sqlite_concurrency.py`, which drives the shared connection directly and moves
hundreds of exceptions and hundreds of wrong answers to zero. (No exact figure, in this file
least of all: three runs of one probe gave 731/552, 740/936 and 781/904. A script whose whole
argument is that these counts drift must not quote four of their digits.) Reach for that
first; this is the slow, blunt confirmation afterwards.

**Why it is blunt.** The flake looks *deterministic per shuffle seed* and is not: a seed that
failed 10/10 in fresh subprocesses gave 5/5, then 0/6, with no code change. The rate drifts
with machine state. So:

- Failures here are real; **absences are not** informative at these sample sizes.
- Do not pin a seed as a regression test. Adding any test reshuffles the lane.

Three things that are *not* the trigger, ruled out the same day so nobody re-derives them:

- **Not the ordering.** Replaying a failing seed's order verbatim through explicit node ids
  under `-p no:randomly` collects byte-identically and passes.
- **Not pytest-randomly's per-test reseeding** (`--randomly-dont-reorganize` with reseeding
  on: passes). Nothing on the path calls `random`.
- **Not in-process-observable.** Repeated `pytest.main()` in one interpreter reported the
  flake as absent where fresh subprocesses reproduced it. Hence one subprocess per iteration
  — do not "optimize" the fork away — and hence `uv run pytest` rather than
  `python -m pytest`, whose extra `sys.path` entry changes the import set enough to matter.

Usage::

    uv run python scripts/sqlite_flake_sweep.py                 # seeds 1..60, one pass
    uv run python scripts/sqlite_flake_sweep.py --seeds 31,47 --repeat 10
    uv run python scripts/sqlite_flake_sweep.py --seeds 1-200 --jobs 4

`--jobs` above 1 changes the timing it is trying to measure, and observably suppresses the
failure rate. Use it to cover more seeds, not to estimate a rate.

Exit status is 1 if any iteration failed.
"""

import argparse
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# The first `E ` line of a pytest failure — enough to tell the flake's faces apart
# (`InterfaceError`, a wrong-result `TypeError`, a double-fire count assertion).
_FACE = re.compile(r"^E\s+(\S.*)$", re.M)
_FAILED = re.compile(r"^FAILED (\S+)", re.M)


def _parse_seeds(spec: str) -> list[int]:
    """`31,47` or `1-200` or a mix of both."""
    seeds: list[int] = []
    for part in spec.split(","):
        if "-" in part.strip("-"):
            lo, hi = part.split("-", 1)
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def run_once(seed: int) -> tuple[bool, list[str], str]:
    """One lane run in its OWN process. Returns (failed, failing node ids, face).

    ``uv run pytest`` rather than ``sys.executable -m pytest``: the ``-m`` form puts the cwd
    on ``sys.path``, which changes the imported module set, and the module set is part of
    what sets the race window. Measured — the ``-m`` form reported 0/5 on a seed the console
    invocation reproduced 5/5 on."""
    proc = subprocess.run(
        [
            "uv",
            "run",
            "pytest",
            "-k",
            "sqlite",
            "-q",
            "--no-header",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "-p",
            "randomly",
            f"--randomly-seed={seed}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return False, [], ""
    out = proc.stdout + proc.stderr
    faces = _FACE.findall(out)
    return True, _FAILED.findall(out), faces[0].strip() if faces else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", default="1-60", help="e.g. 1-60 or 31,47 (default: 1-60)")
    ap.add_argument("--repeat", type=int, default=1, help="iterations per seed (default: 1)")
    ap.add_argument("--jobs", type=int, default=1, help="concurrent subprocesses (default: 1)")
    args = ap.parse_args()

    seeds = _parse_seeds(args.seeds)
    work = [s for s in seeds for _ in range(args.repeat)]
    print(f"{len(work)} iterations over {len(seeds)} seed(s), {args.jobs} job(s)", flush=True)

    per_seed: Counter[int] = Counter()
    faces: Counter[str] = Counter()
    tests: Counter[str] = Counter()

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for seed, (failed, nodes, face) in zip(work, pool.map(run_once, work), strict=True):
            if failed:
                per_seed[seed] += 1
                faces[face] += 1
                tests.update(nodes)
                print(f"  seed {seed}: FAILED — {face[:100]}", flush=True)

    total = sum(per_seed.values())
    print(f"\n{total} / {len(work)} iterations failed")
    for seed, n in sorted(per_seed.items()):
        print(f"  seed {seed}: {n} / {args.repeat}")
    for name, n in tests.most_common():
        print(f"  {n:4d}  {name}")
    for face, n in faces.most_common():
        print(f"  {n:4d}  face: {face[:110]}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
