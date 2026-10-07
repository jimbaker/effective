"""Which tests reach a given source region? — reading `just cov-contexts`' output.

Coverage's *dynamic contexts* record, per executed line, WHICH TEST executed it, turning coverage
from a scalar ("95%") into a bipartite graph test <-> line. That graph answers questions a
percentage cannot, and two of them have bitten this repo:

- **"Is this property covered on BOTH engines?"** A line can be at 100% and still be reached only
  by the SQLite half of a parametrized suite. Run this over the seeding path and the answer is a
  list of test ids you can read the `[postgres]` off.
- **"Did this test go vacuous?"** A test whose assertions stopped running still passes. Its
  context disappearing from the region it was written to cover is the tell.

Usage::

    just cov-contexts                       # writes .coverage-contexts
    uv run python scripts/cov_contexts.py handlers/absurd.py 341 400
    uv run python scripts/cov_contexts.py effective/keys.py            # whole file

`--db` points at a different coverage file. Exits non-zero if the region has no contexts at all,
which is itself the interesting answer.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

from coverage.numbits import numbits_to_nums

DEFAULT_DB = ".coverage-contexts"


def contexts_for(db_path: Path, path_suffix: str, lo: int, hi: int) -> dict[str, list[int]]:
    """`{test id: sorted lines it executed in the region}` for files matching `path_suffix`.

    Matched by suffix rather than exact path so a caller can say `handlers/absurd.py` without
    knowing the checkout root. A suffix matching several files merges them, which is wanted: the
    question is "which tests reach this code", not "which file object".
    """
    db = sqlite3.connect(db_path)
    file_ids = [
        row[0] for row in db.execute("SELECT id FROM file WHERE path LIKE ?", (f"%{path_suffix}",))
    ]
    if not file_ids:
        raise SystemExit(f"no file matching {path_suffix!r} in {db_path}")
    hits: dict[str, list[int]] = {}
    for file_id in file_ids:
        for context_id, numbits in db.execute(
            "SELECT context_id, numbits FROM line_bits WHERE file_id = ?", (file_id,)
        ):
            lines = sorted(n for n in numbits_to_nums(numbits) if lo <= n <= hi)
            if not lines:
                continue
            (name,) = db.execute(
                "SELECT context FROM context WHERE id = ?", (context_id,)
            ).fetchone()
            if name:  # the empty context is "no test running" (import time, fixtures)
                hits.setdefault(name, []).extend(lines)
    return {name: sorted(set(lines)) for name, lines in hits.items()}


def overlap_for(db_path: Path, path_suffix: str, lo: int, hi: int, floor: int) -> list[tuple]:
    """`[(line, [test ids])]` for lines reached by at least `floor` tests, most-covered first.

    **Scope this by test ROLE before reading anything into it.** "Minimize overlap" is a
    UNIT-role rule; the conformance suites and the composition table overlap by design, so an
    unscoped query indicts the strongest suites in the repo. Run `just cov-contexts -m unit`
    first, then this over the resulting DB.

    **Overlap is a proxy for redundancy.** A context records that a line RAN under a test, never
    that the test's assertion was about that line: two unit tests on one line may cover different
    behaviors, and a test can execute a branch it does not exercise, which is how a mutation can
    survive a pin. The output is a list of review candidates.
    """
    by_line: dict[int, list[str]] = {}
    for name, lines in contexts_for(db_path, path_suffix, lo, hi).items():
        for line in lines:
            by_line.setdefault(line, []).append(name)
    crowded = [(line, sorted(names)) for line, names in by_line.items() if len(names) >= floor]
    return sorted(crowded, key=lambda row: (-len(row[1]), row[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="source path suffix, e.g. handlers/absurd.py")
    parser.add_argument("lo", nargs="?", type=int, default=1, help="first line (default: 1)")
    parser.add_argument("hi", nargs="?", type=int, default=10**9, help="last line")
    parser.add_argument("--db", default=DEFAULT_DB, type=Path)
    parser.add_argument(
        "--group",
        default="postgres",
        help="substring that marks a subgroup to count separately (default: postgres)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        metavar="N",
        help="instead of listing tests, report lines reached by >= N tests (review candidates, "
        "not defects — and scope the RUN by role first: `just cov-contexts -m unit`)",
    )
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"{args.db} not found — run `just cov-contexts` first")

    if args.overlap:
        rows = overlap_for(args.db, args.path, args.lo, args.hi, args.overlap)
        print(f"{len(rows)} line(s) in {args.path} reached by >= {args.overlap} tests")
        print("(review candidates: a shared line does not make a test redundant)\n")
        for line, names in rows:
            print(f"   {args.path}:{line}  ({len(names)} tests)")
            for name in names:
                print(f"        {name}")
        return 0 if rows else 1

    hits = contexts_for(args.db, args.path, args.lo, args.hi)
    span = f"{args.path}:{args.lo}-{args.hi}" if args.hi < 10**9 else args.path
    print(f"{len(hits)} tests reach {span}\n")
    for name in sorted(hits):
        print(f"   {name}")

    marked = sorted(name for name in hits if args.group in name)
    print(f"\n--> containing {args.group!r}: {len(marked)}")
    for name in marked:
        print(f"   {name}")
    return 0 if hits else 1


if __name__ == "__main__":
    sys.exit(main())
