"""A deduction reviewed by a live model: does a real critic find how much of a customer's
deduction the contract supports?

    uv run python -m examples.agent_firm.deduction_review --model haiku --calls 3
    uv run python -m examples.agent_firm.deduction_review --verify

The customer paid $97,000 against a $100,000 invoice and deducted $3,000 as an SLA credit, and the
firm proposes accepting the deduction as claimed. The contract caps the credit at $1,000, so a
review counts (`finds_the_cap`) when it revises the credit to exactly that and cites the contract.

Each call runs `critic`'s program once, durably, with a real model as the critic, and banks the run
under `deduction_reviews/<model>/<n>/`: `task.sql`, a text dump of its SQLite store, beside
`result.json`. All calls together spend at most `CAP` tokens, recorded in
`deduction_reviews/tokens.ledger`. `--verify` loads each dump into a fresh database and replays it
with every interpreter raising if asked. A dump keeps the store's schema as banked, so its replay
also checks that the current engine opens a store an older one wrote.
"""

import argparse
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from effective.interpreters.cli import ClaudePrint
from effective.spend import TokenBudget
from examples.agent_firm.critic import Review, finds_the_cap, replay, run
from examples.agent_firm.engage import CritiqueWire
from examples.agent_firm.world import World

BANK = Path(__file__).resolve().parent / "deduction_reviews"
CAP = 150_000
"""Tokens across every call of every arm."""


class Diverged(Exception):
    """A banked run whose replay differs from the result banked beside it."""


def banked(world: World, case: Path) -> bool:
    """One critique, kept in `case` as its store's dump beside its result; whether it found the
    cap."""
    case.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as directory:
        db = Path(directory) / "task.db"
        review = run(world, db)
        with closing(sqlite3.connect(db)) as store:
            (case / "task.sql").write_text("\n".join(store.iterdump()) + "\n")
    found = finds_the_cap(review)
    (case / "result.json").write_text(
        json.dumps({"finds_the_cap": found, **review.model_dump(mode="json")}, indent=2)
    )
    return found


def bank(model: str, calls: int) -> None:
    budget = TokenBudget(BANK / "tokens.ledger", CAP)
    critic = ClaudePrint(model=model, budget=budget)
    for n in range(calls):
        if ((case := BANK / model / str(n)) / "result.json").exists():
            continue
        found = banked(World(critic=critic, critiques=frozenset({CritiqueWire})), case)
        print(case.relative_to(BANK), "finds the cap:", found, "tokens spent:", budget.spent)


def replayed(root: Path = BANK) -> dict[str, Review]:
    """Every run banked under `root`, replayed from its dump and checked against its result."""
    runs = {}
    for case in sorted(path.parent for path in root.glob("*/*/task.sql")):
        with TemporaryDirectory() as directory:
            db = Path(directory) / "task.db"
            with closing(sqlite3.connect(db)) as store:
                store.executescript((case / "task.sql").read_text())
            review = replay(db)
        result = json.loads((case / "result.json").read_text())
        name = case.relative_to(root).as_posix()
        if (
            review != Review.model_validate(result)
            or finds_the_cap(review) != result["finds_the_cap"]
        ):
            raise Diverged(name)
        runs[name] = review
    return runs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="haiku")
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        if not (runs := replayed()):
            print("no banked runs under", BANK, file=sys.stderr)
            sys.exit(1)
        for name, review in runs.items():
            print(name, "replayed, finds the cap:", finds_the_cap(review))
        print("banked runs replayed:", len(runs))
    else:
        bank(args.model, args.calls)


if __name__ == "__main__":
    main()
