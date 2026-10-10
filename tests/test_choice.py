"""`choice.decide`, the one batch rule every interpreter applies, over every small race.

The cases are generated: every race of up to four branches, every `k`, every way its branches
split into earlier successes, this batch's successes and failures, and branches still running,
and both answers to whether the deadline has arrived. Each property is one the race's rulings
state, and a pin collects every case that breaks it.

Eligibility is settled before `decide` sees a batch: a success that landed at or after the
deadline reaches it as a failure, so `expired` here reads the deadline's arrival and nothing else.
"""

from itertools import product

from effective.choice import Choice, decide


def _cases():
    for n in range(1, 5):
        for k in range(1, n + 1):
            for roles in product(("earlier", "won", "lost", "running", "done"), repeat=n):
                earlier = [i for i, r in enumerate(roles) if r == "earlier"]
                if len(earlier) >= k:
                    continue  # the race would have decided in an earlier batch
                won = [i for i, r in enumerate(roles) if r == "won"]
                lost = [i for i, r in enumerate(roles) if r == "lost"]
                running = sum(r == "running" for r in roles)
                for expired in (False, True):
                    yield (
                        n,
                        k,
                        earlier,
                        won,
                        lost,
                        running,
                        expired,
                        decide(k, earlier, won, lost, running, expired),
                    )


def _broken(choice: Choice | None, k: int, earlier, won, lost, running, expired) -> str | None:
    s = len(earlier) + len(won)
    match choice:
        case None if expired:
            return "undecided though the deadline had arrived"
        case None:
            return None if s < k and running else "undecided though the choice is determined"
        case Choice(batch=batch) if list(batch) != sorted(won + lost):
            return "the batch recorded is not the batch read"
        case Choice(kind="timeout", winners=winners) if winners:
            return "a timeout naming a winner"
        case Choice(kind="timeout"):
            return None if expired and s < k else "a timeout the deadline did not decide"
        case Choice(kind="impossible") if expired:
            return "impossible where the deadline had already decided it"
        case Choice(kind="impossible"):
            return None if s < k and not running else "impossible while a branch runs or s >= k"
        case Choice(kind="winners", winners=winners):
            return _misread(list(winners), k, earlier, won)
    return "an unknown choice"


def _misread(winners: list[int], k: int, earlier: list[int], won: list[int]) -> str | None:
    """The first way a winning choice departs from the batch rule, or `None`."""
    rules = [
        (len(winners) == k, "a winning choice without exactly k winners"),
        (winners == sorted(winners), "winners out of branch-index order"),
        (set(earlier) <= set(winners), "an earlier success was passed over"),
        (
            sorted(set(winners) - set(earlier)) == sorted(won)[: k - len(earlier)],
            "the batch's winners are not its lowest-index successes",
        ),
    ]
    return next((reason for holds, reason in rules if not holds), None)


def test_every_small_race_decides_as_adr_0025_rules():
    wrong = {
        (n, k, tuple(earlier), tuple(won), tuple(lost), running, expired): reason
        for n, k, earlier, won, lost, running, expired, choice in _cases()
        if (reason := _broken(choice, k, earlier, won, lost, running, expired)) is not None
    }
    assert wrong == {}
