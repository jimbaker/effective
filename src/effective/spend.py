"""A token budget kept on disk, so a cap binds across runs and is checked before each call.

The ledger is append-only, one line per call's spend, written under an exclusive lock and summed
under a shared one, so concurrent callers in threads or processes each add exactly what they
spent. The check reads the total before the call, so callers that check at once may each make one
call past the limit.
"""

import fcntl
from dataclasses import dataclass
from pathlib import Path


class TokenBudgetExhausted(RuntimeError):
    """A call was refused before it was made: the ledger has reached its limit."""


@dataclass(frozen=True)
class TokenBudget:
    """Tokens spent on one channel, against `limit`, recorded in `ledger`."""

    ledger: Path
    limit: int

    @property
    def spent(self) -> int:
        if not self.ledger.exists():
            return 0
        with self.ledger.open() as lines:
            fcntl.flock(lines, fcntl.LOCK_SH)
            return sum(int(line) for line in lines if line.strip())

    def check(self) -> None:
        if self.spent >= self.limit:
            raise TokenBudgetExhausted(
                f"{self.spent} of {self.limit} tokens spent ({self.ledger})"
            )

    def add(self, tokens: int) -> None:
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.open("a") as lines:
            fcntl.flock(lines, fcntl.LOCK_EX)
            print(tokens, file=lines)
