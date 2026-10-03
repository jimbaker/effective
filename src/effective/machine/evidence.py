"""What a run MEASURED — the mechanical facts a verdict can be computed from without a model.

`CommandRun` is the whole reason three of the coding machine's states need no LLM to reach a
verdict, and it is deliberately a **record of what a command did**, not a judgment about it: the
exit code, what it named as failing, and whether it managed to run at all. Turning those into a
verdict is the embodiment's job (`coding.verdicts`) and is a pure `match`.

**It says `Command`, not `Suite`, and the second embodiment is why.** A machine over `just
docs-check` reads exit codes and dangling citations, which is this record exactly — nothing about
it is a test suite, and the name said otherwise for as long as coding was the only embodiment.
The generic property is that a COMMAND ran; which command, and what its failures mean, is the
embodiment's.

**`collection_error` is separate from `failures` on purpose.** A command that could not run has
said nothing about the code — it is a fact about the environment, and reading it as "the checks
failed" is how a broken import gets mistaken for a red test and sends the machine to DRAFT to fix
code that is fine. That distinction is exactly why `TestVerdict.BROKEN_ENV` exists.

**`green` is a derived property, not a stored field.** A stored boolean could disagree with the
exit code it was derived from, and then two readers of the same record would answer differently.
"""

from dataclasses import dataclass, field
from typing import Protocol

from effective.domain import ToolRefused


class Measured(Protocol):
    """What the SUBSTRATE needs from any predicate record — one member, deliberately.

    The machine has to answer exactly one question about a measurement it did not design: did the
    run pass. Everything else a record carries is the embodiment's to read, and the bound is one
    member because every member added to it excludes a record shape.

    That is not hypothetical. An earlier sketch bound this to `green` AND `exit_code` AND
    `failures`, because the postamble wrote those two into the outcome row by name — and `ty`
    then refused a bench machine's `Pareto(score, cost_usd)`, which has no exit code and never ran
    a process. A bound wide enough to write the row is a bound that admits only command records,
    which is the genericity buying nothing. The row carries the record WHOLE instead
    (`commitment_postamble`), so this stays at one member and a Pareto point is a predicate
    record."""

    @property
    def green(self) -> bool:
        """Did the run pass? Derived, never stored — see `CommandRun.green` for why."""
        ...


@dataclass(frozen=True, slots=True)
class Predicate[R: Measured = CommandRun]:
    """An embodiment's success predicate: the tool a deployment serves, and what it answers with.

    ONE parameter rather than two, because the two facts must agree — a tool name and a record
    type that the tool does not return is a drift surface, and the postamble validates the result
    against `record` on the durable path. Bundled, they are declared once beside an embodiment's
    states and cannot come apart.

    The default (`run_suite`, `CommandRun`) is the coding machine's, kept as the default because
    it is what every existing walk names. It is a DEFAULT, not a law — see `SUITE_TOOL`."""

    tool: str
    record: type[R]


@dataclass(frozen=True, slots=True)
class CommandRun:
    """One execution of a success predicate that is a COMMAND.

    `exit_code` is the command's, verbatim, and it is kept raw rather than normalized so a reader
    can tell "nothing ran" from "it ran and passed", which a boolean cannot. pytest's vocabulary
    is the one to have in mind for the coding machine — 0 passed, 1 failed, 2/3/4
    interrupted-or-misused, 5 nothing-collected — but nothing here is pytest's."""

    exit_code: int
    failures: tuple[str, ...] = ()
    collection_error: str | None = None
    output: str = ""

    @property
    def green(self) -> bool:
        """Passed, and actually ran. Exit 0 with a collection error is not green — nor is exit 5,
        where the command succeeded at running nothing."""
        return self.exit_code == 0 and self.collection_error is None

    def failed(self, target: str) -> bool:
        """Did `target` fail? Substring rather than equality: a pytest node id carries a file,
        a class and parameters (`tests/t.py::test_x[case]`), and a caller naming the test knows
        the test, not the parametrization."""
        return any(target in failure for failure in self.failures)

    def failures_besides(self, target: str) -> tuple[str, ...]:
        """Everything red that is NOT the target — the difference between "still red" and
        "regressed", which is the only thing distinguishing those two DRAFT verdicts."""
        return tuple(f for f in self.failures if target not in f)


@dataclass(frozen=True, slots=True)
class Commitment[R: Measured = CommandRun]:
    """What the postamble recorded: the artifact it committed and what the predicate said.

    Both, always — including when the predicate failed. A recorded failure is what the incumbent's
    unconditional tail bought, and a `Commitment` that could represent "committed but not
    measured" would give it back. A predicate that refused the tree is measured as that refusal,
    which never passes. `files` is every path in the committed tree; `changed` is the paths whose
    content differs from the tree the run was seeded with, a removed path included."""

    artifact_id: str
    measured: R | ToolRefused
    files: tuple[str, ...] = field(default_factory=tuple)
    changed: tuple[str, ...] = field(default_factory=tuple)

    @property
    def passed(self) -> bool:
        # lint: totality(total): the capture arm is `R`, a type parameter no class pattern names.
        match self.measured:
            case ToolRefused():
                return False
            case measured:
                return measured.green
