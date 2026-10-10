"""A second embodiment, from the repo's own tooling: the check that `effective.machine` is generic.

ROLE: conformance. `effective.coding` is the first embodiment. Alone, it makes "generic over the
states an embodiment declares" a claim with one witness, and a claim with one witness cannot tell
genericity from coincidence.

The machine here walks `just docs-check`: SCAN reads the link gate, REPAIR fixes what it named.
Nothing about a dangling documentation link is a test suite, and that is the point: every piece
of vocabulary below is this embodiment's own, so anything coding-shaped that the walk still
demands is a leak in the substrate rather than a choice made here.
"""

from enum import StrEnum
from typing import Any, assert_never

import pytest

from effective.api import call_tool
from effective.domain import CallTool, DomainOp
from effective.engines.sqlite import SqliteApp
from effective.handlers.durable import DurableHandler
from effective.keys import Run
from effective.machine.evidence import CommandRun, Predicate
from effective.machine.outcomes import Advance, Exhausted, Finish, Outcome, Park, ParkReason
from effective.machine.spec import Ctx, Evidence
from effective.machine.specs import build_specs
from effective.machine.trampoline import run_machine

pytestmark = pytest.mark.conformance

LINK_CHECK = "link_check"

DOCS_PREDICATE = Predicate(LINK_CHECK, CommandRun)
"""This embodiment's success predicate, declared once: the tool a docs deployment serves, and the
record it answers with.

`just docs-check` reports an exit code and the citations that dangled, which is exactly what
`CommandRun` is — "a record of what a command did, not a judgement about it". So a second
embodiment whose predicate is also a COMMAND reuses the shipped record and declares only the
name. That reuse is the evidence the record was never about test suites."""

WIKI = "docs/wiki.md"


class DocState(StrEnum):
    SCAN = "scan"
    REPAIR = "repair"


class ScanVerdict(StrEnum):
    CLEAN = "clean"
    DANGLING = "dangling"


def route(state: DocState, verdict: ScanVerdict | Exhausted[DocState]) -> Outcome[DocState]:
    """This embodiment's whole edge table. Two states, three edges, one route to `Finish`.

    Typed rather than `Any`, which is not tidiness: `assert_never` proves nothing against `Any`
    (ty says so — "Inferred type of argument is `Any`"), so an untyped router would have looked
    exactly like a closed one while closing nothing. The dependent-sum discipline is the
    embodiment's to keep, and this file exists to show a second embodiment keeping it."""
    match state, verdict:
        case _, Exhausted():
            return Park(state, ParkReason.EXHAUSTED)
        case DocState.SCAN, ScanVerdict.CLEAN:
            return Finish()
        case DocState.SCAN, ScanVerdict.DANGLING:
            return Advance(DocState.REPAIR)
        case DocState.REPAIR, _:
            return Advance(DocState.SCAN)
        case _:
            assert_never(verdict)


class DocsDeployment:
    """What a docs-check deployment serves — and NOTHING else.

    It refuses an unknown tool rather than answering harmlessly, because a harmless answer is how
    a substrate leak stays invisible: the walk would complete and nobody would learn that the
    generic machine had demanded a name this domain has never heard of."""

    def __init__(self) -> None:
        self.tools: list[str] = []
        self.repaired = False

    def run(self, op: DomainOp) -> Any:
        assert isinstance(op, CallTool)
        self.tools.append(op.name)
        match op.name:
            case "link_check":
                return CommandRun(
                    exit_code=0 if self.repaired else 1,
                    failures=() if self.repaired else (f"{WIKI} -> docs/gone.md",),
                )
            case "repair_link":
                self.repaired = True
                return {WIKI: "every citation resolves"}
            case unserved:
                raise KeyError(f"a docs deployment does not serve {unserved!r}")


def scan(ctx: Ctx) -> Any:
    run: CommandRun = yield from call_tool(LINK_CHECK, {}, CommandRun)
    return Evidence(summary=f"{ctx.state.value}: exit {run.exit_code}", measured=run)


def repair(ctx: Ctx) -> Any:
    tree: dict[str, str] = yield from call_tool("repair_link", {}, dict[str, str])
    return Evidence(summary=f"{ctx.state.value}: rewrote {WIKI}", tree=tree)


def judge_scan(_ctx: Ctx, evidence: Evidence) -> Any:
    assert evidence.measured is not None
    return ScanVerdict.CLEAN if evidence.measured.green else ScanVerdict.DANGLING
    yield  # pragma: no cover  -- a `Judge` is a generator


def judge_repair(_ctx: Ctx, _evidence: Evidence) -> Any:
    return ScanVerdict.DANGLING
    yield  # pragma: no cover


def docs_specs():
    return build_specs(
        DocState,
        workers={DocState.SCAN: scan, DocState.REPAIR: repair},
        judges={DocState.SCAN: judge_scan, DocState.REPAIR: judge_repair},
        canonical=frozenset({DocState.REPAIR}),
    )


@pytest.fixture
def app():
    a = SqliteApp(":memory:")
    yield a
    a.close()


def drive(app, deployment: DocsDeployment):
    @app.register_task("docs")
    def task(params, ctx):
        return DurableHandler(ctx, deployment).run(
            lambda: run_machine(
                Run(params["run_id"]),
                "no dangling citations",
                docs_specs(),
                route,
                start=DocState.SCAN,
                budget=6,
                tree={WIKI: "cites docs/gone.md"},
                predicate=DOCS_PREDICATE,
            )
        )

    return app.run_until_result(app.spawn("docs", {"run_id": "docs-1"}))


def test_a_machine_that_is_not_about_code_walks_to_a_finish(app):
    """The walk, on this embodiment's own vocabulary: scan finds a dangling citation, repair fixes
    it, scan re-reads and finishes."""
    deployment = DocsDeployment()
    snap = drive(app, deployment)
    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert [turn["state"] for turn in snap.result["turns"]] == ["scan", "repair", "scan"]


def test_the_substrate_demands_NO_tool_this_embodiment_has_not_declared(app):
    """THE POINT OF THE FILE. Everything the walk asked for is a name this domain owns.

    Before `predicate` was a parameter, the generic postamble yielded a hard-coded `run_suite` —
    so a docs machine reached its own commitment point and was asked for a pytest runner. The
    deployment refuses an unserved name, which is what turns that from a silent oddity into a
    failing walk."""
    deployment = DocsDeployment()
    snap = drive(app, deployment)

    # The subset check below holds of an EMPTY set, and a run that dies before its first tool
    # satisfies it perfectly — which is how this test passed while the leak it names was live.
    assert snap is not None
    assert snap.state == "completed", snap.failure
    assert set(deployment.tools) == {LINK_CHECK, "repair_link"}, deployment.tools


def test_the_predicate_still_runs_on_every_exit_path(app):
    """Parameterizing the predicate must not make it skippable — the unconditional tail is the
    property the whole postamble exists for. The last tool call is the commitment point's own
    reading of the predicate, taken after the walk has stopped."""
    deployment = DocsDeployment()
    drive(app, deployment)
    assert deployment.tools[-1] == LINK_CHECK
    assert deployment.tools.count(LINK_CHECK) == 3, deployment.tools
