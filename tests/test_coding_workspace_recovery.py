"""The coding table served by a domain that keeps nothing, resumed on a fresh worker.

A domain that held the workspace across calls would start again from the seed on a new process,
so an edit made before a crash would be missing from every tree the resumed run reads or writes.
Here each call carries its tree, bound by `coding_bind` from the record, and every attempt builds
its own domain, as a worker resumed in a new process does. A crash at every op, in both halves,
commits the tree a run without one commits.
"""

from collections.abc import Mapping
from functools import partial
from typing import Any
from uuid import uuid4

import pytest
from _conformance import Fault, FaultPosition
from test_coding_agent_worker_tools import BROKEN, FIXED, MODULE, specs

from effective.coding.runners import CODING_TOOLS, serve_tool
from effective.coding.specs import coding_bind, coding_read
from effective.coding.states import State
from effective.coding.transition import transition
from effective.domain import AskLLM, CallTool, DomainOp
from effective.handlers.base import artifact_id
from effective.keys import Run, compose_key
from effective.machine.evidence import CommandRun
from effective.machine.trampoline import run_machine
from effective.ops import StoreArtifact
from effective.react import AssistantTurn, ToolRequest

SEED = {MODULE: BROKEN}
HELPER = "helper.py"
EXTRA = "extra.py"
STALE = "stale.py"


def writing(path: str, content: str) -> AssistantTurn:
    return AssistantTurn(
        thought=f"write {path}",
        tool=ToolRequest(name="write_file", args={"path": path, "content": content}),
    )


TURNS = (
    writing(MODULE, FIXED),
    writing(HELPER, "TWO = 2\n"),
    AssistantTurn(
        thought="check the fix", tool=ToolRequest(name="read_file", args={"path": MODULE})
    ),
)
"""Two edits and a read, so a crash between any two leaves the rest to run on a worker that never
saw what came before."""

EXPECTED = {MODULE: FIXED, HELPER: "TWO = 2\n", EXTRA: "THREE = 3\n"}
"""The tree a run commits, stated rather than taken from a run: every edit, the third one written
only once the read saw the first."""


def after_the_read(read: str) -> AssistantTurn:
    """The model's fourth turn, which turns on what the read showed: a read of the seed writes a
    file no uncrashed run commits."""
    return writing(EXTRA, "THREE = 3\n") if read == FIXED else writing(STALE, "STALE = 1\n")


class Serving:
    """A deployment of the table: a model answering from the transcript it is shown, and tools
    answering from the tree each call carries."""

    def run(self, op: DomainOp[Any]) -> Any:
        match op:
            case AskLLM(messages=messages):
                seen = [message["content"] for message in messages if message["role"] == "tool"]
                if len(seen) < len(TURNS):
                    return TURNS[len(seen)]
                if len(seen) == len(TURNS):
                    return after_the_read(seen[-1])
                return AssistantTurn(thought="done", answer="fixed the sign")
            case CallTool(name="run_suite", args={"tree": tree}):
                return CommandRun(exit_code=0 if tree.get(MODULE) == FIXED else 1)
            case CallTool(name=name, args=args):
                return serve_tool(name, args)
        raise AssertionError(f"unexpected op: {op!r}")


def fixing(run_id: str, seed: Mapping[str, str] = SEED):
    session = yield from run_machine(
        Run(run_id),
        "fix the sign",
        specs(tools=CODING_TOOLS, read=coding_read, bind=coding_bind),
        transition,
        start=State.DRAFT,
        budget=6,
        tree=seed,
    )
    return session.commitment is not None


def committed(backend, fault: Fault) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run's commit row and outcome row, each attempt on a domain of its own."""
    run_id = f"w{uuid4().hex}"
    name = compose_key(t"coding-recovery:{Run(run_id)}").stored()
    backend.register(name, fixing, None, fault, [], fresh=Serving)
    snap = backend.run_until_result(backend.spawn(name, run_id))
    assert snap.state == "completed", snap
    rows = backend.ledger_payloads(run_id)
    (commit,) = [row for row in rows if row["kind"] == "machine-committed"]
    (outcome,) = [row for row in rows if row["kind"] != "machine-committed"]
    return commit, outcome


EXPECTED_ID = artifact_id(StoreArtifact(value=EXPECTED, content_type="application/json"))
"""The artifact `EXPECTED` is, content and all, computed without running anything."""


def test_a_run_commits_every_edit(backend):
    commit, outcome = committed(backend, Fault())
    assert commit["artifact_id"] == EXPECTED_ID
    assert commit["changed"] == sorted(EXPECTED)
    assert outcome["passed"] is True


@pytest.mark.parametrize("position", list(FaultPosition))
def test_a_crash_at_every_op_commits_what_a_run_without_one_commits(backend, position):
    unarmed = Fault(position=position)
    clean, _ = committed(backend, unarmed)
    assert unarmed.count > 0
    for k in range(1, unarmed.count + 1):
        fault = Fault(k, position=position)
        commit, outcome = committed(backend, fault)
        assert not fault.armed, f"the crash at op {k} never fired"
        assert commit["artifact_id"] == clean["artifact_id"] == EXPECTED_ID, (k, commit)
        assert outcome["passed"] is True, (k, outcome)


def test_an_unsafe_seed_fails_its_task_on_the_first_attempt(backend):
    """A seed key no tool can repair fails where the first call builds its workspace, and a retry
    would refuse it again."""
    run_id = f"u{uuid4().hex}"
    name = compose_key(t"coding-unsafe:{Run(run_id)}").stored()
    unsafe = partial(fixing, seed={**SEED, "../outside.py": "X = 1\n"})
    backend.register(name, unsafe, None, Fault(), [], fresh=Serving)
    task = backend.spawn(name, run_id)
    snap = backend.run_until_result(task)
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "UnsafeTreePath"
    assert backend.task_attempts(task) == 1


@pytest.mark.parametrize(
    "seed",
    [{**SEED, "\udc80.py": "X = 1\n"}, {**SEED, HELPER: "X = '\udc80'\n"}],
    ids=["key", "content"],
)
def test_a_seed_with_no_bytes_to_store_fails_alike_on_both_engines(backend, seed):
    """A lone surrogate has no UTF-8 bytes. Postgres refuses it where SQLite would store it
    escaped, so the run refuses it before any op, and neither engine retries."""
    run_id = f"u{uuid4().hex}"
    name = compose_key(t"coding-unstorable:{Run(run_id)}").stored()
    backend.register(name, partial(fixing, seed=seed), None, Fault(), [], fresh=Serving)
    task = backend.spawn(name, run_id)
    snap = backend.run_until_result(task)
    assert snap.state == "failed", snap
    assert backend.failure_kind(snap) == "CompositionRefused"
    assert backend.task_attempts(task) == 1
    assert backend.checkpoint_keys(task) == []
