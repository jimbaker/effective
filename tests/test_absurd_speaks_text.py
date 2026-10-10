"""Effective hands the Absurd SDK text on every path, so a run binds no `Key` through psycopg.

The suite cannot see a `Key` reaching psycopg on its own: collecting any module that imports
`effective.ledger` registers the dumper for the session. Each case runs in a process that can
import neither the ledger nor the worker, on an SDK connection opened before Effective is touched.
"""

import subprocess
import sys
import textwrap

import pytest
from _durable import DSN, pg_ready

pytestmark = pytest.mark.skipif(not pg_ready(), reason="needs Postgres/Absurd (just pgt-up)")

PROBE = textwrap.dedent(
    """
    import sys
    from uuid import uuid4

    class Absent:
        def find_spec(self, name, path=None, target=None):
            if name in ("effective.ledger", "effective.absurd_worker", "effective.pgkeys"):
                raise ImportError(f"absent: {name}")

    sys.meta_path.insert(0, Absent())
    from absurd_sdk import Absurd

    queue = "speaks_text_" + uuid4().hex[:12]
    app = Absurd(sys.argv[2], queue_name=queue)
    app.create_queue()
    try:
        if sys.argv[1] == "child":
            from effective.cost import MeteredInterpreter, Usage
            from effective.interpreters.scripted import scripted_caller
            from effective.interpreters.tools import make_tool_runner, run_subagent_as_task
            from effective.ops import DONE_EVENT_PARAM
            from effective.react import AssistantTurn

            @app.register_task("child", default_max_attempts=1)
            def child(params, ctx):
                turn = (AssistantTurn(thought="done", answer="42"), Usage())
                tools = make_tool_runner({})
                domain = MeteredInterpreter(llm=scripted_caller([turn]), tools=tools)
                return run_subagent_as_task(params, ctx, domain=domain)

            spawned = app.spawn("child", {"task": "answer", DONE_EVENT_PARAM: "probe:parent"})
        else:
            from effective.api import await_event, call_tool
            from effective.checkpoints import Checkpoint
            from effective.fork import run_fork_as_task
            from effective.keys import Key

            class Ledger:
                hypothetical = True

                def append(self, row, *, writer=None):
                    pass

            def workflow(run_id):
                yield from call_tool("prefix", {}, int)
                return (yield from await_event("probe:base", dict))

            @app.register_task("fork", default_max_attempts=1)
            def fork(params, ctx):
                return run_fork_as_task(
                    params, ctx, workflow=workflow, domain=None,
                    hypothetical_ledger=lambda _: Ledger(),
                    read_base=lambda _: [Checkpoint(Key.parse("step;tool:prefix"), 1)],
                )

            spawned = app.spawn("fork", {
                "child_run_id": "probe-child", "forked_from": "probe-base",
                "fork_point": "probe:base", "base_task_id": str(uuid4()),
                "through": "step;tool:prefix", "forked_at_event": "probe:base",
                "delta": {"decision": "approve"},
            })
        app.work_batch()
        snapshot = app.fetch_task_result(spawned["task_id"])
        assert snapshot.state == "completed", snapshot.failure
    finally:
        app.drop_queue()
        app.close()
    """
)

CASES = {
    "a spawned child answers its parent": "child",
    "a fork delivers its delta": "fork",
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_a_run_completes_with_no_key_dumper_registered(case):
    ran = subprocess.run(
        [sys.executable, "-c", PROBE, case, DSN], capture_output=True, text=True, check=False
    )
    assert ran.returncode == 0, ran.stderr[-2000:]
