"""Absurd smoke test: spawn -> work_batch -> ctx.step -> result, against local PG.

Proves the durable loop end-to-end before wiring the DurableHandler. Uses
``work_batch()`` (synchronous, processes the queued task in-process) so no
background worker is needed.

    uv run python scripts/absurd_smoke.py
"""

import os

from effective.absurd_worker import absurd_worker

DSN = os.environ.get("DATABASE_URL", "postgresql://effective:effective@localhost:5432/effective")

app = absurd_worker(DSN)


@app.register_task("smoke-add", default_max_attempts=3)
def smoke_add(params, ctx):
    total = ctx.step("add", lambda: params["a"] + params["b"])
    return {"sum": total}


def main() -> None:
    spawned = app.spawn("smoke-add", {"a": 2, "b": 40})
    print("spawn result:", dict(spawned))
    task_id = spawned["task_id"]

    app.work_batch()  # claim + run the queued task synchronously

    snap = app.await_task_result(task_id, timeout=10)
    print("state:", snap.state, "| result:", snap.result, "| failure:", snap.failure)
    assert snap.result == {"sum": 42}, snap
    print("SMOKE OK")
    app.close()


if __name__ == "__main__":
    main()
