"""An Absurd app whose worker fails a task the way both engines do.

Absurd retries any exception while attempts remain. A task that raises an `Unretryable` error, or
only refusals, would raise the same again on every retry, so the worker ends the task on the
attempt that raised it. A task
failed for good, of that or of a crash on its last attempt, is reported as its failing leaf and
answers the parent that spawned it, as the embedded engine's `work_batch` does."""

from collections.abc import Callable
from typing import Any

from absurd_sdk import Absurd, CancelledTask, FailedTask, SuspendTask

from effective.bridge_absurd import end_attempts_at_this_run
from effective.engines.absurd import sdk_claim
from effective.handlers.base import failing_leaf
from effective.ops import DONE_EVENT_PARAM, noted
from effective.spawning import failure_answer


def fail_terminally(ctx: Any, execute: Callable[[], Any]) -> Any:
    """The `wrap_task_execution` hook: run the claimed task, and when it fails for good, spend its
    remaining attempts and answer its parent. Only while this run is still the task's latest, so a
    run whose lease expired speaks for nobody. A lease that expires after that check lets the claim
    sweep fail the run first: the task row then reads `$ClaimTimeout` while the parent heard this
    error."""
    try:
        return execute()
    except SuspendTask, CancelledTask, FailedTask:
        raise
    except Exception as raised:
        claim = sdk_claim(ctx)
        if (leaf := failing_leaf(raised, claim.attempt)) is None:
            raise
        latest = end_attempts_at_this_run(
            claim.conn, claim.task_id, claim.run_id, queue=claim.queue
        )
        if latest and DONE_EVENT_PARAM in claim.params:
            ctx.emit_event(claim.params[DONE_EVENT_PARAM], failure_answer(leaf, raised))
        raise noted(leaf, raised) from None


def absurd_worker(
    dsn: str, *, queue_name: str = "default", default_max_attempts: int = 5
) -> Absurd:
    """An Absurd app whose task runs fail for good the way the embedded engine's do. Every process
    that works a queue builds its app here."""
    return Absurd(
        dsn,
        queue_name=queue_name,
        default_max_attempts=default_max_attempts,
        hooks={"wrap_task_execution": fail_terminally},
    )
