"""Spawners for tests: the `Spawner` a `spawn_tool` enqueues a child task through, per engine.

Not a test module. A spawner on SQLite shares the app's connection, which `SqliteApp.spawn` locks;
one on Absurd enqueues through its own app, so a spawn from inside a running task does not share
the worker's connection.
"""

from typing import Any

from effective.engines.absurd import AbsurdEngine
from effective.engines.sqlite import SqliteApp
from effective.interpreters.tools import Spawner


def sqlite_spawner(app: SqliteApp) -> Spawner:
    """Enqueue on `app` with the spawn's `idempotency_key`, so a crash between the enqueue and the
    spawn step's commit cannot enqueue a second child."""

    def spawn(
        task_name: str,
        params: dict[str, Any],
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        return str(
            app.spawn(
                task_name, params, idempotency_key=idempotency_key, max_attempts=max_attempts
            )
        )

    return spawn


def absurd_spawner(
    engine: AbsurdEngine, retry_strategy: Any, seen: list[str] | None = None
) -> Spawner:
    """Enqueue on `engine`'s SDK app, an engine of its own, routing to the spawn's queue; record
    each child's id in `seen` if given."""

    def spawn(
        task_name: str,
        params: dict[str, Any],
        idempotency_key: str,
        queue: str,
        *,
        max_attempts: int | None = None,
    ) -> str:
        result = engine.app.spawn(
            task_name,
            params,
            queue=queue,
            idempotency_key=idempotency_key,
            max_attempts=max_attempts,
            retry_strategy=retry_strategy,
        )
        task_id = str(result["task_id"])
        if seen is not None:
            seen.append(task_id)
        return task_id

    return spawn
