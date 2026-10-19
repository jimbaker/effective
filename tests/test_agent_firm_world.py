"""The scripted world under gather's threads: a tool honors its key once, however calls overlap."""

import os
import threading
from collections import Counter
from collections.abc import Mapping
from typing import Any

from effective.domain import CallTool
from examples.agent_firm.world import World

THREADS, KEYS = 8, 2000


def test_overlapping_calls_under_one_key_reach_the_tool_once():
    delivered: Counter[str] = Counter()

    def notify(args: Mapping[str, Any]) -> dict[str, Any]:
        os.sched_yield()
        delivered[args["idempotency_key"]] += 1
        return {"ok": True}

    world = World(tools={"notify": notify})
    domain = world.domain()
    keys = [str(n) for n in range(KEYS)]
    start = threading.Barrier(THREADS)

    def calling() -> None:
        start.wait()
        for key in keys:
            domain.run(CallTool(name="notify", args={"idempotency_key": key}, result_schema=dict))

    threads = [threading.Thread(target=calling) for _ in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert delivered == Counter(keys)
    assert world.calls["notify"] == THREADS * KEYS
