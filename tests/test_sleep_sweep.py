"""The sleep gate: which waits name only a duration, and a baseline that can only shrink."""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import sleep_sweep  # noqa: E402

FLAGGED = {
    "a sleep called": "import time\ntime.sleep(0.1)",
    "an imported alias": "from time import sleep as nap\nnap(1)",
    "an asyncio sleep": "import asyncio\nasyncio.sleep(10)",
    "a sleep passed as a default": "import time\ndef f(sleep=time.sleep): ...",
    "a turn spun on in a while": "import asyncio\nwhile pending:\n    asyncio.sleep(0)",
    "a turn spun on in a for": "import asyncio\nfor _ in range(3):\n    asyncio.sleep(0)",
    "a select imported by name": "from select import select\nselect([], [], [], 1)",
    "an Event built to time out": "import threading\nthreading.Event().wait(0.1)",
    "a Queue built to time out": "import queue\nqueue.Queue().get(timeout=1)",
    "a select on nothing": "import select\nselect.select([], [], [], 1)",
}

PASSED = {
    "one turn of the loop": "import asyncio\nasyncio.sleep(0)",
    "one turn answering a value": "import asyncio\nasyncio.sleep(0, 7)",
    "an Event waited on without a timeout": "import threading\nthreading.Event().wait()",
    "a select on a descriptor": "import select\nselect.select([fd], [], [], 1)",
    "a durable sleep op": "from effective.api import sleep_until\nsleep_until(when)",
    # unscanned, as the sweep's docstring says: the Event is bound before it is waited on
    "a bound Event": "import threading\nevent = threading.Event()\nevent.wait(1)",
}


@pytest.mark.parametrize("source", FLAGGED.values(), ids=FLAGGED.keys())
def test_a_wait_that_names_only_a_duration_is_flagged(source: str):
    assert len(list(sleep_sweep.sleeps(source))) == 1


@pytest.mark.parametrize("source", PASSED.values(), ids=PASSED.keys())
def test_a_wait_for_something_passes(source: str):
    assert list(sleep_sweep.sleeps(source)) == []


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "waits.py").write_text("import time\ntime.sleep(1)\n")
    return tmp_path


@pytest.mark.parametrize(
    ("baselined", "verdict"),
    [(1, 0), (0, 1), (2, 1)],
    ids=["the count holds", "a sleep added", "a sleep removed and the count kept"],
)
def test_the_gate_fails_on_a_count_that_moves(
    tree: Path, baselined: int, verdict: int, monkeypatch
):
    if baselined:
        (tree / "scripts" / "sleep-baseline.txt").write_text(
            f"src/waits.py :: {baselined} :: a poll of a service that announces nothing\n"
        )
    monkeypatch.setattr(sys, "argv", ["sleep_sweep", "--gate", "src"])
    assert sleep_sweep.main() == verdict
