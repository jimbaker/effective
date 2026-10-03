"""Pinned regression tests: `canonical()` set-order non-determinism across processes.

Each test asserts an INVARIANT CLASS, not a single instance. `canonical()` sorts sets and is
unified with `handlers/base.canonical_form`, so its output does not depend on the hash seed.

Why subprocess + differing PYTHONHASHSEED: a `set`/`frozenset` is emitted by
`to_jsonable_python` in hash-iteration order, which is `PYTHONHASHSEED`-dependent.
An in-process test shares one seed and structurally CANNOT observe the divergence;
the record/resume legs must run in separate
processes with different seeds — the same instrument as
`test_run_code_replay_agrees_across_processes_and_hash_seeds`.
"""

import json
import subprocess
import sys

_ENV_SEED1 = {"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin", "PYTHONPATH": "src:tests"}
_ENV_SEED2 = {"PYTHONHASHSEED": "2", "PATH": "/usr/bin:/bin", "PYTHONPATH": "src:tests"}


def _run(snippet: str, env: dict[str, str]) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        capture_output=True,
        text=True,
        env=env,
        cwd=".",
        check=True,
    )
    return proc.stdout.strip().splitlines()[-1]


# --- root cause: the checkpoint form is not process-stable for a set ----------

_CANONICAL_SNIPPET = """
from effective.code import canonical
import json
# a set of strings (small ints hash to themselves and would mask the bug), plus a
# nested set-of-tuples so the pin covers the nested-unordered case too.
value = {"tags": {"zebra", "apple", "mango"}, "pairs": {("a", "z"), ("m", "q")}}
print(json.dumps(canonical(value)))
"""


def test_canonical_is_process_stable_for_any_set_bearing_value():
    """CLASS: `canonical(v)` for ANY value carrying a set/frozenset must be
    byte-identical across processes with different PYTHONHASHSEED — it is the
    substrate's declared checkpoint form (`code.py:78-83`), and a durable
    worker-death + fresh-worker resume runs under an arbitrary new seed. Today it
    emits the set as a hash-order list, so the two disagree."""
    a = _run(_CANONICAL_SNIPPET, _ENV_SEED1)
    b = _run(_CANONICAL_SNIPPET, _ENV_SEED2)
    assert a == b, f"canonical() diverged across hash seeds:\n  seed1={a}\n  seed2={b}"


# --- the fingerprint: _args_digest is not process-stable for a set arg --------

_DIGEST_SNIPPET = """
from effective.monty import _args_digest
class St:
    args = ({"zebra", "apple", "mango"},)   # a declared host-call with a set argument
    kwargs = {"labels": {"q", "b", "m"}}
print(_args_digest(St()))
"""


def test_args_digest_is_process_stable_for_a_set_bearing_call():
    """CLASS: the call-site fingerprint (`monty._args_digest`,
    monty.py:83-91) must be identical across processes for the same logical
    (args, kwargs). It is built on `canonical()`, so a set-bearing argument makes
    the recorded digest (seed A) differ from the recomputed one (seed B), which
    the re-bind check reads as a divergence (see the end-to-end test below)."""
    a = _run(_DIGEST_SNIPPET, _ENV_SEED1)
    b = _run(_DIGEST_SNIPPET, _ENV_SEED2)
    assert a == b, f"_args_digest diverged across hash seeds: seed1={a} seed2={b}"


# --- end-to-end durable bite: a legitimate cross-process resume must not raise -

# Two-phase driver over MontyEngine (the code engine behind `code-execute`). Phase
# `record` runs segment 0 under one seed: a declared function `tally` is called with
# a set the sandbox itself built, so its fingerprint is recorded in fn_log; the run
# then pauses at the `save` action (a segment boundary — a real crash window). Phase
# `resume` runs segment 1 in a FRESH process/seed with that fn_log: re-execution
# re-binds the `tally` call, recomputes the fingerprint, and compares. This is the
# `run_code` replay-by-re-execution contract with no Postgres needed.
_ENGINE_DRIVER = r"""
import json, sys
from effective.monty import MontyEngine, CodeEngineError
CODE = (
    "s = {'zebra', 'apple', 'mango'}\n"
    "n = tally(s)\n"
    "r = save(n)\n"
    "{'n': n, 'r': r}"
)
def tally(s):
    return len(s)
def seg(fn_log, action_results):
    return MontyEngine(functions={"tally": tally}).execute({
        "code": CODE, "inputs": {}, "functions": ["tally"], "actions": ["save"],
        "fn_log": fn_log, "action_results": action_results,
    })
phase = sys.argv[1]
if phase == "record":
    out = seg([], [])
    assert out.status == "action"
    print(json.dumps({"fn_log": out.fn_log}))
else:
    fn_log = json.loads(sys.argv[2])
    try:
        out = seg(fn_log, [{"return_value": 1}])
        print(json.dumps({"ok": True, "output": out.output}))
    except CodeEngineError as exc:
        print(json.dumps({"CodeEngineError": str(exc)}))
"""


def _drive(phase: str, env: dict[str, str], *extra: str) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _ENGINE_DRIVER, phase, *extra],
        capture_output=True,
        text=True,
        env=env,
        cwd=".",
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_run_code_rebind_survives_fresh_process_resume_with_a_set_arg():
    """CLASS: a `run_code` segment that called a declared host function with a
    set-bearing argument must RE-BIND cleanly on a fresh-worker resume under a
    different hash seed. Today the re-bind fingerprint check (monty.py:164-173)
    recomputes a seed-B digest, finds it != the recorded seed-A digest, and raises
    a false `CodeEngineError` — a legitimate resume that cannot resume. The digest
    backstop, meant to CATCH nondeterminism, is built on the nondeterminism it
    polices."""
    recorded = _drive("record", _ENV_SEED1)
    fn_log = json.dumps(recorded["fn_log"])
    # control: same seed re-binds fine (proves the harness, not the seed, is the cause)
    same = _drive("resume", _ENV_SEED1, fn_log)
    assert same.get("ok"), same
    # the bite: fresh process, different seed
    fresh = _drive("resume", _ENV_SEED2, fn_log)
    assert "CodeEngineError" not in fresh, (
        f"legitimate cross-process resume raised a false re-bind mismatch: {fresh}"
    )
    assert fresh.get("ok"), fresh
