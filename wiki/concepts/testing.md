# Testing a workflow

A workflow performs no I/O, so testing it needs no mocks: a handler answers its ops from canned
responses or from a recording. Each question a test asks has its own instrument.

| question | instrument | where |
|---|---|---|
| does it do the right thing on known answers? | `RecordingHandler` with canned responses; assert the result and the trace of ops it yielded | `effective.handlers.recording` |
| is it deterministic, with stable keys? | `ReplayHandler` over that trace: the test oracle | `effective.handlers.replay` |
| does it survive a crash at every op? | the durable tests: `FaultCtx` crashes a run at its k-th op, before or after the op commits, and the retry resumes it | [`tests/_conformance.py`](../../tests/_conformance.py) (`at_every_op`), `just pgt-test` |
| do the two engines agree? | one conformance suite through `DurableHandler` on Absurd and on SQLite | [`tests/_conformance.py`](../../tests/_conformance.py) |

## `ReplayHandler` is the strict judge

`ReplayHandler` re-runs a workflow against a recorded trace with no model and no tools, and raises
`ReplayMismatch` when the workflow yields an op under a different key at a position, or a
different number of ops. It compares keys, never payloads: a changed prompt, tool argument or
result type replays clean. That catches what a passing run cannot show:

| a mismatch means | for example |
|---|---|
| the workflow read something outside an op, and it changed which ops it yields | a clock, randomness or shared state that picks a branch; one that only reaches a prompt or an argument replays clean |
| a key moved | an op inside a `gather` or `race` branch placed under a different frame |
| the control flow changed | a new check, a reordered pair of calls, a branch taken differently |

It serves each recorded value as it was recorded, without validating it against a changed result
type, and it runs no layers. [`examples/testing_a_workflow.py`](../../examples/testing_a_workflow.py) shows it pass and then refuse a
changed program.

## Production resume is lenient on purpose

`DurableHandler` resuming a run finds each recorded answer by the op's name, so a small deploy can
carry in-flight runs forward:

| the new code | on resume |
|---|---|
| changes a prompt or a tool argument | served the recorded result |
| swaps two ops | each is served by name |
| inserts an op | the new op runs live, in the middle of the replay |
| changes an op's result type | the recorded value is validated against the new type; the run fails if it does not fit |

So order drift is caught in a test, by `ReplayHandler`, and never at resume. Replaying recorded
production runs through `ReplayHandler` before a deploy would close that gap; it is not built.

## Test roles

A test's role says what a pass proves. The markers are registered in `pyproject.toml` under
`--strict-markers`, and a test that declares no role is a `unit` test: [`tests/conftest.py`](../../tests/conftest.py) marks
it, so a test declares a role only when its pass proves more than one seam.

| marker | a pass means |
|---|---|
| `unit` | one seam works in isolation; overlap with another unit test is waste |
| `spine` | the pieces compose; overlap with unit tests is the point |
| `journey` | a realistic path works end to end |
| `adversarial` | an attack failed; the test owes a mutation check |
| `conformance` | interpreters agree on one model; overlap across them is the design |

[concepts/tapes](tapes.md) says what a checkpoint holds and why replay can serve it;
[concepts/forced-schedules](forced-schedules.md) covers what a schedule instrument can decide;
[concepts/evidence](evidence.md) covers how a green test can be weaker than it looks.
