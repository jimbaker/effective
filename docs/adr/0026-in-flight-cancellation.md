# ADR-0026: In-flight cancellation

- **Date:** 2026-09-26
- **Status:** Accepted. Built on the recorder, replay and both engines (`src/effective/cancel.py`),
  with the shell interpreter and the Responses callers as the interpreters that stop.
- **Supersedes:** ADR-0004 D4's "no mid-op cancel". The rest of D4 stands: an interrupt poll is a
  recorded op, and its answer is a checkpoint a replay re-derives.
- **Relates to:** ADR-0004 (the loop's `interrupt` seam), ADR-0009 (the step contract a cancel
  completes under), ADR-0025 (a race loser's stop, which this mechanism does not reach).

## 1. The decision

An op already running can be stopped, and the stop is the op's **result**. The interpreter
answers `Cancelled(partial)` with whatever the op produced first, the op completes, and every
handler records that answer the way it records any other. A replay serves the cancel without
running the op again.

The op set does not grow. A cancel is a result arm, the way a refusal is: each op keeps its
wrapper and its key, and a handler that has never heard of cancellation records the value
unchanged.

**Falsified if:** a replay reruns an op whose recorded result is a cancel; or a cancel requires an
op kind of its own.
**Checked by** `test_a_cancelled_conversation_survives_a_crash_at_every_op` (`tests/test_smol.py`)
and `test_an_action_cancelled_while_it_ran_is_its_observation_and_replays_as_recorded`
(`tests/test_cancel.py`). Not modeled.

## 2. Who stops, and how

A face holds a `CancelToken` and calls `cancel()` from its own thread, on Esc. An interpreter doing
blocking work registers what stops it with `on_cancel` before the work starts and for its duration,
connecting included, so the blocked call returns in the thread running the op.

| interpreter | its stop | what the op answers |
|---|---|---|
| shell | SIGTERM to the command's process group, then SIGKILL to the group after a grace period | `Cancelled` with the output so far, only when the command died of the stop's signal |
| Responses | shut the stream's socket down, then close it | `Cancelled` with the text that arrived |
| Responses, asyncio | cancel the call's task, whose unwinding closes the connection | `Cancelled` with the text that arrived |
| any other | none | its own result, once it finishes |

The token is level-triggered: it stays set until the face calls `reset()` at the next user turn, so
every op and poll after the Esc reads the same answer. Each stop runs at most once and never after
its block has exited, and a stop that raises leaves the others to run.

A command that finishes on its own after the cancel reports its own exit, and one that dies of a
signal the stop did not send reports that signal. A reply whose `response.completed` arrived
before the stop is the answer, metered, and one that ended `incomplete` or `failed` raises as it
would have without the stop. A stop only claims the ops it ended.

**Falsified if:** an op that completed before its stop took effect answers `Cancelled`; a stop runs
after the block that registered it has exited; or a blocked call outlives its stop.
**Checked by** `tests/test_shell.py` (a command that exits, or dies of its own signal, after a
cancel), `tests/test_openai_cancel.py` (a reply completed, or ended `incomplete`, before the stop)
and `tests/test_cancel.py` (a stop whose block exited never runs; a reset never reruns a stop; a
block's exit waits for a stop already running). Not modeled.

## 3. What the workflow sees

The workflow never receives the `Cancelled` value. The step surface raises `OpCancelled` in its
place, whichever handler recorded it, so a workflow cannot mistake a cancel for a completion.

A scoped body delivers a refusal to its parent and ends the run on any other exception, so
`OpCancelled` is caught in the scope that yielded the step. The ReAct loop (`src/effective/react.py`)
catches it around the decide, the compaction and the act:

| cancelled while it ran | the turn |
|---|---|
| the decide or a compaction | ends escaped, with nothing chosen |
| the act | takes the cancel as the tool's observation, then ends escaped |

Either way the transcript keeps what already ran and drops what the model chose but the loop had
not run.

**Falsified if:** a workflow receives `Cancelled` as a step's value; or a cancel inside a scoped body
fails the task.
**Checked by** `test_a_cancel_raises_through_the_generator_surface` and
`test_a_cancel_raises_on_the_coroutine_surface_too` (`tests/test_cancel.py`), and
`test_a_final_decide_cancelled_while_it_ran_ends_the_turn_and_keeps_what_ran` and
`test_a_command_cancelled_while_it_ran_is_the_observation_and_ends_the_turn`
(`tests/test_smol.py`). Not modeled.

## 4. The record

A durable engine checkpoints a cancel as a one-key object under the reserved key `__cancelled__`.
The shape is reserved: a step whose result takes it, from a tool's JSON or a model's object,
is refused on every handler with `ReservedShape`, since read back it would be an Esc nobody
pressed. Only a step's stored result decodes the shape; an event payload, a grant or a spawn
result is validated by its own schema and never read as a cancel.

The Esc lives in the process. An op cancelled and then crashed before its checkpoint commits
re-runs on resume with a fresh token, as any uncommitted step does (ADR-0009).

**Falsified if:** anything other than a step's stored result decodes as a cancel; or a step's
result in the reserved shape is served to the workflow on any handler.
**Checked by** `test_an_event_payload_shaped_like_a_cancel_is_refused_by_its_schema`
(`tests/test_smol.py`), and `test_a_result_in_the_reserved_shape_is_refused_by_the_recorder` and
`test_an_engine_refuses_to_checkpoint_a_result_in_the_reserved_shape` (`tests/test_cancel.py`). A
grant and a spawn result have no pin. Not modeled.

## 5. What this does not rule

| open edge | what holds |
|---|---|
| the spend of a model call cancelled mid-stream | the meter does not see it: a cancelled Responses call reports latency and no tokens |
| whether a provider stops generating, and billing, when the stream is severed | unmeasured |
| stopping a race loser mid-op | cooperative (ADR-0025): a loser finishes the op it started and stops at its next admission |
