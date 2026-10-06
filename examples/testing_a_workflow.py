"""Testing a workflow: canned answers, then replay as the judge of what it yields.

Run it with `uv run python examples/testing_a_workflow.py`. `wiki/concepts/testing.md` says what
each handler here is for: a recording handler answers each op from canned responses and keeps a
trace, and a replay handler re-runs the workflow against that trace and refuses any change in
what it yields.
"""

from first_workflow import set_temperature

from effective import RecordingHandler, ReplayHandler, ReplayMismatch

RESPONSES = {"setpoint": {"celsius": 22}, "tool:thermostat": "set to 22°C"}


def main() -> None:
    recorder = RecordingHandler(RESPONSES)
    target = recorder.run(lambda: set_temperature("r1", "make it warmer"))
    print("record:   ", target, "via", *(entry.key.stored() for entry in recorder.trace))

    replayed = ReplayHandler(recorder.trace).run(lambda: set_temperature("r1", "make it warmer"))
    print("replay:   ", replayed, "with no model and no thermostat")

    hot = RecordingHandler({"setpoint": {"celsius": 45}})
    refused = hot.run(lambda: set_temperature("r2", "make it much warmer"))
    print("guardrail:", refused, "via", *(entry.key.stored() for entry in hot.trace))

    try:
        ReplayHandler(recorder.trace).run(lambda: set_temperature("r1", "make it warmer", 20))
    except ReplayMismatch as drift:
        print("drift:     ReplayMismatch:", drift)


if __name__ == "__main__":
    main()
