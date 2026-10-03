"""The `improve` objective loop (design-space §3, GEPA-shaped).

Proves the four structural rules and the MOO unification with no live model:
- proposal + scoring are sealed ops (a recorded op stream); replay re-derives the frontier
  with zero calls;
- single-objective is the argmax singleton, multi-objective is the Pareto frontier — one loop;
- `propose` receives the frontier WITH its ASI (the reflective signal is threaded, not lost);
- `done` gates termination (the staged "get it working, then raise quality" shape).
"""

from collections.abc import Callable

from effective import Effect, RecordingHandler, ReplayHandler, Suspended, step
from effective.domain import CallTool
from effective.improve import Measurement, Reflection, improve
from effective.pareto import Objective

MAX_VAL = Objective(key="val", direction="max")


def _run[T](handler, wf: Callable[[], Effect[T]]) -> T:
    """Run and narrow away the never-taken Suspended branch (these loops don't await)."""
    out = handler.run(wf)
    assert not isinstance(out, Suspended)
    return out


def _score(c):
    """A sealed scoring op — canned by name in the recorder."""
    m = yield from step("score", CallTool(name="score", result_schema=Measurement, args={"c": c}))
    return m


def _climb(parents, reflection):
    """A sealed proposal op returning the next candidate(s)."""
    out = yield from step(
        "propose", CallTool(name="propose", result_schema=list, args={"n": len(parents)})
    )
    return out


# --- single objective: the frontier is the argmax singleton -----------------


def _single_obj_responses() -> dict:
    """seed=0 -> propose [1] -> propose [2]: climbing a scalar."""
    return {
        "seed;score": Measurement(measures={"val": 0.0}, asi="start"),
        "gen:0;propose": [1],
        "cand:0,0;score": Measurement(measures={"val": 1.0}, asi="better"),
        "gen:1;propose": [2],
        "cand:1,0;score": Measurement(measures={"val": 2.0}, asi="best"),
    }


def test_single_objective_is_the_argmax_singleton():
    h = RecordingHandler(_single_obj_responses())
    front = _run(h, lambda: improve(0, _climb, _score, objectives=[MAX_VAL], rounds=2))
    assert [s.candidate for s in front] == [2]  # only the top survives
    assert front[0].asi == "best"  # the ASI rides along
    # proposal and scoring are the recorded op stream, in order
    assert [e.key.stored() for e in h.trace] == [
        "seed;step:score",
        "gen:0;step:propose",
        "cand:0,0;step:score",
        "gen:1;step:propose",
        "cand:1,0;step:score",
    ]


def test_improve_replays_without_the_model():
    rec = RecordingHandler(_single_obj_responses())
    live = _run(rec, lambda: improve(0, _climb, _score, objectives=[MAX_VAL], rounds=2))
    replayed = _run(
        ReplayHandler(rec.trace),
        lambda: improve(0, _climb, _score, objectives=[MAX_VAL], rounds=2),
    )
    assert [s.candidate for s in replayed] == [s.candidate for s in live]


# --- multi objective: the same loop keeps the Pareto frontier ---------------


def test_multi_objective_keeps_the_pareto_frontier():
    # two candidates, each best on a different axis -> both non-dominated
    # two children -> parallel scoring via gather (responses keyed by the QUALIFIED name)
    responses = {
        "seed;score": Measurement(measures={"quality": 0.1, "speed": 0.1}),  # dominated by both
        "gen:0;propose": ["fast", "good"],
        # >1 candidate => a real fan-out, so each score carries its gather branch coordinate
        # ahead of the candidate's own frame: two frames, both handler-applied.
        "gather:0,0;cand:0,0;score": Measurement(
            measures={"quality": 0.2, "speed": 0.9}, asi="fast one"
        ),
        "gather:0,1;cand:0,1;score": Measurement(
            measures={"quality": 0.9, "speed": 0.2}, asi="good one"
        ),
    }
    objs = [Objective("quality", "max"), Objective("speed", "max")]
    h = RecordingHandler(responses)
    front = _run(h, lambda: improve("seed", _climb, _score, objectives=objs, rounds=1))
    cands = {s.candidate for s in front}
    assert cands == {"fast", "good"}  # the seed is dominated by both; both children survive


# --- ASI threading + staged termination -------------------------------------


def test_propose_sees_parent_asi_and_done_gates_termination():
    seen_asi: list[str] = []

    def reflecting_propose(parents, reflection):
        seen_asi.extend(p.asi for p in parents)  # the reflection reads the ASI
        out = yield from step("propose", CallTool(name="propose", result_schema=list, args={}))
        return out

    responses = {
        "seed;score": Measurement(measures={"val": 1.0}, asi="tests fail: add(2,3)=-1"),
        "gen:0;propose": [9],
        "cand:0,0;score": Measurement(measures={"val": 9.0}, asi="tests pass"),
    }

    # stop as soon as a frontier point clears the bar (the staged "get it working" gate)
    def done(front):
        return any(s.measures["val"] >= 9 for s in front)

    h = RecordingHandler(responses)
    front = _run(
        h,
        lambda: improve(0, reflecting_propose, _score, objectives=[MAX_VAL], rounds=5, done=done),
    )
    assert seen_asi == ["tests fail: add(2,3)=-1"]  # propose reflected on the seed's ASI
    assert [s.candidate for s in front] == [9]
    # done fired after round 0, so round 1 never proposed (only one propose op)
    assert sum(1 for e in h.trace if e.key.stored().endswith("propose")) == 1


# --- compaction: fold the ASI history, PIN the rubric -----------------------


def test_compaction_folds_asi_history_but_pins_the_rubric():
    RUBRIC = "CONSTRAINT: stay pure"
    seen: list[Reflection] = []
    summarized: list[str] = []

    def rec_propose(parents, reflection):
        seen.append(reflection)
        out = yield from step("propose", CallTool(name="propose", result_schema=list, args={}))
        return out

    def summarize(history):
        summarized.append(history)  # exactly what the summarizer is shown
        digest = yield from step("digest", CallTool(name="digest", result_schema=str, args={}))
        return digest

    responses = {
        "seed;score": Measurement(measures={"val": 0.0}, asi="ASI-seed"),
        "gen:0;propose": [1],
        "cand:0,0;score": Measurement(measures={"val": 1.0}, asi="ASI-0"),
        "compact:1;digest": "DIGEST",  # the recorded summary op result
        "gen:1;propose": [2],
        "cand:1,0;score": Measurement(measures={"val": 2.0}, asi="ASI-1"),
    }

    def compact(history):  # fire once round-0's ASI has accumulated
        return "ASI-0" in history

    h = RecordingHandler(responses)
    front = _run(
        h,
        lambda: improve(
            0,
            rec_propose,
            _score,
            objectives=[MAX_VAL],
            rounds=2,
            rubric=RUBRIC,
            compact=compact,
            summarize=summarize,
        ),
    )
    assert [s.candidate for s in front] == [2]
    # the summarize op ran as a recorded step
    assert any(e.key.stored() == "compact:1;step:digest" for e in h.trace)
    # the summarizer saw the ASI history but NEVER the rubric
    assert summarized == ["ASI-seed\nASI-0"]
    assert all(RUBRIC not in shown for shown in summarized)
    # the rubric is pinned in EVERY reflection the proposer saw, before and after compaction
    assert [r.rubric for r in seen] == [RUBRIC, RUBRIC]
    assert seen[0].digest == "ASI-seed"  # round 0: raw ASI
    assert seen[1].digest == "DIGEST"  # round 1: folded — rubric still carried separately
