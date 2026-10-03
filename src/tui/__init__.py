"""The terminal surface over a durable run — `tape x ui`, at the terminal end.

**Thin on purpose, and the thinness is the design.** A tape is durable, so it supports many UIs,
and what each one adds should be presentation and nothing else. Everything with a decision in it
already lives in `effective`: the projections (`graphview.project` / `fold_cycles`), the
containment split (`grammar.split_frames`), the tree renderer (`graphview.to_text`), the joins
(`runview.run_view`), the markdown floor (`runview.to_markdown`) and the read guard
(`viewing.ViewingCtx`). This package holds widgets, key bindings and workers.

The test of whether that line is drawn in the right place is simple: nothing here should be worth
porting to a second UI. A Shiny board or an MCP resource wants `RunView` and `to_markdown`, not
this.

**It never claims a task.** A page host without a registry once emitted a park's answer and called
`work_batch()` per click; the human clicked approve and the run failed permanently while every
per-task check passed. `SqliteApp._claim` refuses a name it did not register, so a surface that
only reads tables and appends events cannot reach that failure — and this one registers nothing.
Answering a park marks it claimable; a drain still has to run.

**And it never holds a writable ctx.** The decisions pane replays a run to recover what the tape
cannot carry, which is a read only so long as the replay stays behind the tape's frontier.
`ViewingCtx` is what makes that structural rather than a rule someone remembers.
"""

from tui.app import RunViewApp, main

__all__ = ["RunViewApp", "main"]
