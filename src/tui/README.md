# `tui` — the terminal run view

A Textual surface over a durable run: what it did, where it is now, and the one answer you can
give it. `tape × ui`, at the terminal end — the tape is durable, so it supports many views, and
this is the one that fits in a shell beside the agent that wrote it.

**Executable.** Every command below runs as written and the output blocks are real captures, not
sketches. Only the UUIDs differ between runs — the demo store is seeded from a table with no model
and no network, so the shapes are stable. If a block does not match, that is a defect in this
document or in the code, not in your terminal.

**Prerequisites:** none beyond the repo's dev environment. The `tui` extra (textual, rich) is in
the dev group, so `uv sync` already installed it. A terminal **100 columns or wider** — the tree
pane is 40% of the width and a run id is 36 characters, so it truncates below that.

---

## Quickstart

### 1. Seed a store

```
just tui-demo
```

```
uv run python examples/tui_demo.py "$@"
seeded build/tui-demo.db with 3 runs:
  machine  01a046c7-2f17-7340-a081-b152f600e413
  react    01a046c7-2f4a-7a5a-9e46-24d9f6c6a2ba
  fanout   01a046c7-2f88-7d1a-9c3f-1f8c9f4a1e07

open it:  just tui build/tui-demo.db
          (0 = fanout, 1 = react, 2 = machine — newest first)
```

Three runs, chosen so the view tells the truth rather than only the happy path. What each one is
for is in `examples/tui_demo.py`, which is also where the workflows live — `src/tui` ships none,
because a run view is a reader and something else has to have written a run.

### 2. Open it

```
just tui build/tui-demo.db
```

It opens the newest run, `fanout`:

```
▼ 01a046c6-3df2-72dd-b021-2b29b7d501a9
├── ▼ gather:0,0
│   ├── step;tool:fetch
│   └── ledger;fetched:r-fanout,0
└── ▼ gather:0,1
    ├── step;tool:fetch
    └── ledger;fetched:r-fanout,1
```

A `gather:{g},{i};` frame is a frame like any other, so the tree nests it. **This is also the
picture's one honest limit:** the two branches interleaved on the tape and the tree draws them as
two contiguous blocks, because grouping by containment is what a tree *is*. The status line's op
count and the order of `RunView.nodes` are where a reader recovers sequence.

### 3. Switch runs — `:` opens the command line

Press `:`, type `open 2`, press Enter.

```
▼ 01a046c6-3d91-74ba-b534-55236fff98eb
├── ▼ d:0
│   └── ▼ state:test
│       ├── step;tool:read_file
│       └── step;tool:run_suite
├── ▼ d:1
│   └── ▼ state:draft
│       ├── step;tool:read_file
│       └── step;tool:run_suite
├── ▼ d:2
│   └── ▼ state:review
│       ├── step;tool:read_file
│       └── step;tool:run_suite
├── ledger;committed:r-machine
└── event;review:r-machine  (parked)
```

This is the shape `effective.coding` mints and the one route T exists for: **the tree is the
trajectory**. `d:{n}` counts the trampoline's visits, `state:{name}` says which code ran, and
dropping either loses a distinction.

Two things worth noticing in that block. The **postamble draws last** — `ledger;` sits below the
visits because each level sorts by `Node.order`, which is the one field a projection cannot
recompute from a key. And the parked node is there at all: no engine checkpoints an await while it
is pending, so it arrives through `from_keys(pending=…)` and is the whole difference between a
finished run's graph and a parked one's.

### 4. `p` cycles the projection — and the two are not the same fold

Press `p` once (`project` — axes **discovered** from the tape):

```
├── ▼ d
│   └── ▼ state
│       ├── step;tool:read_file  (x3)
│       └── step;tool:run_suite  (x3)
├── ledger;committed:r-machine
└── event;review:r-machine  (parked)
```
```
parked  ·  …  ·  4 ops  ·  cost unmeasured  ·  project  ·  1 parked
```

Press `p` again (`fold`: axes **declared**, the role each coordinate registers at its mint):

```
├── ▼ state:test
│   ├── step;tool:read_file
│   └── step;tool:run_suite
├── ▼ state:draft
│   ├── step;tool:read_file
│   └── step;tool:run_suite
├── ▼ state:review
│   ├── step;tool:read_file
│   └── step;tool:run_suite
├── ledger;committed:r-machine
└── event;review:r-machine  (parked)
```
```
parked  ·  …  ·  8 ops  ·  cost unmeasured  ·  fold  ·  1 parked
```

**8 → 4 under one and 8 → 8 under the other, on the same tape.** `project` asks the tape which
coordinates vary and strips them, so three visits of two tools collapse to one position with
`x3`. `fold_cycles` reads a declared list: `d:` is an unrolling coordinate and goes,
`state:` is a *naming* scope and stays — `state:test` and `state:draft` run different code, so
they are two program positions and there is nothing to collapse.

Both survive on purpose. The discovered one is sharper and moves with its data, which is what a
view wants; the declared one is stable across runs, which is what comparing two runs needs.

### 5. `m` toggles the markdown floor

The right pane switches to `runview.to_markdown` — the same value a Shiny board, a report export
or an MCP fallback takes unchanged. One renderer, four targets, and the typed write-back survives
the flatten (`answer` and its target stay legible rather than becoming prose).

### 6. Answer the park — and then run a drain

Press `:`, type `answer {"decision": "approve"}`, press Enter.

```
 ready  ·  …  ·  7 ops  ·  cost unmeasured  ·  unrolled  ·  0 parked
                            answered review:r-machine
```

The park is gone (8 ops → 7) — **and the run has not resumed.** Check:

```
sqlite3 build/tui-demo.db "SELECT name, state FROM tasks WHERE name='machine'"
```

```
machine|ready
```

`ready`, not `completed`. That is the contract, not a bug: `parked.answer` appends the event and
marks the task claimable, and *"a drain must still run — nothing polls."* **This surface never
claims.** It registers no task, and `SqliteApp._claim` refuses a name it did not register — which
is what makes the incident behind the split unreachable here: a page host that did hold the
registry once claimed an answered task and permanently failed the run, while every per-task check
passed.

So the drain is a different process, holding the workflows:

```
just tui-demo --drain
```

```
uv run python examples/tui_demo.py "$@"
drained build/tui-demo.db: 1 batch(es)
```

(`just` echoes the recipe line before running it — no recipe in this repo is `@`-prefixed, so
that first line is expected output rather than noise.)

Now press `r` in the view:

```
├── ▼ d:0
│   └── ▼ state:test
│       ├── step;tool:read_file
│       └── step;tool:run_suite
├── ▼ d:1
│   └── ▼ state:draft
│       ├── step;tool:read_file
│       └── step;tool:run_suite
├── ▼ d:2
│   └── ▼ state:review
│       ├── step;tool:read_file
│       └── step;tool:run_suite
└── ledger;committed:r-machine
```

The pending node is gone and the run is complete.

---

## Keys

| key | does |
|---|---|
| `enter` (on a leaf) | show what that op RECORDED — the checkpoint, verbatim, with multi-line fields laid out rather than escaped onto one line |
| `:` | open the command line; a command returns you here |
| `escape` | leave the command line without running anything |
| `r` | re-read and redraw |
| `p` | cycle `unrolled` → `project` → `fold` |
| `m` | toggle the markdown floor in the detail pane |
| `q` | quit |

Commands: `open <n>`, `answer <json>`, `help`.

**The status line leads with the run's STATE** — `ready`, `running`, `parked`, `completed`,
`failed` — because it decides whether the rest of the line is a snapshot or a final picture. Before
it was there, a run whose workflow had raised drew a tree with no nodes and `0 ops · 0 parked`,
which is exactly what a run that did nothing looks like. Neither bookkeeper can separate those: a
run that died before its first checkpoint did nothing, truthfully. `effective.runs` reads the
engine's task row, which is where the difference lives.

**A refused command keeps what you typed.** `nothing is parked` usually means *not yet* — a drain
has not advanced the machine — so the payload is worth keeping rather than retyping. Only a command
that acted clears the input.

**Why a mode at all**, since it is the first thing that looks like over-design: both arrangements
without one are broken in opposite directions. Leave the input focused and every single-key
binding is swallowed as text. Focus the tree instead and the command line is unreachable — `Tab`
lands on the parks table, so typing `open 2` fires `p` and cycles the projection. Both were found
by driving the app under tmux, and neither is visible from reading the code.

## What it will not do

- **Claim or drain.** Above. Answering marks a task claimable; something holding the registry runs.
- **Write through a ctx.** The decisions pane replays a run to recover what the tape cannot carry,
  and a replay driven past the last recorded op *commits* the ops after it into a run this process
  does not own. `effective.viewing.ViewingCtx` makes that structural rather than a rule someone
  remembers: a step with no record raises `OutranTheTape` instead of running.
- **Shape anything.** The tree text is `graphview.to_text`, the joins are `runview.run_view`, the
  containment is `grammar.split_frames`. This package is widgets. If a pane ever needs a shape it
  has to compute itself, that computation belongs in `effective`.

  **That rule has been exercised, twice, and it held.** The first real drive against a machine this
  package did not ship wanted two things it could not draw — is this run dead, and what did that op
  record — and both became `effective.runs` and `RunView.results` rather than SQL in a pane. The
  older `read.runs` had predicted the first in as many words while working around it.

- **Show a park's QUESTION.** It draws the wake event's KEY, which says which state is asking and
  about what, and not which answers are legal — that is domain vocabulary the substrate does not
  record, since `await_event` takes a `schema` and nothing durable carries it.
  (Open work.)

- **Show a resolved park at all.** `await_event` leaves no checkpoint, so a park-driven machine's
  answered visits live in the engine's `events` table and are invisible to `graphview.from_keys`.
  Measured: a nine-visit run drew two. Durable, not projected. (Open work: record event consumption.)

## Driving a real machine, not the demo

The demo store is three seeded runs. The surface's actual subject is a machine that parks for a
human, and `effective.prose` is the one it was shaken out against — a de-essaying pass as eight
states, seven of which are a park::

```
uv run python scripts/prose_run.py start src/effective/keys/processor.py --subject key-scope --line 232
uv run python scripts/prose_run.py drain --watch      # a SEPARATE process; see below
just tui build/prose.db                                # answer the parks here
```

Nine regions of the key module went through it this way. What the loop looks like from the view:

- the tree grows a `d:{n};state:{name}` frame per state, so **the tree is the trajectory**;
- `READ` runs a caller query first, so selecting `step;tool:find_callers` shows who uses the thing
  under the docstring — the evidence the next answer is made against;
- `VERIFY` derives its verdict from the gates rather than asking, so a failure bounces to `DRAFT`
  and selecting its node shows *which* check failed;
- a refused answer re-parks under `#2` rather than killing the run, and the refusal is recorded as
  an artifact you can select and read.

**The drain is a separate process on purpose.** This surface registers no task, so
`SqliteApp._claim` refuses a name it did not register — which is what makes the incident behind
the drain/client split structurally unreachable here. Answering marks a task claimable; something
holding the registry runs it.

## Next: a model drives the transitions, a judge still rules

Everything above was driven by an agent typing into a terminal. The next step is to keep the
judgment and hand over the *stepping*: a model (Luna over the API, or `claude -p` / codex in the
same configuration) performs the state transitions — read the park, do the mechanical work, submit
the answer — while a judge with more context rules the ones that need ruling.

The seam already exists and needs nothing new. `effective.machine`'s `StateSpec` has ONE slot, a
`Run`, and `specs.fuse` is the worker/judge pair as a construction an embodiment may decline —
which `effective.prose` currently declines, because every state is one park. Splitting it is
per-state: give `DRAFT` and `RELOCATE` a model worker and keep `CLASSIFY` and `REVIEW` parked, and
the tape records which was which. The machine core never learns about it.

Two things this surface would then owe, and neither is built:

- **the parks are answered by two different kinds of agent**, so a reader wants to know which. That
  is a per-park attribute the substrate does not carry.
- **`REVIEW` is the author** when the author also drafted, and an author cannot check a premise
  it wrote. A model-driven `DRAFT` makes the judge a genuinely different reader, which is the
  point; nothing enforces the split.

## Driving it without a human

The app is a real terminal program, so an agent, a CI job or an optimization loop drives it the
way a person does — through a terminal. This is the exact sequence the captures above came from:

```bash
tmux kill-session -t tuidrive 2>/dev/null
tmux new-session -d -s tuidrive -x 118 -y 30
tmux send-keys -t tuidrive "just tui build/tui-demo.db" Enter
sleep 7                                    # mount + the first read worker

tmux send-keys -t tuidrive ":"; sleep 1
tmux send-keys -t tuidrive "open 2" Enter; sleep 3
tmux capture-pane -t tuidrive -p           # assert on this
```

Two gotchas, both measured. Send `:` as its **own** `send-keys` before the command text, or tmux
delivers the colon and the text together and the input misses the mode change. And `sleep` after
each step: reads run on a thread worker, so a capture taken too early shows the previous frame
rather than an error.

For assertions that do not need a terminal at all, `App.run_test()` is faster and is what
`tests/test_tui.py` uses — it drives the same app with a pilot and no tmux. Use tmux when what you
are checking is the *rendering*; use the pilot when it is the behaviour.
