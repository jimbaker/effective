# coder

A small coding agent on Effective, in the spirit of [Pi](https://github.com/earendil-works/pi):
four tools, one working state, and a test suite that decides when the work is done.

```
export OPENAI_API_KEY=...            # or however you keep it
podman build -q -t effective-code-agent:pin -f infra/code-agent/Dockerfile infra/code-agent  # once per machine
uv run python -m examples.coder src/examples/coder/fixture "fix the bug in mod.py so the test passes"
```

It prints how the run stopped, what it cost, and a diff of what it committed. `--write` copies the
committed files back into the directory. The run lives in `<dir>/.coder/runs.db`, with a span
sidecar beside it, so `just tui <dir>/.coder/runs.db` draws it.

## What it does

The project is copied in as a `dict[str, str]` and never touched on disk. A visit is a ReAct loop
over four tools, one call per turn, and it ends when the model answers. The suite then runs in the
container over what the visit produced: green finishes the run, red sends the next visit back to
work, and a suite that cannot run parks. On every outcome the machine's postamble commits the tree
as one content-addressed artifact and records what the suite said.

| tool | what it takes |
|---|---|
| `read` | a path, and optional `offset` and `limit`; the file's text stops at 2000 lines or 50 KB, and a notice beside it says how to continue |
| `edit` | a path and disjoint exact replacements, each matched against the original file |
| `write` | a path and the file's whole content |
| `bash` | a command, run in a fresh container over a copy of the project with no network; it returns the exit code and the last 2000 lines or 50 KB, and changes it makes to files are discarded |

`edit` and `write` lint what they would produce, and a refusal comes back as the next observation
rather than as an error, so the model fixes what the refusal names. `bash` and the suite refuse
when the image is absent instead of running on the host.

## What the model reads

An observation is a `Template`, and every hole carrying content the tools READ declares `:data`: a
command's output, a file's text, and the path a model chose. The channel processor renders such a
hole between `[[ ## data <expression> ## ]]` and `[[ ## end ## ]]`, taking a different mark if the
content already uses that one. The delimiters share no metacharacter with JSON, shell or code, so
**nothing is escaped**: a file arrives byte for byte, and a file that quotes the fence is still one
block. `effective.envelope` writes the reply leg in the same alphabet.

The coder's own notices stay outside that block. A paging notice inside one is indistinguishable
from a line of the file imitating it, which is why `read` returns the text and the notice
separately rather than concatenated.

The framing is composed once, above the loop, so the prefix a provider sees is byte-identical
every turn. That is what makes caching possible at all, and it is why a loop prompt declaring a
cache boundary is refused rather than ignored.

## How it is put together

| file | holds |
|---|---|
| `tools.py` | the four argument models, their runners, and the tool server. A runner is a function of its arguments, and the project is one of them |
| `machine.py` | the WORK state, the worker, the judge and the transition |
| `prompt.py` | the framing as one `Template`, whose tool lines come from the argument models |
| `__main__.py` | the driver: copy in, run, print, optionally write back |

Two things follow from the tree travelling in each call's arguments. Nothing is kept between calls,
so a worker that picks the run up after a crash edits the files the crashed one left, wherever it
resumes; and a replay re-serves each recorded result without running a tool at all. The command
above always starts a new run, so resuming an interrupted one means driving the task yourself.

The same models give the prompt's tool lines and the strict function catalog the model is called
with, so what a tool is described as taking is what it validates. One call per turn, asked for on
the request and refused in the reply: the envelope holds one action.

## What it leaves out

Pi's parallel tool calls, its fuzzy match for a near-miss `oldText`, its host shell and its
steering queue; MCP, sub-agents and permission prompts. None of them is implemented.

What it keeps is checked against Pi's own tool tests, translated in `tests/conformance/pi/`. That
directory's `SOURCE.md` names the cases the coder answers differently and the cases that do not
apply.
