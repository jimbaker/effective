# Rosetta Stone: DSPy, Pydantic AI and LangGraph

The same ideas appear under different names in each framework. This page maps each framework's
terms onto Effective's, so a reader who knows one can port a program and see what changes. The
rows were checked against each framework's documentation, and the behaviors under "Measured" were
run on the versions [`examples/rosetta/uv.lock`](../../examples/rosetta/uv.lock) pins. All four release often, so check a row
against a newer release before relying on it.

## What a run's record is

The frameworks differ most in what they keep of a run and how a resumed run finds its place.

| framework | the record | a resume | a fork |
|---|---|---|---|
| Effective | a tape of op results, each under a key the author composes ([concepts/tapes](tapes.md)) | re-runs the workflow from the top and serves every op the tape holds, by key | serves the recorded prefix, substitutes one result, and runs the tail live (`fork_at`, `run_fork`) |
| DSPy | none for a run; a disk cache memoizes LM requests by a hash of their arguments, and `save` keeps an optimized program's instructions and demos | none | none |
| Pydantic AI | in a durable capability, each model request and tool call is a durable unit: Temporal and Absurd replay from the top, DBOS checkpoints step outputs, Prefect loads cached task results; Harness `StepPersistence` keeps an event log and message-history snapshots | by the engine; on Absurd the step names compose the agent's name, the toolset id and the tool, with repeats numbered `#2`, `#3` | `fork_run` branches from a message-history snapshot under a new run id |
| LangGraph | a checkpoint of the graph state per super-step; the Functional API restores completed `@task` results | a graph resumes at a node and re-executes that node from its start; an `@entrypoint` replays from its start, restoring tasks from the checkpointer | `update_state` branches a new checkpoint; nodes after it run again, model calls included |

Effective and Pydantic AI on Absurd share the model: replay from the top, results served by a
composed name. LangGraph's Functional API is close too, and identifies a task by its name and its
call order. A graph node is the unit LangGraph re-executes on resume, so code in a node before an
`interrupt` runs again, model calls included. Effective re-runs the plain code before a wait too,
which is why it must be pure, and serves every op before the wait from the tape.

## Measured

[`examples/rosetta/`](../../examples/rosetta/) ports one retrieval program to each framework, a search and then a model
call whose citation must index a passage, and drives each port with offline fakes that count their
calls. It is a uv project of its own, so none of the three frameworks enters Effective's
environment. `uv run --project examples/rosetta python examples/rosetta/compare.py` prints:

| question | Effective | Pydantic AI | DSPy | LangGraph |
|---|---|---|---|---|
| a bad citation, then a good one: calls to a valid answer | the workflow receives `Repair`; with the author's re-prompt, 2 model, 1 search | `ModelRetry`: 2 answering requests, 3 in all, since choosing the search is a request; 1 search | `Refine`: 3 model, one of them feedback, and 2 searches, since a retry re-runs the whole module; `BestOfN`: 2 and 2 | a conditional edge: 2 model, 1 search |
| a crash after the model answered: what the resume asks again | nothing: the answer was recorded when its op returned | `StepPersistence` on SQLite: the model, once | the whole program, unless the LM cache holds the answer | the model, when the crash is inside the node or task that called it; nothing after it |
| a side effect before an approval: how often it runs | once, as an op; plain code there runs again | twice, as code before `ApprovalRequired` in the same tool | no wait primitive | twice, as code before `interrupt` in the same node |
| a fork with the search result changed: what the fork asks | the model, once | the model, once, from an edited snapshot | the whole program | the model, once, after `update_state` |

The unit each framework records explains most of the table: an op in Effective, a node or a task in
LangGraph, a model request or a tool call in Pydantic AI, and nothing in DSPy. A side effect in its
own recorded unit runs once in each of them, so the third row measures what an author writes inside
one unit. The crash is an exception raised in the process, and Pydantic AI's Temporal, DBOS and
Absurd engines need a service and were not run.

## DSPy

| DSPy | Effective | what changes |
|---|---|---|
| `Signature`, `InputField`, `OutputField` | a `t"..."` prompt whose interpolations are inputs and output channels (`Field`, `Gated`), with a Pydantic model as the output | the template is written, where DSPy's adapter generates it from the signature |
| `Module.forward` | a workflow: a generator that yields ops | the program performs no I/O, so a handler decides what each call means |
| `Predict` | `render(prompt, output=Model)`, then `ask_llm` | the call is an op with a key, so it is recorded and replayed |
| `ChainOfThought` | a `reasoning` output channel before the answer | |
| `ReAct`, `ReActV2` | `run_agent`, the ReAct loop | |
| `RLM` | `run_code` on Monty, and the `recurse` combinator | |
| `Refine`, `BestOfN` | `Gated` returning `Repair`, which the workflow returns or re-prompts with | a typed check on one field, where `Refine` scores the whole module with a reward |
| `GEPA` | `improve`, over a Pareto frontier ([concepts/evals](evals.md)) | the same family: reflective prompt evolution |
| `MIPROv2`, `BootstrapFewShot`, `SIMBA` | none | |
| `dspy.configure(lm=...)`, adapters | an interpreter for `AskLLM`, chosen by the handler | |
| the LM disk cache | the content cache (`Cache` in [`src/effective/cache.py`](../../src/effective/cache.py)) and the tape | the tape is per run and keyed by name; the cache is across runs and keyed by content |
| MLflow tracing, callbacks | OTLP spans under the GenAI conventions, and layers | |
| `dspy.Tool.from_mcp_tool` | none | |

## Pydantic AI

| Pydantic AI | Effective | what changes |
|---|---|---|
| `Agent`, `run` | `run_agent`, or a workflow of your own | |
| `output_type` | the output model passed to `render` | |
| `@agent.output_validator` raising `ModelRetry` | `Gated` returning `Repair` | Pydantic AI validates the whole output against a retry budget; a channel checks one field inside the prompt |
| `@agent.tool`, toolsets | `call_tool`, answered by an interpreter | |
| `deps`, `RunContext` | a workflow's arguments, and the interpreter's own state | the workflow holds no handle to the world |
| `Hooks` (`before_model_request`, `wrap_tool_execute`) | layers: generators around each op ([`examples/hooks_as_layers.py`](../../examples/hooks_as_layers.py)) | |
| `requires_approval`, `DeferredToolRequests` | `await_event`, or a `govern` policy that parks | the run parks and resumes by replay, where Pydantic AI ends the run and starts a new one with the approvals |
| `UsageLimits`, Harness `SpendLimits` | budgets under `govern` | Effective, like `SpendLimits`, lets the call that crosses a spend ceiling complete, then refuses or parks the next call; the verdict is recorded, so a resumed run meets the same one |
| Harness guardrails (`allow`, `block`, `retry`, `approve`) | `govern`'s verdict: proceed, park or refuse | |
| `TemporalDurability`, `AbsurdDurability` | `DurableHandler` on Absurd or SQLite | |
| `pydantic_graph` | a generator; a branch is `match` | |
| `pydantic_evals` (`Dataset`, `Case`, `Evaluator`) | not built; see [concepts/evals](evals.md) | |
| Harness `Memory` | not built ([concepts/context](context.md)) | |
| `MCPToolset`, the `MCP` capability | not built | |
| `Instrumentation`, Logfire | OTLP spans under the same GenAI conventions | |

## LangGraph and LangChain

| LangGraph or LangChain | Effective | what changes |
|---|---|---|
| `StateGraph`, nodes, edges | a generator; its frame holds the state, and `match` is a conditional edge | |
| reducers | pure code between yields | |
| `Send` | `gather` | branch count is data, known before the branches run |
| `interrupt`, `Command(resume=...)` | `await_event` | a resume replays the workflow from the top: each op before the wait is served from the tape, and the plain code between ops runs again |
| a checkpointer | an engine: Absurd on Postgres, or SQLite | the engine records op results, where a checkpointer records graph state |
| `durability="exit" \| "async" \| "sync"` | each op's result is recorded before the workflow sees it | |
| `@entrypoint`, `@task` | a workflow, and an op | a task's identity is its name and call order; an op's is a key the author composes, and a repeated name is numbered by call order |
| subgraphs | `yield from` another workflow, or `spawn_child` | |
| `update_state`, time travel | `fork_at`, `run_fork` | both run the tail live; Effective substitutes one recorded result at the fork point |
| `Store` | not built ([concepts/context](context.md)) | |
| `create_agent` middleware (`before_model`, `wrap_tool_call`) | layers | |
| `HumanInTheLoopMiddleware` | a `govern` policy that parks | |
| `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` | budgets under `govern` | |
| `ToolStrategy(handle_errors=True)` | `Gated` returning `Repair` | |
| LangSmith | a reader of OTLP spans | an exporter to a collector is not built ([concepts/tapes](tapes.md)) |
| `langchain.mcp.MCPAdapter` | not built | |

## A DSPy module, ported

[`examples/port_dspy_module.py`](../../examples/port_dspy_module.py) ports a retrieval module whose search tool numbers its passages. In DSPy:

<!-- source: examples/port_dspy_module.py -->
```python
class GenerateAnswer(dspy.Signature):
    "Answer the question from the passages, citing one by its number."

    passages: list[str] = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    citation: int = dspy.OutputField()


class RAG(dspy.Module):
    def __init__(self, k: int = 3):
        self.k = k
        self.respond = dspy.ChainOfThought(GenerateAnswer)

    def forward(self, question: str):
        passages = search(question, k=self.k)
        return self.respond(passages=passages, question=question)
```

In Effective, the signature's output fields become a model and the prompt's channels, and
`forward` becomes a generator:

<!-- source: examples/port_dspy_module.py -->
```python
class GenerateAnswer(BaseModel):
    reasoning: str
    answer: str
    citation: int


def rag(question: str, k: int = 3) -> Effect[GenerateAnswer | Repair]:
    passages = yield from call_tool("search", {"query": question, "k": k}, list[str])
    context = "\n".join(passages)
    reasoning, answer = Field(str), Field(str)
    citation = Gated(int, lambda i: 0 <= i < len(passages), "cite a passage by its number")
    prompt = t"""Answer the question from the passages, citing one by its number.
{context}
Question: {question}
Reasoning: {reasoning}
Answer: {answer}
Citation: {citation}"""
    respond = render(prompt, output=GenerateAnswer)
    response = yield from ask_llm("respond", respond.messages, dict)
    return respond.resolve(response)
```

| DSPy | the port |
|---|---|
| the signature's docstring | the template's first line |
| `ChainOfThought` | the `reasoning` channel |
| `search(...)` inside `forward` | `call_tool("search", ...)`, an op the tape records |
| a citation that names no passage | `Gated` returns `Repair` before the answer is used |

`uv run python examples/port_dspy_module.py` drives it offline with a recording handler, replays the
recording with no search and no model, and shows the guardrail refusing a citation of passage 7.
`tests/test_readme.py` pins the output and both blocks above.
