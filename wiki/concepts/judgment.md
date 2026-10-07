# Judgment

A judgment is its own domain op, `Judge` (`effective.domain`), beside `AskLLM` and `CallTool`: bounded,
typed questions about one state, asked in one call. The questions are one of a set (`Choice`), the
probability a condition holds (`Noul`), and a position on ordered levels (`Score`), and the answer
is no prose and no plan. Anything that answers the wire can interpret the op: TypeSafe's Jev
(`effective.interpreters.jev`, pinned as `typesafe-sdk==0.5.7` behind the `judge` extra), a scripted table in a
test, a recorded tape. It is metered like `AskLLM`, forwarded by a dry run, and traced as a model
call: the two form `ModelCall` and carry the marker `AsksModel`, so a layer that treats every
model call alike matches one arm, `case AsksModel():`.

## The template is the request

A judgment is a t-string whose holes are the state and the questions, so a reader of the code sees
exactly what the judge is asked. That is the reason to mix them: the mixing is for the reader, and
the processor keeps the two apart on the wire.

| part of the template | becomes |
|---|---|
| a hole holding `Choice`, `Noul` or `Score` | a question named by its expression |
| a question hole `{q:each path}` | `q` asked once per element of state list `path`, as `q[i]` |
| literal text before a question hole | that question's instructions |
| any other hole | a state field named by its dotted expression, nested |
| literal text before a state hole | context, each state hole written as its backticked path |

`effective.api.judge(name, template, output)` yields the battery as one `Judge` step keyed
`judge:{name}` and returns the typed model; `effective.judgment.battery` is the processor.
`select(name, template, candidates)` is `judge` with one `Choice` over code-found candidates and a
no-match option. **A judgment has no `Repair`**: the same state gets the same decision, so a low
answer is routed to escalation or a FLAG, and asking again buys only noise.

## Rules

| rule | what it guards against |
|---|---|
| the answer has to be in the state | a judge asked about facts it was not shown answers from priors, and fetched pages often lack the fact |
| the policy has to be in the question, or in code | a judge splits on a policy question it never saw; composing separate answers in code does not |
| read the decision, never a number near a threshold | identical requests return the same choice while the probability drifts |
| batch a few questions over one state | a small battery agrees with separate calls at a fraction of the cost |
| a large battery costs separation | many questions in one call blur the match scores |
| merge spellings against one anchor | an anchor and a few candidates per call separates matches far better than a list of pairs |
| merge before `select`, by anchor | merging first gathers the true spellings into one cluster, so `select` chooses among entities rather than spellings |
| `select`'s coverage rule does not suffice alone | a near-duplicate spelling placed among the candidates still slips through |
| more fields on a name add nothing to a merge | extra attributes beside a name did not move the merge |
| layout is legibility, not protection | an instruction injected into the state rarely reached the judge, but why is unmeasured |

## Retrieval and replay

A page is a recorded `call_tool("fetch", ...)`, cached by URL, so a replay re-serves it. A subject
with no readable page is flagged by code with no judgment: a finding needs a source. Expect a
large share of fetches to be unreadable: connection errors, pages with almost no text, refusals,
not-found and timeouts.

Replay matches a step by its key alone, so a changed question is served its old answer. Recording
the rendered request beside the answer, and comparing it on replay, is the open fix.

`effective.interpreters.jev.Jev` keeps answers under a hash of the request and checks a
`TokenBudget` before each call, so a consumer sets its own ceiling. `effective.interpreters.cli`
answers `AskLLM` through `claude -p` and `codex exec` for prototyping on a subscription. Upstream:
[concepts/flatten](flatten.md), [concepts/tapes](tapes.md).
