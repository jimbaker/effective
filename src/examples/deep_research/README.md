# deep_research

Deep research on Effective: a frontier search over leads, stopped when two hosts settle every cell
of the answer rather than when a model says it is done.

```
export BRAVE_SEARCH_API_KEY=... JEV_API_KEY=...   # or however you keep them
uv run python -m examples.deep_research "Kestrel probe" \
    launch="the year it launched" operator="the agency that operates it"
uv run python -m examples.deep_research ... --held   # the kept searches: no Brave key or spend
```

It prints, for each cell, the value two hosts agree on and the pages that state it, or `null` for a
cell the steps ran out on. The run lives in `.deep_research/runs.db`, beside the kept searches and
the three spend caps. Reads, judgments and pages are kept in `.deep_research/ops` by their content,
so a second run that asks nothing new spends nothing, and with `--held` needs only the Jev key.

## What it does

A question comes with cells, the facts a satisfactory answer states. A lead is a web search aimed
at one cell, and the search is `effective.search.frontier` over leads:

| step                | what happens                                                                 | answered by |
|---------------------|------------------------------------------------------------------------------|-------------|
| pick                | the two queued leads worth most                                              | code        |
| act on a lead       | search, fetch the top three pages, read each for claims and further leads    | Brave, the fetcher, `claude -p` |
| keep a claim        | only when its quote states the value and is on the page, elided parts in order | code      |
| score a new lead    | how likely its search is to state its cell's value                           | Jev         |
| merge               | fold the claims into the evidence; queue a refuting search for a contested cell | code     |
| stop                | every cell settled, or the steps run out                                     | code        |

A cell is settled when one value is quoted by at least two hosts and by more than any rival. The
report is a projection of the evidence, written by code.

Each lead's search, pages and reading are recorded under the lead's own key, so a run that crashes
and resumes under a different `pick` never hands one lead another's results.

## What the model sees

`brief` is a t-string, and it is all a reader sees: the question, the cells, the cell the lead is
for, and one page. The page's address and text arrive in `:data` holes, fenced as data from the
open web, so text on a page cannot pass for instructions. The evidence so far stays on the tape.

## Tests

`tests/test_deep_research.py` runs the program against a fictional world on `.example` hosts,
where the first step leaves the launch year contested and a refuting search settles it.
