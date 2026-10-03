# Sources

A fact a workflow decides on carries the source that asserted it, and a source is a tool: one
`call_tool` per kind of record, with a typed result, a cache keyed by the query, and a budget of
its own. What makes a fact authoritative is the source and what corroborates it. Web search is one
source among these, the one to use when the structured ones say nothing.

The case that made this a design is researching an organization with one opaque call: an LLM
with web search finds the organization, names its domain and quotes its evidence in a single
answer. Checked against the world, such answers name domains that do not resolve and quote text
the cited page does not contain, and each call reads tens of times more tokens than a bounded
judgment over gathered records ([[concepts/judgment]]). Splitting the call into sourced records
and bounded decisions is what this page describes.

## The tiers

| tier | examples | settles | form |
|---|---|---|---|
| registries | GLEIF, SEC EDGAR, Wikidata, company registries | legal name and former names, status, parent and ultimate parent, sometimes the official site | free structured APIs |
| regulators | a sector regulator's public database | licenses, approvals, enforcement actions, the responsible party as registered | free structured APIs |
| filings | 10-K and 8-K | business description, acquisitions, going-concern language | an index and documents |
| the web | a search API, then a fetch | a site when nothing above names one; what the registries miss | `effective.interpreters.web`: `fetch`, whose `Fetched` is a `Page`, an `Unreadable` answer or `Unreached`; `BraveSearch`; and `HeldSearch` to replay kept searches without a key. An op cache keeps no transient fetch (`web.lasting`) |

A tier says how a record is reached and what its publisher is accountable for. It does not rank
sources: a registry can be stale where the company's own page is current, which is why a fact is
corroborated rather than looked up.

## What a source is in Effective

| concern | how it is met |
|---|---|
| reaching it | an interpreter answering `call_tool("<source>", {...})`, with a typed result schema |
| replay | the answer is a recorded op, so a replay re-serves the record and never asks again |
| repeat queries | a cache keyed by the query, like the fetcher's cache keyed by URL |
| spend | a cap per source, checked before the call, counted in queries or in money as the source bills |
| terms | whether the provider lets us keep what it returns, since recording is keeping |
| what it asserts | the fact, the record it came from, and the record's own date |

The workflow keeps its shape: code gathers records, a judge decides the bounded questions over
them, and code composes the call. Whether a record is about this organization is the question an
anchored merge answers ([[concepts/judgment]]).

## Authority by corroboration

A domain is authoritative when independent sources name it and the site names the organization:
an official-website claim in a registry, a filing's cover page, a directory listing, and the
site's own text. That is a decision table in code over sourced facts, and so is every other fact
the policy reads. A fact one source asserts and another contradicts is a near tie, and goes where
near ties go.

## Which source next

An audit that records `would_change`, the observation that would flip its call, turns that
observation into a query once sources are named: a product launch is a press release, an acquisition
is a parent link or an 8-K. Choosing the next source by what it could change, at what it costs,
is the value-of-information step, and it gives web search its place: the source to ask when the
structured ones have nothing that could change the call.

## Reuse by content

Replay and a content cache answer different questions and share one piece.

| mechanism | keyed by | spans | answers |
|---|---|---|---|
| checkpoint replay | a step's position: its name and occurrence | one run | what this run saw; the durable truth a fork seeds from |
| content cache | the op's content: tool and arguments, a judgment's request, a prompt and schema | runs | whether anyone asked this before |

The piece they share is the op's digest: replay needs it to detect a changed op, which it cannot
today, since it compares keys alone; the cache needs it to find a reusable answer. Holding a
second round's searches, site picks and pages fixed while its registry queries change is a job
for content: a positional seed cannot do it, because the new queries change how many merge
batches and fetches each subject has, so the first round's answers would land on different ops,
which `SeedingCtx` refuses.

The substrate form is `effective.cache.Cache`, a wrapper over the metered base, where every op
answers `(result, usage)`. It is a wrapper rather than a layer because a hit must answer without
reaching the base at all:

| concern | how it is met |
|---|---|
| the key | the op's content, each value with its type, its schema, and the answerer's identity |
| spend | a hit reports zero usage, so the meter and the budget see nothing spent |
| telemetry | a hit is marked, so a trace tells a reused answer from a fresh one |
| scope | opt-in per op kind: a model call is sometimes meant to be sampled again |
| a fork by content | a run through the cache with its store seeded from a base run |

## Status

The content cache is `effective.cache`, and the web tier is `effective.interpreters.web`. The
structured tiers are reached by whatever interpreter a consumer writes for them; none ships here.
A filing that mentions a name is evidence only once a judgment names the party it concerns.
