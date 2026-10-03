"""The reference corpus, rendered — `just elk-demo` writes `build/graphs/`.

The eyeball gate. Every property in the test suite is machine-checkable (no overlaps, finite
bounds, feedback edges routed), and none of them answers *does this look like something a person
would want to read*. So the corpus exists twice: as assertions, and as files you open.

**The corpus is derived from what the projector actually emits, not from a wish list.** The obvious
fixture names — "gather fan", "nested gather" — describe shapes `from_keys` cannot produce: it
builds edges with `pairwise`, so an unrolled run graph is always a *chain*, and a gather appears as
the tape hopping between branch coordinates rather than as a fan. Fan-in shows up only after
folding, where sibling branches collapse onto one node. Naming the fixtures after the real shapes
keeps the corpus honest about what this substrate records.
"""

from pathlib import Path

from tdom import Markup, html

from effective.graphlayout.elkjs import ElkJs
from effective.graphlayout.prepare import prepare
from effective.graphlayout.svg import to_svg
from effective.graphview import RunGraph, fold_cycles, from_keys
from effective.keys import Key

LOOP = [
    "plan",
    *(f"{step}#{n}" if n > 1 else step for n in range(1, 5) for step in ("ask", "act")),
]
# Key-shaped DATA, and it must stay grammar-correct or the demo stops demonstrating: a corpus
# whose keys the fold cannot group folds 8 -> 8 and shows no folding at all. This one folds
# 8 -> 6. `--key-literals` reads `src` and would refuse a malformed `gather:` here, since the
# registry owns that tag; `start`, `fetch` and `summarize` are tags production does not own, so
# they are unread by design and only `test_every_demo_key_is_in_the_language` sees them.
GATHER = [
    "start",
    "gather:0,0;fetch",
    "gather:0,1;fetch",
    "gather:0,2;fetch",
    "gather:0,1;ledger;b:r1",
    "gather:0,0;ledger;a:r1",
    "gather:0,2;ledger;c:r1",
    "summarize",
]
PARKED = ["extract", "classify", "ledger;submitted:r1"]
LONG = [
    "extract",
    "gather:0,1;ledger;ticket-processed:sha256-9f2c1a,epoch-1785067200.0",
    "artifact:application/json,sha256-9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
]


def corpus() -> dict[str, RunGraph]:
    """Name → graph. Every one is a projection of a key list; nothing is hand-built."""
    loop_run = from_keys("loop", LOOP)
    gather_run = from_keys("gather", GATHER)
    return {
        "01-linear": from_keys("linear", ["extract", "classify", "ledger;ticket-processed:r1"]),
        "02-telemetry": from_keys(
            "telemetry",
            ["extract", "ask", "ledger;done:r1"],
            telemetry={"ask": (0.0142, 2_300_000_000), "extract": (0.0031, 410_000_000)},
        ),
        "03-unrolled-loop": loop_run,
        "04-folded-loop": fold_cycles(loop_run),
        "05-gather-interleaved": gather_run,
        "06-folded-gather": fold_cycles(gather_run),
        # The pending node rides `pending=`, not the key list: neither engine checkpoints an
        # await while it is pending, so a key list containing one is synthetic input with no
        # producer. `event;{name}` is the spelling `op_key(AwaitEvent)` produces.
        "07-parked-ask": from_keys("parked", PARKED, pending=Key.parse("event;approve:r1")),
        "08-long-labels": from_keys("long", LONG),
    }


PAGE = """<!doctype html>
<meta charset="utf-8"><title>effective — run-graph layout corpus</title>
<style>body{{font-family:ui-sans-serif,system-ui,sans-serif;margin:2rem;max-width:1100px}}
figure{{margin:0 0 2.5rem}}figcaption{{font:600 13px ui-monospace,monospace;margin-bottom:.5rem}}
svg{{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px}}</style>
<h1>Run-graph layout corpus</h1>
<p>Rendered by <code>just elk-demo</code>:
<code>RunGraph → prepare → ELK JSON → elkjs → SVG</code>.
Dashed purple is a <em>feedback</em> edge (the loop a fold recovered); dotted grey is a
<em>commit-order</em> edge (cross-branch commit order, not causation); teal enters or leaves a
gather region.</p>
{figures}
"""


def main(out: Path | None = None) -> Path:
    out = out or Path("build/graphs")
    out.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []
    with ElkJs() as elk:
        for name, run in corpus().items():
            graph = prepare(run, direction="DOWN" if run.cyclic else "RIGHT")
            svg = to_svg(graph, elk.layout(graph), inline_css=True, title=name)
            (out / f"{name}.svg").write_text(svg, encoding="utf-8")
            figures.append(
                # `to_svg` returns rendered markup, so it is spliced as `Markup`; the name is
                # a corpus key and rides as text.
                str(
                    html(
                        t"""<figure><figcaption>{name} — {len(graph.nodes)} nodes, \
{len(graph.edges)} edges</figcaption>{Markup(svg)}</figure>"""
                    )
                )
            )
    index = out / "index.html"
    index.write_text(PAGE.format(figures="\n".join(figures)), encoding="utf-8")
    print(f"wrote {len(figures)} graphs to {out}/ — open {index}")
    return index


if __name__ == "__main__":
    main()
