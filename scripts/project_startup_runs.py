"""Draw the startup workflows' runs into `wiki/concepts/building-with-effective.md`.

Each block between `<!-- projected:{scenario}:{view} -->` and `<!-- /projected -->` holds one
projection of one run of that scenario's timeline:

| view       | projection                                           |
|------------|------------------------------------------------------|
| `sequence` | `to_sequence`                                        |
| `graph`    | `to_mermaid(fold_cycles(...))`, drawn left to right  |
| `tree`     | `to_text`                                            |

Every block is drawn from one run per scenario. A gather's branches commit in whatever order they
finish, so two runs can draw one program differently, and `--check` compares what the program
fixes:

| view       | compared                                                                      |
|------------|-------------------------------------------------------------------------------|
| `sequence` | the main line in order, and each branch's messages between two of its steps   |
| `graph`    | the folded nodes, and every edge that touches no node led by a branch frame   |
| `tree`     | the tree in order, except that sibling branches may come in any order         |

    uv run python scripts/project_startup_runs.py           # rewrite the blocks
    uv run python scripts/project_startup_runs.py --check   # exit 1 when a block disagrees
"""

import argparse
import re
import sys
from functools import cache
from pathlib import Path

from effective.graphview import (
    RunGraph,
    fold_cycles,
    from_keys,
    to_mermaid,
    to_sequence,
    to_text,
)
from effective.keys.frame import is_branch_frame, split_frames
from effective.keys.grammar import ARITY_SEPARATOR
from examples.startup.engine import run
from examples.startup.scenarios import SCENARIOS

PAGE = Path(__file__).resolve().parent.parent / "wiki" / "concepts" / "building-with-effective.md"
BLOCK = re.compile(r"<!-- projected:(\w+):(\w+) -->\n(.*?)<!-- /projected -->\n", re.S)
FENCE = {"sequence": "mermaid", "graph": "mermaid", "tree": "text"}
NODE = re.compile(r"^(n\d+)[\[{(]+\"(.*)\"[\]})]+$")
EDGE = re.compile(r"^(n\d+) (-->(?:\|[^|]*\|)?) (n\d+)$")
MAIN = ("sequenceDiagram", "participant")
"""Statements that open a sequence diagram, fixed by the program like the main line's."""
UNBRANCHED = str.maketrans("├└│─", "    ")
"""A tree line with its branch glyphs blanked: which sibling is last depends on the order."""


@cache
def timeline(scenario: str) -> tuple[str, ...]:
    """One fresh run of `scenario`, shared by every view of it."""
    chosen = SCENARIOS[scenario]
    return run(chosen.program, chosen.world(), chosen.deliver).timeline


def drawn(scenario: str, view: str) -> str:
    """One view of `scenario`'s run, as its fenced block."""
    return render(view, from_keys(scenario, timeline(scenario)))


def render(view: str, graph: RunGraph) -> str:
    """One view of `graph`, as its fenced block."""
    match view:
        case "sequence":
            text = to_sequence(graph)
        case "graph":
            text = to_mermaid(fold_cycles(graph), direction="LR")
        case "tree":
            text = to_text(graph)
        case _:
            raise ValueError(f"no view named {view!r}: {', '.join(FENCE)}")
    return f"```{FENCE[view]}\n{text}\n```\n"


def agreed(view: str, block: str) -> object:
    """What every run of one program draws alike in `block`, whatever order its branches took."""
    body = block.strip().split("\n")[1:-1]
    statements = [line.strip() for line in body]
    match view:
        case "sequence":
            drawn: list[object] = []
            lanes: dict[str, list[str]] = {}
            for statement in statements:
                speaker = _speaker(statement)
                if speaker in ("run", "world") or statement.startswith(MAIN):
                    drawn += [sorted(lanes.items())] if lanes else []
                    drawn.append(statement)
                    lanes = {}
                else:
                    lanes.setdefault(speaker, []).append(statement)
            return drawn + ([sorted(lanes.items())] if lanes else [])
        case "graph":
            labels = {m[1]: m[2] for s in statements if (m := NODE.match(s))}
            fixed = sorted(
                (labels[a], arrow, labels[b])
                for s in statements
                if (edge := EDGE.match(s))
                for a, arrow, b in [edge.groups()]
                if not _gathered(labels[a]) and not _gathered(labels[b])
            )
            return sorted(labels.values()), fixed
        case "tree":
            return _ordered(_nested(body))
        case _:
            raise ValueError(f"no view named {view!r}: {', '.join(FENCE)}")


def _gathered(label: str) -> bool:
    """A folded node standing for concurrent branches, whose edges follow commit order: one whose
    key leads with a gather or race branch frame."""
    return any(is_branch_frame(frame) for frame in split_frames(label.split(" ")[0])[0])


type Tree = tuple[str, tuple["Tree", ...]]


def _nested(lines: list[str]) -> Tree:
    """The tree a `to_text` block draws, children in drawn order, under an unnamed top so a
    second root is part of what is compared."""
    root: list = ["", []]
    stack = [root]
    for line in lines:
        name = line.translate(UNBRANCHED).lstrip()
        depth = (len(line) - len(name)) // 3
        node: list = [name, []]
        del stack[depth + 1 :]
        stack[-1][1].append(node)
        stack.append(node)
    return _frozen(root)


def _frozen(node: list) -> Tree:
    return node[0], tuple(_frozen(child) for child in node[1])


def _ordered(tree: Tree) -> Tree:
    """`tree` with the branches of each gather or race sorted: they run concurrently, so their
    order is the run's, and every other child keeps the program's order."""
    name, children = tree
    kept: list[Tree] = []
    run: list[Tree] = []
    for child in map(_ordered, children):
        if run and _fanned(child[0]) != _fanned(run[0][0]):
            kept += sorted(run)
            run = []
        if is_branch_frame(child[0]):
            run.append(child)
        else:
            kept.append(child)
    return name, (*kept, *sorted(run))


def _fanned(frame: str) -> str:
    """The gather or race a branch frame belongs to: `gather:1,2` belongs to `gather:1`."""
    return frame.partition(ARITY_SEPARATOR)[0]


def _speaker(statement: str) -> str:
    """The participant a sequence statement belongs to: the receiver of the world's reply, the
    sender of a message, the subject of a note."""
    if reply := re.match(r"world-->>(\w+)", statement):
        return reply[1]
    return re.split(r"->>|:", statement.removeprefix("Note over "))[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="compare, and write nothing")
    args = parser.parse_args(argv)
    page = PAGE.read_text()
    found = BLOCK.findall(page)
    if not found:
        print(f"{PAGE}: no projected blocks")
        return 1
    if args.check:
        stale = [
            f"{scenario}:{view}"
            for scenario, view, block in found
            if agreed(view, block) != agreed(view, drawn(scenario, view))
        ]
        for name in stale:
            print(f"{PAGE.name}: {name} differs from a fresh run; run {sys.argv[0]} to redraw")
        return 1 if stale else 0
    redrawn = BLOCK.sub(
        lambda m: f"<!-- projected:{m[1]}:{m[2]} -->\n{drawn(m[1], m[2])}<!-- /projected -->\n",
        page,
    )
    PAGE.write_text(redrawn)
    print(f"{PAGE.name}: drew {len(found)} blocks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
