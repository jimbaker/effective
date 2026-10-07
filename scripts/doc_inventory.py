"""Deterministic doc/code inventory: the ground truth a wiki pass reasons over.

No LLM, no source changes, stdlib only, so every number here is reproducible. The documents are
the maintained layer, `docs/` and `wiki/`; the code is every Python module in the tree.

Emits two JSON artifacts and a markdown summary:

| artifact       | holds                                                                     |
|----------------|---------------------------------------------------------------------------|
| refgraph.json  | per document: the wiki pages it links and its `path:line` code anchors;   |
|                | the links that resolve to no wiki page or `docs/` stem                    |
| codemap.json   | per module: subsystem, LOC, def/class counts, intra-repo import edges     |

    uv run python scripts/doc_inventory.py [--out build/wiki] [--quiet]

The markdown summary prints to stdout. The default `--out` is a derived build directory:
disposable, rebuilt on read, never committed.
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# The code-anchor grammar is SHARED with the link gate, which owns it: this script is a REPORTER
# (always exits 0) and `link_check.py` is the GATE, so the two agree on what a path reference is.
try:  # imported as a package member (tests, `from scripts.doc_inventory import …`)
    from scripts.link_check import PATHLINE_RE
    from scripts.wiki_lint import outbound
except ImportError:  # run as a script: scripts/ is on sys.path, the repo root is not
    from link_check import PATHLINE_RE
    from wiki_lint import outbound

REPO = Path(__file__).resolve().parent.parent
DOCS = REPO / "docs"
WIKI = REPO / "wiki"

# A page name: lowercase, digits, hyphens, and `/` for a wiki subdirectory. The shape keeps code
# that happens to sit in double brackets, such as [[DomainOp[Any]]], out of the graph.
WIKILINK_RE = re.compile(r"\[\[([a-z0-9][a-z0-9/-]+)\]\]")
# meta-examples that document the [[…]] syntax itself, not real links
WIKILINK_META = {"wikilink", "name", "double-bracket", "field", "links", "page"}


def read(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def documents() -> list[Path]:
    """The maintained layer's markdown, in a stable order."""
    return sorted(p for root in (DOCS, WIKI) if root.exists() for p in root.rglob("*.md"))


def inventory_docs() -> dict:
    wikilinks: dict[str, list[str]] = {}
    codeanchors: dict[str, list[str]] = {}
    for p in documents():
        rel = p.relative_to(REPO).as_posix()
        text = read(p)
        wikilinks[rel] = sorted(set(WIKILINK_RE.findall(text)) | outbound(p))
        codeanchors[rel] = sorted({f"{a}:{b}" for a, b in PATHLINE_RE.findall(text)})
    return {"wikilinks": wikilinks, "code_anchors": codeanchors}


def resolve_wikilinks(wikilinks: dict) -> dict:
    """Which [[names]] resolve to a wiki page or a `docs/` stem, and which dangle.

    A wiki page is named by its path under `wiki/` without the suffix (`concepts/flatten`).
    """
    known: set[str] = set()
    if WIKI.exists():
        known |= {p.relative_to(WIKI).with_suffix("").as_posix() for p in WIKI.rglob("*.md")}
    if DOCS.exists():
        known |= {p.stem for p in DOCS.rglob("*.md")}

    targets = sorted({t for tl in wikilinks.values() for t in tl})
    dangling = sorted(t for t in targets if t not in known and t not in WIKILINK_META)
    return {"known_count": len(known), "targets": targets, "dangling": dangling}


# --- code map ---------------------------------------------------------------
SUBSYSTEMS = [
    ("effective-core", "src/effective"),
    ("effective-handlers", "src/effective/handlers"),
    ("effective-cards", "src/effective/cards"),
    ("agent", "src/agent"),
    ("examples", "examples"),
    ("tests", "tests"),
    ("scripts", "scripts"),
    ("formal-lean", "formal/lean"),
    ("formal-quint", "formal/quint"),
    ("migrations", "migrations"),
]
IMPORT_RE = re.compile(r"^\s*(?:from\s+([\w.]+)\s+import|import\s+([\w.]+))", re.MULTILINE)
DEF_RE = re.compile(r"^\s*(?:async\s+)?def\s+\w+", re.MULTILINE)
CLASS_RE = re.compile(r"^\s*class\s+\w+", re.MULTILINE)
INTRA = ("effective", "agent")


def subsystem_of(rel: str) -> str:
    best = "other"
    best_len = -1
    for name, prefix in SUBSYSTEMS:
        if rel.startswith(prefix + "/") and len(prefix) > best_len:
            best, best_len = name, len(prefix)
    return best


def inventory_code() -> dict:
    modules = []
    edges: Counter = Counter()  # (from_subsys -> to_top_pkg)
    roll: dict[str, dict[str, int]] = defaultdict(
        lambda: {"files": 0, "loc": 0, "defs": 0, "classes": 0}
    )
    for pat in (
        "src/**/*.py",
        "examples/**/*.py",
        "scripts/*.py",
        "tests/**/*.py",
    ):
        for p in sorted(REPO.glob(pat)):
            if "__pycache__" in p.parts:
                continue
            rel = p.relative_to(REPO).as_posix()
            text = read(p)
            loc = sum(1 for ln in text.splitlines() if ln.strip())
            imports = sorted({(a or b).split(".")[0] for a, b in IMPORT_RE.findall(text)})
            sub = subsystem_of(rel)
            for imp in imports:
                if imp in INTRA:
                    edges[(sub, imp)] += 1
            defs, classes = len(DEF_RE.findall(text)), len(CLASS_RE.findall(text))
            # rollup accumulated here where sub/loc/defs/classes are concretely typed
            r = roll[sub]
            r["files"] += 1
            r["loc"] += loc
            r["defs"] += defs
            r["classes"] += classes
            modules.append(
                {
                    "path": rel,
                    "subsystem": sub,
                    "loc": loc,
                    "defs": defs,
                    "classes": classes,
                    "intra_imports": [i for i in imports if i in INTRA],
                }
            )
    return {
        "modules": modules,
        "subsystems": roll,
        "intra_import_edges": {f"{a}->{b}": n for (a, b), n in sorted(edges.items())},
    }


def md_summary(graph, wiki, code) -> str:
    lines: list[str] = []
    P = lines.append
    P("# Doc/code inventory\n")

    P(f"**Documents:** {len(graph['wikilinks'])} markdown files in `docs/` and `wiki/`\n")

    P("## Wikilinks\n")
    P(f"{len(wiki['targets'])} distinct `[[targets]]`, {len(wiki['dangling'])} unresolved:\n")
    for d in wiki["dangling"]:
        P(f"- `[[{d}]]`")
    P("")

    P("## Code map (subsystem rollup)\n")
    P("| subsystem | files | loc | defs | classes |")
    P("|---|---|---|---|---|")
    for name, _prefix in SUBSYSTEMS:
        r = code["subsystems"].get(name)
        if r:
            P(f"| {name} | {r['files']} | {r['loc']} | {r['defs']} | {r['classes']} |")
    P("")
    P("## Intra-repo import edges (seam view)\n")
    P("| edge | files |")
    P("|---|---|")
    for e, n in code["intra_import_edges"].items():
        P(f"| {e} | {n} |")
    P("")
    return "\n".join(lines)


def main() -> None:
    args = sys.argv[1:]
    out = REPO / "build" / "wiki"
    if "--out" in args:
        out = Path(args[args.index("--out") + 1])
        if not out.is_absolute():
            out = REPO / out
    quiet = "--quiet" in args
    out.mkdir(parents=True, exist_ok=True)

    graph = inventory_docs()
    wiki = resolve_wikilinks(graph["wikilinks"])
    code = inventory_code()

    (out / "refgraph.json").write_text(
        json.dumps({**graph, "wikilink_resolution": wiki}, indent=2)
    )
    (out / "codemap.json").write_text(json.dumps(code, indent=2))
    summary = md_summary(graph, wiki, code)
    (out / "inventory-summary.md").write_text(summary)

    if not quiet:
        print(summary)
    print(
        f"\n[wrote {out.relative_to(REPO)}/ : refgraph.json, codemap.json, inventory-summary.md]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
