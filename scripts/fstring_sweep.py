"""Every f-string in `src/`, gated: allowed only where a t-string processor renders, or a raise.

    uv run python scripts/fstring_sweep.py --gate          # the gate `just check` runs
    uv run python scripts/fstring_sweep.py                 # report, no exit code
    uv run python scripts/fstring_sweep.py --init-baseline # bootstrap only

**The rule: an f-string is for a t-string processor's RENDER BACKEND, and
generally nowhere else.** Effective is the PEP 750 showcase; an f-string standing where a
`Template` belongs is what the project argues against. Read it as `isinstance`'s twin on the data
axis — it can be done correctly, it has many edge cases, so it usually is not — and ask what
totality asks of an `isinstance` chain: was a structure available here, and did this flatten it?

**Two designated positions, and for now only two.**

1. **A t-string processor's render backend** — `DESIGNATED`, below. Once the structural decisions
   are made, eager format-then-concat is exactly right and it is the fastest thing Python has
   (26.6 ns/op against `str.format`'s 72.1). The decision has to be ABOVE it.
2. **Creating an exception.** Sanctioned for now, and expected to be structured later — a
   deferral, not a clearance.

Everything else is a finding. The ones that exist today are recorded in the baseline with the
grammar each one builds, so the gate fails on GROWTH and on any entry that no longer reproduces:
the record can only shrink. Where each grammar's processor already lives — a key is `compose_key`,
SQL is the psycopg t-string boundary, a prompt is `effective.channels`, a web search query is
`effective.query`, markdown is `effective.markdown`, HTML/SVG is `tdom.html`
(vendored at `infra/tdom`). A URL path and a data URI have none yet, which is a gap, not a licence.

Keyed on `(path, statics)` and never a line number, so an edit above a finding does not invalidate
its entry — the same reason `key-sweep-baseline.txt` is keyed that way.
"""

import argparse
import ast
import pathlib
import re
import sys

BASELINE = pathlib.Path("scripts/fstring-baseline.txt")

DESIGNATED: frozenset[tuple[str, str]] = frozenset(
    {
        # `grammar` is the key language's own renderer: these turn a decided structure into bytes.
        ("src/effective/keys/grammar.py", "Atom.render"),
        ("src/effective/keys/grammar.py", "Coordinate.render"),
        ("src/effective/keys/grammar.py", "Term.render"),
        ("src/effective/keys/grammar.py", "ParsedKey.render"),
        ("src/effective/keys/grammar.py", "_render_terms"),
        # the registry renders a SKELETON (a template with its holes shown) for `explain`
        ("src/effective/keys/registry.py", "_render_term"),
        # the channel processor's data fence: the tag is chosen before these bytes are composed
        ("src/effective/channels.py", "_fenced"),
        # the judgment processor: `battery` has decided which hole and which element before these
        # render a question's name, its instructions, and a state path in the context
        ("src/effective/judgment.py", "_element"),
        ("src/effective/judgment.py", "_element_instructions"),
        ("src/effective/judgment.py", "_cited"),
        # the markdown processor: every cell is rendered and escaped before a row joins them
        ("src/effective/markdown.py", "_row"),
        # the OTLP encoder, the telemetry processor's target domain: `_name_of` decides a span
        # name's operation and target before this joins them, and a validator `Problem` carries
        # its location as a path until it is printed
        ("src/effective/telemetry.py", "_render_name"),
        ("src/effective/telemetry.py", "Problem.render"),
        # the capture guard, the encoder's last step for content: what to keep, elide and hash
        # is decided above these, which only spell the marker
        ("src/effective/telemetry.py", "_captured"),
        ("src/effective/telemetry.py", "_media_marker"),
    }
)
"""Render backends: a processor's last step, where the structure is already decided and is
mapped into its target domain. A wire encoder at a serialization boundary is one: the telemetry
processor maps a span's structure into OTLP.

Keyed by QUALIFIED name, class included, so designating `Problem.render` exempts no other
`render` in the file. Function-scoped rather than module-scoped on purpose: `grammar.py` also
raises, parses and diagnoses, and a module-wide exemption would clear all of it."""


def _enclosing(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parent: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parent[child] = node
    return parent


def findings(paths: list[str]) -> list[tuple[str, int, str, str]]:
    """`(path, line, statics, why)` for every f-string that is not in a designated position."""
    out = []
    for path in _files(paths):
        text = path.read_text()
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        parent = _enclosing(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.JoinedStr):
                continue
            if not any(isinstance(v, ast.FormattedValue) for v in node.values):
                continue  # no interpolation: an ordinary string wearing an `f`
            chain, cur = [], node
            while cur in parent:
                cur = parent[cur]
                chain.append(cur)
            if any(isinstance(c, ast.Raise) for c in chain):
                continue  # designated: creating an exception
            scopes = [
                c.name
                for c in chain
                if isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            ]
            if scopes and (str(path), ".".join(reversed(scopes))) in DESIGNATED:
                continue  # designated: a t-string processor's render backend
            statics = "".join(
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            out.append((str(path), node.lineno, statics, _grammar(statics)))
    return out


def _grammar(statics: str) -> str:
    """Which grammar this f-string is building — i.e. which processor should own it."""
    # Mermaid borrows `<br/>` for a line break inside a node LABEL and spells its entities
    # `#quot;` rather than `&quot;`. It is not HTML and `tdom` does not own it — inflected
    # markdown, mermaid included, is a later pass.
    if "#quot;" in statics or (
        statics.strip() in {"<br/>", "<br>"} or statics.startswith("<br/>")
    ):
        return "mermaid -> no processor yet"
    # A TAG needs a following attribute or a close: `<reason>` in a diagnostic is a placeholder,
    # not markup, and matching it put three lint messages in the HTML bucket.
    if re.search(r"<!doctype|</[a-zA-Z][\w-]*>|<[a-zA-Z][\w-]*(\s[^>]*)?/?>", statics, re.I) and (
        statics.count("<") > 1 or "/>" in statics or "</" in statics
    ):
        return "html -> tdom.html"
    if re.search(r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE)\b", statics):
        return "sql -> psycopg t-string"
    if statics.startswith("/") or "://" in statics:
        return "url -> no processor yet"
    if statics and " " not in statics and re.fullmatch(r"[a-z0-9;:,/#{}\-]*", statics):
        return "key -> compose_key"
    return "text"


def _files(paths: list[str]) -> list[pathlib.Path]:
    out: list[pathlib.Path] = []
    for p in paths:
        root = pathlib.Path(p)
        out.extend(sorted(root.rglob("*.py")) if root.is_dir() else [root])
    return out


Entry = tuple[str, str]


def load_baseline() -> dict[Entry, str]:
    if not BASELINE.exists():
        return {}
    entries: dict[Entry, str] = {}
    for line in BASELINE.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        path, rest = line.split(" :: ", 1)
        literal, reason = rest.rsplit(" :: ", 1)
        # The statics are stored as a Python literal, so a `::` or a newline inside them cannot
        # be read as the field separator — the first version round-tripped 137 of 389 wrongly.
        entries[(path.strip(), ast.literal_eval(literal))] = reason.strip()
    return entries


def write_baseline(entries: dict[Entry, str]) -> None:
    head = [
        "# f-string baseline — sites recorded as accepted, so the gate can fail on GROWTH.",
        "#",
        "# Format:  <path> :: <statics, as a Python literal> :: <which processor should own it>",
        "#",
        "# An f-string is for a t-string processor's RENDER BACKEND, and (for now) creating an",
        "# exception. Those two are DESIGNATED in fstring_sweep.py and never appear here.",
        "# Everything below is a finding waiting for its processor. The gate fails on a finding",
        "# NOT listed, and on an entry nothing produces — so this file can only shrink.",
        "#",
        "# Keyed on (path, statics), never a line number: an edit above a site keeps its entry.",
        "",
    ]
    body = [f"{p} :: {s!r} :: {r}" for (p, s), r in sorted(entries.items())]
    BASELINE.write_text("\n".join([*head, *body, ""]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["src"])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--init-baseline", action="store_true")
    args = ap.parse_args()

    found = findings(args.paths or ["src"])
    live: dict[Entry, str] = {(p, s): g for p, _, s, g in found}

    if args.init_baseline:
        write_baseline(live)
        print(f"wrote {len(live)} entries to {BASELINE}")
        return 0

    baseline = load_baseline()
    new = {k: v for k, v in live.items() if k not in baseline}
    stale = {k for k in baseline if k not in live}

    if not args.gate:
        for path, line, statics, grammar in sorted(found):
            print(f"{path}:{line}: [{grammar}] {statics[:60]!r}")
        print(f"\n{len(found)} f-strings outside a designated position")
        return 0

    for (path, statics), grammar in sorted(new.items()):
        print(f"{path}: [{grammar}] {statics[:60]!r} — NOT in the baseline")
    for path, statics in sorted(stale):
        print(f"{path}: {statics[:60]!r} — baselined but GONE; drop the line")
    print(f"\nfstring-sweep: {len(baseline)} baselined, {len(new)} new, {len(stale)} stale")
    return 1 if (new or stale) else 0


if __name__ == "__main__":
    sys.exit(main())
