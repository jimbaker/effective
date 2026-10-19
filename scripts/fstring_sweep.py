"""Every f-string in `src/`, gated: allowed where a t-string processor renders and where an
exception says itself.

    uv run python scripts/fstring_sweep.py --gate                # the gate `just check` runs
    uv run python scripts/fstring_sweep.py --gate --touched main # where changes are written
    uv run python scripts/fstring_sweep.py                       # report, no exit code
    uv run python scripts/fstring_sweep.py --init-baseline       # bootstrap, keeping reasons

**The rule: an f-string is for a t-string processor's RENDER BACKEND, and
generally nowhere else.** Effective is the PEP 750 showcase; an f-string standing where a
`Template` belongs is what the project argues against. Read it as `isinstance`'s twin on the data
axis — it can be done correctly, it has many edge cases, so it usually is not — and ask what
totality asks of an `isinstance` chain: was a structure available here, and did this flatten it?

**Two designated positions.**

1. **A t-string processor's render backend** — `DESIGNATED`, below. Once the structural decisions
   are made, eager format-then-concat is exactly right and it is the fastest thing Python has
   (26.6 ns/op against `str.format`'s 72.1). The decision has to be ABOVE it.
2. **A structured exception's `__str__`.** The exception keeps its context in plain fields, and
   says itself from them, so the structure is decided before the message is.

Everything else is a finding, a raise that builds its message included. The ones that exist today
are recorded in the baseline with the grammar each one builds, and a count where several sites
share a key, so the gate fails on GROWTH and on any entry that no longer reproduces: the record
can only shrink. With `--touched`, run where changes are written, a raise is also refused on any
line a branch changed, so a change structures each exception it touches.

Where each grammar's processor already lives: a key is `compose_key`, SQL is the psycopg
t-string boundary, a prompt is `effective.channels`, a web search query is `effective.query`,
markdown is `effective.markdown`, HTML/SVG is `tdom.html` (vendored at `infra/tdom`). A URL path
and a data URI have none yet, which is a gap to fill.

Keyed on `(path, statics)` and never a line number, so an edit above a finding does not invalidate
its entry — the same reason `key-sweep-baseline.txt` is keyed that way.
"""

import argparse
import ast
import builtins
import importlib.util
import pathlib
import re
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Iterator

BASELINE = pathlib.Path("scripts/fstring-baseline.txt")
PRIVATE_BASELINE = pathlib.Path("scripts/fstring-private-baseline.txt")
"""A second baseline, if present; the gate reads both."""
MANIFEST = pathlib.Path("scripts/public-manifest.txt")
"""The files a tree ships, if present; a site in a file it does not list belongs to the second
baseline. Without it, every file ships."""

RAISE = "raise -> structured exception"

DESIGNATED: frozenset[tuple[str, str]] = frozenset(
    {
        # `grammar` is the key language's own renderer: these turn a decided structure into bytes.
        ("src/effective/keys/grammar.py", "Atom.render"),
        ("src/effective/keys/grammar.py", "Coordinate.render"),
        ("src/effective/keys/grammar.py", "_render_term"),
        ("src/effective/keys/grammar.py", "render_terms"),
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
        # the effect's spelling of a schema class: the bytes `repr` gives the same class inside a
        # generic, so a bare class and a generic of it spell it alike
        ("src/effective/handlers/base.py", "_spelled"),
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


def _exception_classes(trees: list[ast.Module]) -> frozenset[str]:
    """The class names in `trees` that derive from an exception, read by simple name: a builtin
    exception, or a class whose base already is one, to a fixed point. A name is one only when
    every class of that name is, so two classes sharing a name never lend each other the
    designation."""
    definitions: dict[str, list[set[str]]] = {}
    for tree in trees:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                definitions.setdefault(node.name, []).append(
                    {
                        base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
                        for base in node.bases
                    }
                )
    known = {
        name
        for name, value in vars(builtins).items()
        if isinstance(value, type) and issubclass(value, BaseException) and name not in definitions
    }
    while grown := {
        name
        for name, each in definitions.items()
        if name not in known and all(bases & known for bases in each)
    }:
        known |= grown
    return frozenset(known)


def _says_an_exception(chain: list[ast.AST], exceptions: frozenset[str]) -> bool:
    """Whether the innermost function in `chain` is the `__str__` of an exception class."""
    functions = [c for c in chain if isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef)]
    if not functions or functions[0].name != "__str__":
        return False
    above = chain[chain.index(functions[0]) + 1 :]
    return bool(above) and isinstance(above[0], ast.ClassDef) and above[0].name in exceptions


def _designated(path: pathlib.Path, chain: list[ast.AST], exceptions: frozenset[str]) -> bool:
    """Whether an f-string under `chain` stands in a designated position."""
    if any(isinstance(c, ast.Raise) for c in chain):
        return False  # a raise builds its message before any structure is decided
    if _says_an_exception(chain, exceptions):
        return True  # a structured exception says itself
    scopes = [
        c.name
        for c in chain
        if isinstance(c, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    ]
    return bool(scopes) and (str(path), ".".join(reversed(scopes))) in DESIGNATED


def findings(paths: list[str]) -> list[tuple[str, int, str, str]]:
    """`(path, line, statics, why)` for every f-string that is not in a designated position."""
    out = []
    parsed = []
    for path in _files(paths):
        try:
            parsed.append((path, ast.parse(path.read_text())))
        except SyntaxError:
            continue
    exceptions = _exception_classes([tree for _, tree in parsed])
    for path, tree in parsed:
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
            if _designated(path, chain, exceptions):
                continue
            statics = "".join(
                v.value
                for v in node.values
                if isinstance(v, ast.Constant) and isinstance(v.value, str)
            )
            raised = any(isinstance(c, ast.Raise) for c in chain)
            out.append((str(path), node.lineno, statics, RAISE if raised else _grammar(statics)))
    return out


def _interpolates(node: ast.AST) -> bool:
    return any(
        isinstance(n, ast.JoinedStr) and any(isinstance(v, ast.FormattedValue) for v in n.values)
        for n in ast.walk(node)
    )


def _builds_its_message(raised: ast.Raise, bound: set[str]) -> bool:
    """Whether a raise interpolates an f-string itself, or names one bound in its own scope."""
    return _interpolates(raised) or any(
        isinstance(n, ast.Name) and n.id in bound for n in ast.walk(raised)
    )


def _raises_by_scope(tree: ast.Module) -> list[tuple[ast.Raise, set[str]]]:
    """Each raise, with the names its scope binds to an interpolating f-string."""
    out = []
    for scope in [tree, *(n for n in ast.walk(tree) if isinstance(n, _FUNCTIONS))]:
        inner = list(_walk_scope(scope))
        bound = {
            target.id
            for n in inner
            if isinstance(n, ast.Assign) and _interpolates(n.value)
            for target in n.targets
            if isinstance(target, ast.Name)
        }
        out.extend((n, bound) for n in inner if isinstance(n, ast.Raise))
    return out


def _walk_scope(scope: ast.AST) -> Iterator[ast.AST]:
    """The nodes of `scope`, stopping at a nested function, which is its own scope."""
    for child in ast.iter_child_nodes(scope):
        yield child
        if not isinstance(child, _FUNCTIONS):
            yield from _walk_scope(child)


_FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def touched_raises(paths: list[str], changed: dict[str, set[int]]) -> list[tuple[str, int]]:
    """`(path, line)` for every raise that builds its message with an f-string and spans a line in
    `changed`, a path's changed lines."""
    out = []
    for path in _files(paths):
        if not (lines := changed.get(str(path))):
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        out.extend(
            (str(path), raised.lineno)
            for raised, bound in _raises_by_scope(tree)
            if _builds_its_message(raised, bound)
            and lines & set(range(raised.lineno, (raised.end_lineno or raised.lineno) + 1))
        )
    return sorted(out)


def changed_lines(base: str) -> dict[str, set[int]] | None:
    """The lines the working tree adds or changes since the merge base of `base`, or of
    `origin/<base>`, and `HEAD`, by path; a deletion marks the lines on either side of it. `None`
    when neither resolves, as in a shallow checkout."""
    for ref in (base, f"origin/{base}"):
        merge = subprocess.run(
            ["git", "merge-base", ref, "HEAD"], capture_output=True, text=True, check=False
        )
        if merge.returncode == 0:
            break
    else:
        return None
    # every option that shapes the headers and hunks is fixed here, whatever the user's config
    diff = subprocess.run(
        [
            *("git", "-c", "core.quotePath=false", "diff", "-U0", "--inter-hunk-context=0"),
            *("--no-color", "--no-ext-diff", "--src-prefix=a/", "--dst-prefix=b/"),
            merge.stdout.strip(),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    out: dict[str, set[int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[6:].rstrip("\t") if line.startswith("+++ b/") else None
        elif path and (hunk := re.match(r"@@ -\S+ \+(\d+)(?:,(\d+))? @@", line)):
            start, count = int(hunk[1]), int(hunk[2] or 1)
            lines = range(start, start + count) if count else (start, start + 1)
            out.setdefault(path, set()).update(lines)
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


def load_baseline() -> dict[Entry, tuple[str, int]]:
    """Each entry's reason and how many sites share its key."""
    entries: dict[Entry, tuple[str, int]] = {}
    for line in (
        line
        for file in (BASELINE, PRIVATE_BASELINE)
        if file.exists()
        for line in file.read_text().splitlines()
    ):
        if not line.strip() or line.startswith("#"):
            continue
        path, rest = line.split(" :: ", 1)
        literal, reason = rest.rsplit(" :: ", 1)
        count = 1
        if reason.strip().isdigit():
            count = int(reason)
            literal, reason = literal.rsplit(" :: ", 1)
        # The statics are stored as a Python literal, so a `::` or a newline inside them cannot
        # be read as the field separator.
        entries[(path.strip(), ast.literal_eval(literal))] = (reason.strip(), count)
    return entries


def write_baseline(entries: dict[Entry, tuple[str, int]]) -> None:
    """Write `entries` to the shared baseline, and a private file's to the private baseline where
    this tree keeps a manifest."""
    head = [
        "# f-string baseline: sites recorded as accepted, so the gate can fail on GROWTH.",
        "#",
        "# Format:  <path> :: <statics, as a Python literal> :: <which processor should own it>",
        "#          [:: <how many sites share the key, when more than one>]",
        "#",
        "# An f-string is for a t-string processor's RENDER BACKEND and a structured exception's",
        "# `__str__`. Those two are DESIGNATED in fstring_sweep.py and never appear here.",
        "# Everything below is a finding waiting for its processor, and a raise is waiting for a",
        "# structured exception. The gate fails on a site NOT counted, and on a count nothing",
        "# produces, so this file can only shrink.",
        "#",
        "# Keyed on (path, statics), never a line number: an edit above a site keeps its entry.",
        "",
    ]
    private_head = [
        "# Private entries for scripts/fstring-baseline.txt: sites in files that do not ship."
        " Same format.",
    ]
    ships = shipping()
    shared = {k: v for k, v in entries.items() if ships(k[0])}
    private = {k: v for k, v in entries.items() if not ships(k[0])}
    BASELINE.write_text("\n".join([*head, *_lines(shared), ""]))
    if private:
        PRIVATE_BASELINE.write_text("\n".join([*private_head, *_lines(private), ""]))
    else:
        PRIVATE_BASELINE.unlink(missing_ok=True)


def _lines(entries: dict[Entry, tuple[str, int]]) -> list[str]:
    return [
        f"{p} :: {s!r} :: {r}" + (f" :: {n}" if n > 1 else "")
        for (p, s), (r, n) in sorted(entries.items())
    ]


def shipping() -> Callable[[str], bool]:
    """Whether a path ships, by this tree's manifest; every path ships in a tree without one."""
    reader = MANIFEST.parent / "sync_public.py"
    if not (MANIFEST.exists() and reader.exists()):
        return lambda _path: True
    spec = importlib.util.spec_from_file_location("sync_public", reader)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module.read_manifest(MANIFEST.read_text()).ships


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", default=["src"])
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--init-baseline", action="store_true")
    ap.add_argument("--touched", metavar="BASE", help="refuse an f-string raise on a changed line")
    args = ap.parse_args()

    found = findings(args.paths or ["src"])
    counts = Counter((p, s) for p, _, s, _ in found)
    raised = Counter((p, s) for p, _, s, g in found if g == RAISE)
    live: dict[Entry, tuple[str, int]] = {(p, s): (g, counts[(p, s)]) for p, _, s, g in found}

    baseline = load_baseline()
    if args.init_baseline:
        # an entry keeps the reason a reader wrote for it; a new one gets the grammar's
        write_baseline({k: (baseline.get(k, v)[0], v[1]) for k, v in live.items()})
        print(f"wrote {len(live)} entries, {len(found)} sites")
        return 0

    new = {k: v for k, v in live.items() if v[1] > baseline.get(k, ("", 0))[1]}
    stale = {k: v for k, v in baseline.items() if v[1] > live.get(k, ("", 0))[1]}
    # a raise's message built above it still counts under the raise's key
    hoisted = {k for k, (r, _) in baseline.items() if r == RAISE and counts[k] > raised[k]}

    if not args.gate:
        for path, line, statics, grammar in sorted(found):
            print(f"{path}:{line}: [{grammar}] {statics[:60]!r}")
        print(f"\n{len(found)} f-strings outside a designated position")
        return 0

    for (path, statics), (grammar, n) in sorted(new.items()):
        had = baseline.get((path, statics), ("", 0))[1]
        print(f"{path}: [{grammar}] {statics[:60]!r}: {n} sites, {had} in the baseline")
    for (path, statics), (_, n) in sorted(stale.items()):
        left = live.get((path, statics), ("", 0))[1]
        fix = "lower the count" if left else "drop the line"
        print(f"{path}: {statics[:60]!r}: {n} baselined, {left} left; {fix}")
    for path, statics in sorted(hoisted):
        print(f"{path}: {statics[:60]!r}: a raise's message, now built outside the raise")
    print(
        f"\nfstring-sweep: {len(baseline)} baselined, {len(new)} new, {len(stale)} stale,"
        f" {len(hoisted)} hoisted"
    )
    touched: list[tuple[str, int]] = []
    unresolved = False
    if args.touched:
        if unresolved := (changed := changed_lines(args.touched)) is None:
            print(
                f"fstring-sweep: neither {args.touched} nor origin/{args.touched} resolves;"
                f" fetch it (`git fetch origin {args.touched}`) so the changed lines can be read"
            )
        else:
            touched = touched_raises(args.paths or ["src"], changed)
            for path, line in touched:
                print(
                    f"{path}:{line}: a changed raise builds its message; structure the exception"
                )
            print(f"fstring-sweep: {len(touched)} changed raises since {args.touched}")
    return 1 if (new or stale or hoisted or touched or unresolved) else 0


if __name__ == "__main__":
    sys.exit(main())
