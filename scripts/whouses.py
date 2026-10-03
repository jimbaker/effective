"""Who actually uses this symbol: candidates by syntax, resolved by `ty`'s language server.

    uv run python scripts/whouses.py src/effective/keys/grammar.py:961
    uv run python scripts/whouses.py src/effective/keys/grammar.py:961 --all
    uv run python scripts/whouses.py <path:line> [roots...] [--references] [--callers] [--timing]

Two rungs, because neither is enough alone: ast-grep finds every site of a shape cheaply and
over-generates, `ty` decides which of them are real. Three rules the code cannot state for itself:

- **Enumerate with RULES, not patterns.** `$X.attr` is blind to a PEP 634 keyword pattern and
  reports the enclosing match's range rather than the node's own. One shipped as a false zero and
  the other as a crash.
- **Never resolve a keyword position.** `case Cls(attr=x)` is aimed at `Cls`, and resolving the
  keyword itself returns a homonym: a same-named parameter in a nearby function, confidently wrong.
- **UNRESOLVED is a third outcome**, reported under `--all`, never merged into "not a use". An
  inference miss and a genuine non-use are different facts.

A sweep costs the FILES it touches rather than the sites, because `ty` memoizes through Salsa.
`--timing` recomputes the ratio rather than this docstring pinning it.

`ty` rather than jedi, because jedi does not resolve PEP 634 capture patterns. The export sweep
is `scripts/dead_exports.py`.
"""

import argparse
import ast
import functools
import json
import pathlib
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import NamedTuple

ROOT = pathlib.Path.cwd()


class Server:
    """A `ty server` session, spoken to over stdio. Closed by `close`, or by the process ending."""

    def __init__(self, root: pathlib.Path) -> None:
        self.proc = subprocess.Popen(
            ["uvx", "ty", "server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._id = 0
        self._opened: set[pathlib.Path] = set()
        rid = self._send(
            "initialize",
            {
                "processId": None,
                "rootUri": root.as_uri(),
                "capabilities": {
                    "textDocument": {
                        "definition": {"linkSupport": True},
                        "references": {},
                        "callHierarchy": {},
                    }
                },
                "workspaceFolders": [{"uri": root.as_uri(), "name": root.name}],
            },
        )
        self._await(rid)
        self._send("initialized", {}, notify=True)

    def _send(self, method: str, params: dict, *, notify: bool = False) -> int | None:
        message: dict = {"jsonrpc": "2.0", "method": method, "params": params}
        if not notify:
            self._id += 1
            message["id"] = self._id
        body = json.dumps(message).encode()
        assert self.proc.stdin is not None
        self.proc.stdin.write(b"Content-Length: %d\r\n\r\n%s" % (len(body), body))
        self.proc.stdin.flush()
        return None if notify else self._id

    def _read(self) -> dict | None:
        out = self.proc.stdout
        assert out is not None
        length = None
        while True:
            line = out.readline()
            if not line:
                return None
            text = line.decode().strip()
            if text.lower().startswith("content-length:"):
                length = int(text.split(":")[1])
            elif text == "" and length is not None:
                return json.loads(out.read(length))

    def _await(self, rid: int | None) -> dict | None:
        while (message := self._read()) is not None:
            if message.get("id") == rid:
                return message.get("result")
        return None

    def open(self, path: pathlib.Path) -> None:
        if path in self._opened:
            return
        self._send(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": path.as_uri(),
                    "languageId": "python",
                    "version": 1,
                    "text": path.read_text(),
                }
            },
            notify=True,
        )
        self._opened.add(path)

    def definition(self, path: pathlib.Path, line: int, column: int) -> tuple[str, int] | None:
        """`(path, line)` the name at this 0-based position resolves to, or `None`."""
        self.open(path)
        result = self._await(
            self._send(
                "textDocument/definition",
                {
                    "textDocument": {"uri": path.as_uri()},
                    "position": {"line": line, "character": column},
                },
            )
        )
        if not result:
            return None
        first = result[0] if isinstance(result, list) else result
        # A LocationLink carries `targetUri`/`targetSelectionRange`; a plain Location carries
        # `uri`/`range`. `ty` correctly refuses the `or` chain without this, because a server may
        # send neither and `None.removeprefix` is what that would reach.
        uri = first.get("targetUri") or first.get("uri")
        span = first.get("targetSelectionRange") or first.get("range")
        if uri is None or span is None:
            return None
        return str(uri).removeprefix("file://"), span["start"]["line"] + 1

    def references(self, path: pathlib.Path, line: int, column: int) -> set[tuple[str, int]]:
        """Every `(path, line)` the server calls a reference of the name at this position.

        Workspace-wide and unbounded by any candidate set: `Key.scope` answers 30 locations across
        `src/` and `tests/` where a `src`-rooted sweep enumerates 10. **It does not replace the
        enumerator**, because it is blind to a PEP 634 keyword pattern: measured on three
        attributes with a keyword use, none appears, and `GatherBranch.gather` answers zero against
        a real use. Two enumerators that miss different things are worth running together, and
        `--references` reports where they disagree rather than picking one."""
        self.open(path)
        found = self._await(
            self._send(
                "textDocument/references",
                {
                    "textDocument": {"uri": path.as_uri()},
                    "position": {"line": line, "character": column},
                    "context": {"includeDeclaration": False},
                },
            )
        )
        return {
            (_relative(row["uri"]), row["range"]["start"]["line"] + 1) for row in (found or [])
        }

    def callers(self, path: pathlib.Path, line: int, column: int) -> list[tuple[str, str, int]]:
        """Which FUNCTIONS call this, by name: the question neither rung answers.

        `references` and the enumerator both report lines that mention a name; a call hierarchy
        reports the enclosing definition, so `kind_of` comes back as `is_tag` and `__post_init__`
        rather than as two line numbers."""
        self.open(path)
        items = self._await(
            self._send(
                "textDocument/prepareCallHierarchy",
                {
                    "textDocument": {"uri": path.as_uri()},
                    "position": {"line": line, "character": column},
                },
            )
        )
        if not items:
            return []
        incoming = self._await(self._send("callHierarchy/incomingCalls", {"item": items[0]})) or []
        return [
            (
                call["from"]["name"],
                _relative(call["from"]["uri"]),
                call["from"]["range"]["start"]["line"] + 1,
            )
            for call in incoming
        ]

    def close(self) -> None:
        self.proc.terminate()


def _relative(uri: str) -> str:
    return str(uri).removeprefix("file://").replace(f"{ROOT}/", "")


def ast_grep() -> str:
    """The PINNED CLI, resolved, never the bare name, which resolves against PATH.

    `ast-grep-cli` is co-versioned with the `ast-grep-py` binding `effective.lint` uses, and the
    pin's comment names this script as the reason it exists. Shelling to `ast-grep` walked past it
    to whatever the machine carried: linuxbrew's 0.45.1 here against the project's 0.42.3. The two
    agree on every rule sent below, which is why it stayed invisible."""
    pinned = pathlib.Path(sys.executable).with_name("ast-grep")
    return str(pinned) if pinned.exists() else "ast-grep"


def _scan(rule: dict, roots: list[str]) -> list[dict]:
    """Every node matching an ast-grep RULE, reported as its own range rather than its parent's.

    Rules and not patterns, for two reasons the pattern DSL cannot answer. `$X.name` reports the
    range of the whole match, so an attribute on the third line of a multi-line receiver was looked
    for on the first and raised. And `$X.name` cannot express a PEP 634 keyword pattern at all,
    159 of them under `src/effective`, invisible. A `kind` rule reaches both."""
    inline = json.dumps(
        {
            "id": "whouses",
            "language": "python",
            "severity": "info",
            "message": "candidate",
            "rule": rule,
        }
    )
    done = subprocess.run(
        [
            ast_grep(),
            "scan",
            "--inline-rules",
            inline,
            # A `.gitignore`d file, a dot-directory and a symlink are skipped writing NEITHER
            # stderr nor a non-zero exit, so the guard below cannot see them. The stderr rule holds
            # forward and not backward, and this closes the half it cannot reach.
            "--no-ignore",
            "hidden",
            "--no-ignore",
            "vcs",
            *roots,
            "--json=compact",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # **The exit code is not the signal.** A missing root and a rule the grammar rejects both print
    # to stderr and exit ZERO, so a `returncode` check passes and `[]` comes back: an empty domain
    # for a reason unrelated to the property, which is the answer this whole file exists to refuse.
    # A run that legitimately finds nothing writes no stderr, so stderr is the discriminator.
    if done.returncode != 0 or done.stderr.strip():
        raise SystemExit(f"ast-grep failed on {roots}: {done.stderr.strip() or done.returncode}")
    return json.loads(done.stdout or "[]")


@functools.cache
def _lines(path: pathlib.Path) -> list[str]:
    return path.read_text().splitlines()


def _site(file: str, line: int, column: int) -> tuple[pathlib.Path, int, int, str]:
    path = pathlib.Path(file)
    return path, line, column, _lines(path)[line].strip()


def _at(row: dict) -> tuple[pathlib.Path, int, int, str]:
    return _site(row["file"], row["range"]["start"]["line"], row["range"]["start"]["column"])


def _start(row: dict) -> tuple[int, int]:
    return row["range"]["start"]["line"], row["range"]["start"]["column"]


def _end(row: dict) -> tuple[int, int]:
    return row["range"]["end"]["line"], row["range"]["end"]["column"]


def _class_offset(pattern: str) -> int:
    """How far into a class pattern its class NAME starts.

    Derived from the token, never from the text before the paren. `case mod . Name(...)` and a
    head split across lines both parse, and measuring `len(head) - len(after the last dot)` put the
    cursor on a space or on the module in each, which files a genuine use as a non-use, silently.
    Zero instances in the tree today, and the point is that a latent version of this defect is the
    one already shipped once."""
    if (match := re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", pattern)) is None:
        raise SystemExit(f"class pattern with no resolvable class name: {pattern[:60]!r}")
    if "\n" in pattern[: match.start(1)]:
        raise SystemExit(f"class pattern head spans lines; cursor undefined: {pattern[:60]!r}")
    return match.start(1)


def _keyword_pattern_sites(
    name: str, roots: list[str]
) -> list[tuple[pathlib.Path, int, int, str]]:
    """Keyword patterns naming `name`, each aimed at its CLASS rather than at itself.

    At a keyword position `ty` does not decline to answer; it answers with a homonym. At
    `case Cleared(granted=advanced)` the keyword resolves to a same-named parameter one file away.
    The class is safe, and PEP 634 supplies the rest: a keyword in a class pattern must name an
    attribute of the matched class, so attributing by name is a language guarantee. The cursor is
    the class node's own start, moved to the last segment of a dotted name; the join takes the
    innermost enclosing pattern, because class patterns nest."""
    classes = _scan({"kind": "class_pattern"}, roots)
    sites = []
    for row in _scan({"kind": "keyword_pattern", "regex": rf"^{re.escape(name)}\s*="}, roots):
        enclosing = [
            cls
            for cls in classes
            if cls["file"] == row["file"] and _start(cls) <= _start(row) and _end(row) <= _end(cls)
        ]
        if not enclosing:
            continue
        innermost = max(enclosing, key=_start)
        line, column = _start(innermost)
        sites.append(_site(row["file"], line, column + _class_offset(innermost["text"])))
    return sites


def member_sites(
    name: str, roots: list[str], *, member: bool
) -> tuple[list[tuple[pathlib.Path, int, int, str]], list[tuple[pathlib.Path, int, int, str]]]:
    """Every site of the right SHAPE, split by which DEFINITION each half resolves to.

    An attribute read resolves to the attribute; a keyword pattern resolves to the class it was
    aimed at. Merging them would make the caller compare both against one target and file every
    keyword site under "resolved elsewhere": the false zero this tool exists to refuse, wearing a
    different costume."""
    escaped = re.escape(name)
    if not member:
        # Every spelling the name is reachable under, not just its own. `as_policy` reported
        # `USES (0)` beside three import lines, because every call site spells `budget_policy`,
        # the same self-contradiction the export sweep had, in the half that did not get the rule.
        spellings = "|".join(sorted(re.escape(n) for n in {name} | _aliases_of(name, roots)))
        rows = _scan({"kind": "identifier", "regex": f"^({spellings})$"}, roots)
        return [_at(row) for row in rows], []

    reads = _scan(
        {
            "kind": "identifier",
            "regex": f"^{escaped}$",
            "inside": {"kind": "attribute", "field": "attribute"},
        },
        roots,
    )
    return [_at(row) for row in reads], _keyword_pattern_sites(name, roots)


def _enclosing_class(path: pathlib.Path, line: int) -> int | None:
    """Where the class that DECLARES the member at `line` is defined, or `None` for a free name.

    The same chain walk `_named_at` uses, and for the same reason. Walking one level answered
    `None` for an attribute declared under `if TYPE_CHECKING:`, so the keyword site resolved to its
    class, was compared against nothing, and was filed as resolving elsewhere: the tool computing
    the right answer and discarding it."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if any(
            child.lineno == line
            for child, _ in _scoped(node, in_class=True)
            if hasattr(child, "lineno")
        ):
            return node.lineno
    return None


def _named_at(path: pathlib.Path, line: int) -> tuple[str, str, bool]:
    """The name DEFINED at this line, read off the tree rather than split out of the text.

    Whether a name is reached through a dot is a question about SCOPE, not about the parent node,
    which is why `_scoped` walks the chain and stops at the first `def`. Reading the tree is also
    what makes a miss a refusal: deriving a name from the words on a line answers `USES (0)` for a
    docstring."""
    text = path.read_text()
    tree = ast.parse(text)
    for node, within_class in _scoped(tree, in_class=False):
        if getattr(node, "lineno", None) != line:
            continue
        source = text.splitlines()[line - 1].strip()
        match node:
            case ast.FunctionDef(name=named) | ast.AsyncFunctionDef(name=named):
                return named, source, within_class
            case ast.ClassDef(name=named):
                return named, source, False
            case (
                ast.Assign(targets=[ast.Name(id=named)])
                | ast.AnnAssign(target=ast.Name(id=named))
                | ast.TypeAlias(name=ast.Name(id=named))
            ):
                return named, source, within_class
            case _:
                continue
    if line == 1 and ast.get_docstring(tree) is not None:
        # A MODULE, whose docstring is a big share of the tree's prose, 149 of them in `src/`
        # holding 28,933 words. Its "callers" are its importers, which is a different question
        # with a different kind of answer; `_importers` says so rather than dressing it up.
        # Gated on there BEING a docstring: line 1 of a file that opens with an import defines
        # nothing, and answering `_importers` there would be the reassuring answer again.
        return "", text.splitlines()[0].strip(), False
    raise SystemExit(
        f"{path}:{line} defines nothing. Point at the line that names it: a `def`, a `class`, "
        f"or an assignment"
    )


def _scoped(node: ast.AST, *, in_class: bool) -> Iterator[tuple[ast.AST, bool]]:
    """Every node, paired with whether a CLASS BODY encloses it.

    A `def` ends the class body for this purpose: a name bound inside a method is a local, not an
    attribute, however deeply the class nests around it."""
    for child in ast.iter_child_nodes(node):
        yield child, in_class
        match child:
            case ast.ClassDef():
                yield from _scoped(child, in_class=True)
            case ast.FunctionDef() | ast.AsyncFunctionDef():
                yield from _scoped(child, in_class=False)
            case _:
                yield from _scoped(child, in_class=in_class)


def _importers(path: pathlib.Path, roots: list[str]) -> int:
    """Who imports this MODULE: the callers question for a module docstring.

    Syntactic, and it says so in the output: an import statement names its module directly, so
    there is no name for `ty` to resolve and nothing to narrow. Reporting it under the same `USES`
    heading as a resolved result would make two different kinds of evidence look alike, which is
    the failure this tool exists to prevent one level down."""
    module = ".".join(path.with_suffix("").parts[1:])
    print(f"{path}  module `{module}`")
    print("  importers, found SYNTACTICALLY: an import names its module, so ty resolves nothing\n")
    pattern = f"^\\s*(from {re.escape(module)} import|import {re.escape(module)})"
    done = subprocess.run(
        ["grep", "-rnE", "--include=*.py", pattern, *roots],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    print(f"IMPORTERS ({len(lines)}):")
    for line in lines:
        print(f"   {line.strip()[:110]}")
    return 0


def _column_of(path: pathlib.Path, line: int, name: str) -> int:
    """Where `name` sits on its own definition line: the position the server wants."""
    return _lines(path)[line - 1].index(name)


def _report_references(referenced: set[tuple[str, int]], resolved: set[tuple[str, int]]) -> None:
    """The two enumerators side by side, and the DISAGREEMENT, which is the point.

    Neither is a superset. `ty`'s reference search reaches files no candidate set was pointed at,
    and is blind to a PEP 634 keyword pattern; the rule enumerator sees the keyword pattern and
    stops at the roots you passed. Reporting the intersection alone would hide both."""
    print(f"\nREFERENCES, per ty ({len(referenced)}):")
    for where, line in sorted(referenced):
        print(f"   {where}:{line}")
    only_rules = sorted(resolved - referenced)
    only_ty = sorted(referenced - resolved)
    print(
        f"\nDISAGREEMENT: {len(only_rules)} the rules found alone, {len(only_ty)} ty found alone"
    )
    for where, line in only_rules:
        print(f"   rules only   {where}:{line}   (a keyword pattern, or a shape ty scopes out)")
    for where, line in only_ty:
        print(f"   ty only      {where}:{line}   (outside the roots passed, or a re-export)")


class Site(NamedTuple):
    """One indexed occurrence of a name, and HOW it occurs.

    The kind is not decoration. An `import` alias resolves to the definition like any other site,
    so an index that forgets which sites were imports counts a package `__init__` that merely
    re-exports a name as one of its consumers."""

    path: pathlib.Path
    line: int
    column: int
    kind: str  # "name" | "attribute" | "alias"


def _name_index(roots: list[str]) -> dict[str, list[Site]]:
    """Every site that could name something, indexed by name, via `ast` rather than ast-grep.

    This is the one workload where `ast` wins, and it wins by two orders. ast-grep costs a flat
    22 ms per query; an index costs 128 ms once and answers free, so the crossover is around six
    names and an `__all__` sweep asks four hundred. A per-name ast-grep sweep of `src/` would be
    9.4 s of subprocess against a tenth of a second here.

    **Attributes are indexed too, at the attribute's own position.** Leaving them out made the
    export sweep contradict a single-symbol query in one screen: `elkjs.available` has a consumer
    at `dashboard.py:355` and the sweep called it unused, because `mod.NAME` is not an `ast.Name`.
    Nine of thirty-nine reported zeros were that.

    An `import x as y` binds two names and both are indexed; matching only the bound one hid
    `parked.answer`'s third consumer behind `answer_park`."""
    index: dict[str, list[Site]] = {}

    def record(name: str, node: ast.AST, kind: str, line: int, column: int) -> None:
        index.setdefault(name, []).append(Site(path, line, column, kind))

    for root in roots:
        for path in sorted(pathlib.Path(root).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                match node:
                    case ast.Name(id=name):
                        record(name, node, "name", node.lineno, node.col_offset)
                    case ast.Attribute(
                        attr=attr, end_lineno=int() as end, end_col_offset=int() as col
                    ):
                        record(attr, node, "attribute", end, col - len(attr))
                    case ast.alias(name=imported, asname=bound):
                        record(
                            imported.rpartition(".")[2],
                            node,
                            "alias",
                            node.lineno,
                            node.col_offset,
                        )
                        if bound is not None and bound != imported:
                            record(bound, node, "alias", node.lineno, node.col_offset)
                    case _:
                        continue
    return index


def _aliases_of(name: str, roots: list[str]) -> set[str]:
    """What any file under `roots` renamed `name` to when it imported it.

    A rename is not a different symbol, since `ty` resolves `budget_policy` straight back to
    `as_policy`, but a sweep for the original spelling never puts the question. 41 names under
    `src` and `tests` are imported under a different local name, and `kind_of`, this branch's own
    worked example, is one of them."""
    tail = rf"^(?:[\w.]*\.)?{re.escape(name)}\s+as\s+(\w+)$"
    bound = set()
    for row in _scan({"kind": "aliased_import"}, roots):
        if (match := re.match(tail, row["text"].strip())) is not None:
            bound.add(match.group(1))
    return bound


def _report_timing(cold: list[float], warm: list[float]) -> None:
    """The cost model, recomputed. The MEDIAN RATIO, and only against the two medians.

    The figure this replaced divided the very first query, which also pays for whatever the session
    had not yet done, by the warm median. Two true measurements fused into a ratio that follows
    from neither, in five documents, with no committed way to recheck it. Hence this flag."""

    def median(xs: list[float]) -> float:
        return sorted(xs)[len(xs) // 2] if xs else float("nan")

    print(
        f"\n  first query in a NEW file  {median(cold):7.1f} ms median  "
        f"{max(cold, default=0):7.1f} max  (n={len(cold)})"
    )
    print(
        f"  repeat in a SEEN file     {median(warm):7.1f} ms median  "
        f"{max(warm, default=0):7.1f} max  (n={len(warm)})"
    )
    print(f"  whole sweep               {(sum(cold) + sum(warm)) / 1000:7.2f} s")
    if warm and median(warm):
        print(f"  median ratio              {median(cold) / median(warm):7.0f}x")


SYMBOL_ROOTS = ("src",)
"""Where a single symbol's uses are looked for when the caller names no root."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Who uses this symbol: candidates by syntax, resolved by ty."
    )
    parser.add_argument(
        "target", help="path:line of the DEFINITION, e.g. src/effective/keys/grammar.py:961"
    )
    # No default here. One declared beside a second one in the body is how this shipped answering
    # over `src` while its help, its comment and the task page all said `src tests`.
    parser.add_argument("roots", nargs="*")
    parser.add_argument("--all", action="store_true", help="also list what resolved elsewhere")
    parser.add_argument(
        "--timing", action="store_true", help="report per-query latency, split new-file vs repeat"
    )
    parser.add_argument(
        "--references",
        action="store_true",
        help="also run ty's own reference search, and report where the two enumerators disagree",
    )
    parser.add_argument(
        "--callers", action="store_true", help="which FUNCTIONS call this, by name"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """`argv` is taken so a test can RUN this. Nothing did, which is why three review passes and a
    pinned fix all missed a default the documented command contradicts."""
    args = _parser().parse_args(argv)

    path_text, _, line_text = args.target.rpartition(":")
    target = (str(pathlib.Path(path_text).resolve()), int(line_text))
    attribute, source, member = _named_at(pathlib.Path(path_text), target[1])
    if not attribute:
        return _importers(pathlib.Path(path_text), args.roots or list(SYMBOL_ROOTS))

    roots = args.roots or list(SYMBOL_ROOTS)
    reads, keywords = member_sites(attribute, roots, member=member)
    sites = reads + keywords
    declared = _enclosing_class(pathlib.Path(path_text), target[1]) if member else None
    # A keyword pattern is a use of `Cls.attr` exactly when it resolves to `Cls`; PEP 634 supplies
    # the rest. Without its own target every one of them lands under "resolved elsewhere".
    by_class = {(site[0], site[1], site[2]) for site in keywords}
    class_target = (target[0], declared) if declared is not None else None
    print(f"{args.target}  {source}")
    shape = f"`.{attribute}`" if member else f"the name `{attribute}`"
    print(f"  {len(sites)} syntactic candidate(s) for {shape}; resolving with ty...\n")

    server = Server(ROOT)
    used, other = [], []
    seen_files: set[pathlib.Path] = set()
    cold: list[float] = []
    warm: list[float] = []
    try:
        for site_path, line, column, text in sites:
            resolved_path = site_path.resolve()
            first = resolved_path not in seen_files
            seen_files.add(resolved_path)
            started = time.perf_counter()
            resolved = server.definition(resolved_path, line, column)
            (cold if first else warm).append((time.perf_counter() - started) * 1000)
            want = class_target if (site_path, line, column) in by_class else target
            (used if resolved == want else other).append((site_path, line + 1, text, resolved))
        column = _column_of(pathlib.Path(path_text), target[1], attribute)
        referenced = (
            server.references(pathlib.Path(path_text).resolve(), target[1] - 1, column)
            if args.references
            else set()
        )
        calls = (
            server.callers(pathlib.Path(path_text).resolve(), target[1] - 1, column)
            if args.callers
            else []
        )
    finally:
        server.close()

    if args.timing:
        _report_timing(cold, warm)

    imports = {
        (str(site.path), site.line)
        for site in _name_index(roots).get(attribute, [])
        if site.kind == "alias"
    }
    _report(args, used, other, referenced, calls, imports, (target[0], target[1]))
    return 0


def _report(
    args: argparse.Namespace,
    used: list[tuple[pathlib.Path, int, str, tuple[str, int] | None]],
    other: list[tuple[pathlib.Path, int, str, tuple[str, int] | None]],
    referenced: set[tuple[str, int]],
    calls: list[tuple[str, str, int]],
    imports: set[tuple[str, int]],
    home: tuple[str, int],
) -> None:
    """**`USES` counts uses**, which took a reviewer to say out loud.

    A bare-name sweep matches the definition's own line and every `import` of it, and both were
    landing in the same total as the call sites, so `kind_of` read "7 uses" against four calls,
    while a member sweep's number was calls only. The same heading over two different quantities is
    the defect this file refuses a "dead" column for; the definition drops out and the imports get
    their own line."""
    sites = [
        entry
        for entry in used
        if (str(entry[0].resolve()), entry[1]) != home and (str(entry[0]), entry[1]) not in imports
    ]
    print(f"USES ({len(sites)}):")
    for site_path, line, text, _ in sites:
        print(f"   {site_path}:{line}  {text[:90]}")
    if imported := [entry for entry in used if (str(entry[0]), entry[1]) in imports]:
        print(f"\nIMPORTED ({len(imported)}), which is reach and not use:")
        for site_path, line, text, _ in imported:
            print(f"   {site_path}:{line}  {text[:90]}")
    if args.references:
        _report_references(referenced, {(str(p), line) for p, line, _t, _r in used})
    if args.callers:
        print(f"\nCALLERS ({len(calls)}), by function:")
        for name, where, line in calls:
            print(f"   {name:32} {where}:{line}")
    if not args.all:
        return
    print(f"\nRESOLVED ELSEWHERE ({len(other)}):")
    for site_path, line, text, resolved in other:
        where = (
            "UNRESOLVED" if resolved is None else f"{pathlib.Path(resolved[0]).name}:{resolved[1]}"
        )
        print(f"   {site_path}:{line}  -> {where}   {text[:60]}")


if __name__ == "__main__":
    sys.exit(main())
