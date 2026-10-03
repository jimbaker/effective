"""Keys written down in f-strings and in prose, checked against the grammar they have to obey.

**Two stages, and the first is what keeps it honest: FILTER structurally, then VALIDATE.** A key
is written down in three media, and each is located by the thing that can actually see it —

| medium | located by | the finding |
|---|---|---|
| Python **code** | ast-grep over `string` nodes | an f-string COMPOSING a key |
| Python **prose** | ast-grep over `comment` + docstring nodes | a spelling the grammar refuses |
| Markdown | text + Mermaid node labels | a spelling the grammar refuses |

The first version of this ran a token regex over raw lines and reported 536 hits, most of them
text that merely looked like a key. A regex cannot tell a call from a comment from a sentence; an
AST can, so the regex now runs only inside prose that an AST has already handed over — the same
argument `lint._compose_key_templates` makes one module over.

**Scoped by production ownership, not by a denylist.** A candidate counts only when its leading
tag appears in `build/key-registry.json`, so it is making a claim about a namespace the substrate
mints. Then `grammar.parse` — the production reader — says whether it is still in the language.
Nobody enumerates what went out of it.

**The two findings differ, and the difference is the point.**

- `compose` — an **f-string that composes a key**. Always a finding, whether or not it parses: an
  f-string is eager format-then-concat, so it puts the flatten where the *decision* belongs, and
  the composer's guarantees (delimiter-free by type, one tag one shape, the registry that makes a
  key decodable) are all bypassed. This is the Bobby Tables position on the identity axis.
- `prose` — a key **spelled in a docstring, comment or document** that the grammar refuses. Drift:
  documentation describing a spelling no composer can mint.

**A REPORT by default and a RATCHET under `--gate`**, because a refused spelling can be
deliberate — a rejected form quoted as a counterexample (`a:b:c` in a docstring saying what the
grammar refuses) is reported here and is exactly right where it sits. The sweep supplies
candidates; a reader supplies the judgement, and `scripts/key-sweep-baseline.txt` is where that
judgement is written down. The gate fails on a finding the baseline does not carry AND on an entry
nothing produces, so the record can only shrink; an entry still reading `unreviewed` is a finding
nobody has judged yet, not an accepted one.

**Two cases that READ like they belong on that list never reach it**, and the difference between
a domain and a guess is saying which. Another runtime's identity short-circuits on `FOREIGN`
before the parser sees it, so `$awaitEvent:…` is not a finding rather than a tolerated one. And a
MIME subtype that reads like a tag **parses**: `artifact:message/ledger,sha256-9f2c` is a legal
key, because `/` is path structure and `ledger` in that position is an atom, not a frame. This
paragraph claimed the opposite of both until a reproduction was run against it, which is the
defect this whole sweep exists to find, one level up.

**WHAT IT CANNOT SEE, said first because a scanner's domain is its grammar.** Key-shaped **data** —
a literal in an assertion, a dict key, a fixture table — is invisible to this: that is
`lint.check_key_literals`, and it reads `tests/`, so key-shaped data in `src/` is scanned by
nothing at all. **History in a CURRENT spelling** is invisible too, and unavoidably so — no parser
tells "this narrates the past" apart from "this states the rule". So a green sweep means the
spellings are in the language, never that the prose is true or that the history is gone.

**A layer can be exempt BY RULE rather than by omission from a root list** (`EXEMPT_DIRS`). A
directory of dated reports is where history is deliberately relocated TO, so sweeping it turns a
correct relocation into a fresh finding.

    uv run python scripts/key_sweep.py [--compose|--prose] [paths...]

Run `just key-registry` first if the oracle is stale — this reads it and never rebuilds it, so a
stale registry shows up as a wrong answer rather than hiding behind a silent regeneration.
"""

import argparse
import json
import pathlib
import re
import sys

from ast_grep_py import SgRoot

from effective.keys.grammar import KeySyntaxError, parse

REGISTRY = pathlib.Path("build/key-registry.json")

TICKED = re.compile(r"``([^`\n]+)``|`([^`\n]+)`")
MERMAID_LABEL = re.compile(r'\["([^"\n]+)"\]|\[([^\]\n]+)\]')
# Mermaid's numeric escape (`#59;` for `;`): a diagram shows the character it names, so a key
# written with one inside a fence is read as the key the reader sees.
MERMAID_ENTITY = re.compile(r"#(\d+);")
BARE = re.compile(
    r"(?<![\w/.-])(\$?[A-Za-z][A-Za-z0-9_-]*[:;](?:\{[^{}\n]*\}|[A-Za-z0-9_,;:/@%.#*-])+)"
)
"""A key with no backticks — a module docstring's `fork:r-fork;review:m1 -> absurd.py:153`, or a
trailing `# same tool as gather:0,0`. Safe here and nowhere else: this only ever runs over text an
AST has already identified as a docstring or a comment.

**A `{…}` group is matched WHOLE, which is what stops the tail truncating a template.** The
charset outside braces deliberately excludes `(`, `)` and `=`, so a flat scan cut
`t"artifact:{Segment(kind)}/…"` off at `artifact:{Segment` and reported a token nobody wrote —
16 of them. Widening the charset instead would have loosened the match in prose; admitting a
balanced group does not, and it upgrades the class rather than hiding it: the FULL template now
reaches `refused()`, where the holes normalize to atoms and a genuinely wrong template is still
a finding.

**The leading `$` is captured too, or the `FOREIGN` guard never fires.** `refused()` short-circuits
on a `$`-prefixed token; a capture starting at a letter would hand it `awaitEvent:…` for
`$awaitEvent:…`, the guard would see nothing foreign, and another runtime's identity would be
reported as OUR drift."""

# `{g}`, `<key>`, `…`, `*` are NOTATION for "an atom goes here". Normalizing them to one atom keeps
# the sweep about DELIMITERS rather than about how somebody wrote a placeholder.
PLACEHOLDER = re.compile(r"\{[^{}\n]*\}|<[^<>\n]*>|\.\.\.|…|\*")
FOREIGN = "$"
"""A leading `$` marks another runtime's identity — Absurd writes `$awaitEvent:{name}`. Our
grammar admits it as a foreign term and has no opinion on the payload."""

EXEMPT_DIRS = ("reports",)
EXEMPT_FILES: tuple[str, ...] = ()
"""The layer the no-history rule allows, skipped wherever a root reaches them.

Named rather than merely absent from the default roots, because the difference shows: relocating
a history passage INTO the sanctioned destination would raise the sweep's count, so the tool would
penalise the correct move. A dated report is dated by contract; that is not drift.

`EXEMPT_FILES` is an empty tuple because the *mechanism* (exempt a named file, not only a
directory) is the part worth keeping."""


def registry() -> dict[str, list[str]]:
    """`{tag: [registered template, …]}` — the namespaces production actually mints."""
    if not REGISTRY.exists():
        raise SystemExit(f"{REGISTRY} missing — run `just key-registry` first.")
    shapes = json.loads(REGISTRY.read_text())["shapes"]
    return {tag: [v["template"] for v in e["variants"]] for tag, e in shapes.items()}


def leading_tag(token: str) -> str:
    return re.split(r"[:;]", token.lstrip(FOREIGN), maxsplit=1)[0]


def claimed_tag(token: str, tags: dict[str, list[str]]) -> str | None:
    """The first tag in `token` that production owns — at ANY term, not only the leading one.

    Checking the head alone under-reports, and the miss is the interesting kind: a key whose
    leading term is a CONSUMER's scope hides the substrate's tags behind it. A key leading with
    `q`, which no registry knows, can still carry two drifted namespaces after it.
    Found by the damage detector rather than by the sweep, which is the sweep reporting its own
    blind spot."""
    for term in re.split(r"[;]", token.lstrip(FOREIGN)):
        head = re.split(r"[:,]", term, maxsplit=1)[0]
        if head in tags:
            return head
    return None


def repair(token: str, templates: list[str]) -> str | None:
    """The token re-spelled with the REGISTERED template's separators — or `None`.

    Derived, never guessed. The token's atoms are read off it by splitting on any separator, then
    poured into a registered template left to right, with the LAST hole absorbing whatever remains
    (a splice takes the rest of the key — `hyp:{};{}`'s second hole is a whole nested key). The
    result is emitted only if `grammar.parse` accepts it, so a suggestion is a parse rather than a
    rewrite somebody hoped for.

    That is what makes a repair safe where a regex is not: it cannot invent a spelling the grammar
    refuses, and it stays SILENT when the atom count does not fit — which is exactly the ambiguous
    case a mechanical pass gets wrong."""
    parts = re.split(r"[:;,]", token)
    atoms = [p for p in parts[1:] if p]
    for template in templates:
        holes = template.count("{}")
        if not holes or len(atoms) < holes:
            continue
        fill = [*atoms[: holes - 1], _rejoin(token, atoms[holes - 1 :])]
        out = template
        for value in fill:
            out = out.replace("{}", value, 1)
        try:
            parse(out)
        except KeySyntaxError:
            continue
        if out != token:
            return out
    return None


def _rejoin(token: str, atoms: list[str]) -> str:
    """The tail atoms, with the separators they carried in the ORIGINAL token between them."""
    separators = re.findall(r"[:;,]", token)
    if len(atoms) < 2:
        return atoms[0] if atoms else ""
    tail = separators[-(len(atoms) - 1) :]
    out = atoms[0]
    for separator, atom in zip(tail, atoms[1:], strict=False):
        out += separator + atom
    return out


def refused(token: str) -> bool:
    """Does the grammar refuse this token? `False` for anything it has no opinion about.

    **It does NOT normalize a doubled separator, and that is the point.** This collapsed
    `[:;,/]{2,}` to a single `:` before parsing, so `gather:0,3;;ledger` parsed clean — and a
    doubled separator is the SIGNATURE of a half-applied mechanical rewrite, which is the one
    defect this instrument exists to find. An instrument that repairs its own evidence reports
    green on the damage it was built for."""
    probe = PLACEHOLDER.sub("x", token).strip().rstrip(":;,/").strip()
    if not probe or probe.startswith(FOREIGN):
        return False
    try:
        parse(probe)
        return False
    except KeySyntaxError:
        return True


def mangled(token: str) -> bool:
    """A doubled separator — the fingerprint of a rewrite that ran and did not finish.

    Reported apart from an ordinary refusal because the FIX differs: a retired spelling was
    written in an older grammar and wants re-spelling, while this one was written correctly and
    then damaged, so the question is which pass damaged it and what else that pass touched."""
    return re.search(r"[:;,/]{2,}", PLACEHOLDER.sub("x", token).strip().rstrip(":;,/")) is not None


# --- stage 1a: code — an f-string that composes a key ------------------------------------------


def composing_fstrings(source: str, tags: dict[str, list[str]]):
    """Every f-string CLAIMING a namespace production owns — an attempted compose.

    An AST node, not a line match, so a `compose_key(t"…")` beside it and a sentence about one in
    the docstring above it are both invisible to this rule. The `t"…"` prefix is what separates
    the sanctioned composer from a hand-rolled flatten, and it is one character of node text.

    **`claimed_tag`, not the leading one** — the same widening the prose rule needed, for the same
    reason: `f"q;ledger:{x}"` leads with a consumer's scope no registry knows and composes a
    substrate namespace behind it. Reading the head alone missed exactly the shape this rule
    exists to catch."""
    for node in SgRoot(source, "python").root().find_all(kind="string"):
        text = node.text()
        prefix = text[: len(text) - len(text.lstrip("rRbBuUfFtT"))].lower()
        if "f" not in prefix or "{" not in text:
            continue
        body = text[len(prefix) :].strip("\"'")
        tag = claimed_tag(body, tags)
        if tag is not None and not _is_not_a_key(body):
            yield node.range().start.line + 1, body, tag


def _is_not_a_key(body: str) -> bool:
    """Two structural exclusions, both borrowed from `lint.check_key_literals`.

    **A SPACE means prose.** No atom may contain one, so `f"respawn: the step returned Again at
    generation {n}"` is an error message that happens to open with a registered tag. Three of the
    five `src/` candidates were this.

    **A trailing SEPARATOR means a PREFIX, not a key.** `f"gather:{g},{i};"` is a frame fragment,
    and building one with an f-string is the RENDER-BACKEND position the layering rule sanctions —
    the decisions are already made, and an f-string is the fastest way to concatenate. What the
    rule is for is a flatten standing where a decision belongs."""
    return " " in body or body.rstrip().endswith((":", ";", ",", "/"))


# --- stage 1b: prose — docstrings, comments, markdown ------------------------------------------


def python_prose(source: str):
    """(line, text) for every comment and docstring — the SAME reader that finds the f-strings.

    One structural filter, not two: a `comment` node and a bare `string` standing alone as an
    expression are both things ast-grep can name, so the bare-token regex downstream only ever
    sees text an AST has already identified as prose."""
    root = SgRoot(source, "python").root()
    for node in root.find_all(kind="comment"):
        yield node.range().start.line + 1, node.text()
    for node in root.find_all(kind="expression_statement"):
        inner = node.child(0)
        if inner is None or inner.kind() != "string":
            continue
        text = inner.text()
        if text[:1] in "fFtTbB" and text[1:2] in "\"'":
            continue  # an f/t-string standing alone is not a docstring
        start = inner.range().start.line + 1
        for offset, line in enumerate(text.splitlines()):
            yield start + offset, line


def markdown_prose(source: str):
    """(line, text) for markdown, with Mermaid node labels unpacked into their own tokens."""
    in_mermaid = False
    for number, line in enumerate(source.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_mermaid = "mermaid" in line
            continue
        if in_mermaid:
            line = MERMAID_ENTITY.sub(lambda m: chr(int(m.group(1))), line)
        yield number, line
        if in_mermaid:
            for match in MERMAID_LABEL.finditer(line):
                label = (match.group(1) or match.group(2) or "").replace("<br/>", " ")
                yield number, label.replace("|", " ")


def spellings(text: str):
    """Every key-looking token in one line of prose, backticked first.

    A bare match that is a PREFIX of a backticked one on the same line is the same token seen
    twice — `artifact:app/json:9f2…` ticked, `artifact:app/json:9f2` bare, because the ellipsis
    is not in the bare charset. Reporting both doubles the count and neither row is new."""
    ticked = [(m.group(1) or m.group(2)).strip() for m in TICKED.finditer(text)]
    yield from ticked
    for match in BARE.finditer(text):
        token = match.group(1).rstrip(".,")
        if not any(other.startswith(token) for other in ticked):
            yield token


# --- stage 2: validate -------------------------------------------------------------------------


def is_exempt(path: pathlib.Path) -> bool:
    """Is this path inside a layer the no-history rule allows?

    A DIRECTORY match and a path-tail match are two different questions, so they are two
    constants rather than one clever pattern. `parts` is what makes the first exact: a file named
    `reports.md` is not the `reports` directory."""
    posix = path.as_posix()
    return any(directory in path.parts for directory in EXEMPT_DIRS) or any(
        posix.endswith(name) for name in EXEMPT_FILES
    )


def sweep(roots, tags, want):
    found: dict[str, list[tuple[str, int, str, str]]] = {}
    for root in roots:
        base = pathlib.Path(root)
        # A FILE argument is the common way to check one thing; `rglob` on one yields nothing,
        # so a naive walk answers "clean" for any path a reader passes directly.
        walk = [base] if base.is_file() else sorted(base.rglob("*"))
        for path in walk:
            if path.suffix not in (".py", ".md") or not path.is_file() or is_exempt(path):
                continue
            source = path.read_text()
            rows: list[tuple[str, int, str, str]] = []
            if path.suffix == ".py" and "compose" in want:
                rows += [("compose", n, t, tag) for n, t, tag in composing_fstrings(source, tags)]
            if "prose" in want:
                lines = python_prose(source) if path.suffix == ".py" else markdown_prose(source)
                for number, text in lines:
                    for token in spellings(text):
                        tag = claimed_tag(token, tags)
                        if tag is not None and refused(token):
                            rows.append(("prose", number, token, tag))
            if rows:
                found[str(path)] = sorted(set(rows), key=lambda r: (r[1], r[2]))
    return found


# --- the baseline, and the stale rule that makes it a ratchet ----------------------------------

BASELINE = pathlib.Path("scripts/key-sweep-baseline.txt")

type Entry = tuple[str, str, str]
"""`(path, rule, token)` — deliberately NOT the line number.

A baseline keyed on a line is invalidated by every edit above it, which turns the record into
churn and teaches a reader to regenerate rather than to fix. Keyed on identity, a finding that
MOVED is the same finding and a finding that was FIXED is the only thing that disappears."""


def load_baseline() -> dict[Entry, str]:
    if not BASELINE.exists():
        return {}
    out: dict[Entry, str] = {}
    for line in BASELINE.read_text().splitlines():
        if not (line := line.strip()) or line.startswith("#"):
            continue
        path, rule, token, reason = (part.strip() for part in line.split(" :: ", 3))
        out[(path, rule, token)] = reason
    return out


def write_baseline(entries: dict[Entry, str]) -> None:
    header = [
        "# key-sweep baseline — findings recorded as accepted, so the gate can fail on GROWTH.",
        "#",
        "# Format:  <path> :: <rule> :: <token> :: <reason>",
        "#",
        "# Keyed on (path, rule, token) and NOT on a line number, so an edit above a finding",
        "# does not invalidate its entry. The gate fails on any finding NOT listed here, AND",
        "# on any entry here matching no current finding (stale — delete it), so the record",
        "# can only shrink. Regenerate with `--init-baseline` only when bootstrapping.",
        "#",
        "# A reason is what makes an entry reviewable. `--init-baseline` writes a placeholder;",
        "# replacing it with the actual judgement is the work, and an unreplaced one is a",
        "# finding nobody has read yet.",
        "",
    ]
    body = [
        f"{p} :: {rule} :: {tok} :: {reason}" for (p, rule, tok), reason in sorted(entries.items())
    ]
    BASELINE.write_text("\n".join([*header, *body, ""]))


def live_entries(found: dict[str, list[tuple[str, int, str, str]]]) -> dict[Entry, str]:
    return {(path, rule, tok): "" for path, rows in found.items() for rule, _n, tok, _t in rows}


def gate(found: dict[str, list[tuple[str, int, str, str]]]) -> int:
    """Fail on a finding the baseline does not carry, and on a baseline entry nothing produces."""
    baseline, live = load_baseline(), live_entries(found)
    new = sorted(set(live) - set(baseline))
    stale = sorted(set(baseline) - set(live))
    for path, rule, token in new:
        print(f"{path}: [{rule}] {token} — NOT in the baseline")
    for path, rule, token in stale:
        print(f"{path}: [{rule}] {token} — STALE baseline entry, delete it")
    if new or stale:
        print(f"\n{len(new)} new, {len(stale)} stale — the baseline is {BASELINE}")
        return 1
    print(f"key-sweep: {len(baseline)} baselined, 0 new, 0 stale")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose", action="store_true", help="only the f-string rule")
    parser.add_argument("--prose", action="store_true", help="only the spelling rule")
    parser.add_argument("--gate", action="store_true", help="fail on growth or a stale entry")
    parser.add_argument("--init-baseline", action="store_true", help="bootstrap the baseline")
    parser.add_argument("paths", nargs="*")
    args = parser.parse_args(argv)
    want = {"compose"} if args.compose else {"prose"} if args.prose else {"compose", "prose"}

    tags = registry()
    roots = args.paths or ["src", "docs", "tests", "scripts", "examples", ".claude"]
    found = sweep(roots, tags, want)

    if args.init_baseline:
        entries = {
            k: "unreviewed — bootstrapped, replace with the judgement" for k in live_entries(found)
        }
        write_baseline(entries)
        print(f"wrote {len(entries)} entries to {BASELINE}")
        return 0
    if args.gate:
        return gate(found)

    counts = {rule: 0 for rule in want}
    damaged = 0
    for rows in found.values():
        for rule, _number, token, _tag in rows:
            counts[rule] += 1
            damaged += rule == "prose" and mangled(token)
    summary = ", ".join(f"{n} {rule}" for rule, n in sorted(counts.items()))
    print(f"{summary} — across {len(found)} files, {damaged} of them MANGLED")
    print(f"(oracle: {REGISTRY}, {len(tags)} namespaces; roots: {' '.join(roots)})")
    print(
        f"(exempt: {' '.join((*EXEMPT_DIRS, *EXEMPT_FILES))} — the layers history is allowed in)"
    )
    print("(blind to: key-shaped DATA, and history spelled in the CURRENT grammar)\n")
    for path, rows in sorted(found.items(), key=lambda kv: -len(kv[1])):
        print(f"{path}  ({len(rows)})")
        for rule, number, token, tag in rows:
            label = "prose/MANGLED" if rule == "prose" and mangled(token) else rule
            print(f"    :{number:<5} [{label}] {token}")
            fix = repair(token, tags[tag]) if rule == "prose" else None
            print(f"    {'':<6} mints: {' | '.join(tags[tag])}")
            if fix:
                print(f"    {'':<6}    -> {fix}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
