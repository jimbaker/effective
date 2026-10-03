"""Preflight: the gate's own toolchain, checked before the gate believes itself.

`just check` reports on the toolchain it happened to run on, and says so nowhere. A toolchain can
move under a green gate with nothing committed: `uv` repoints its `cpython-3.14` symlink to a new
patch release, or an unpinned `uvx ty` resolves a newly shipped release at invocation, and the
same commit and the same `uv.lock` go from green to red.

This runs first, and refuses. `check: preflight lint …` takes seconds, needs no infra, and makes
no network call past one `--version` per unlocked tool.

**THE DOMAIN IS DERIVED, NOT ENUMERATED**, which is the whole reason this is a script and not
three asserts. A hand-written list of "the three toolchains" is the failure mode this repo keeps
re-learning: a domain spelled as a list of the things you remembered will always be one entry
short, because the referent grows new entries and nothing links them. So the rule is stated
structurally instead:

    `uv.lock` governs everything the gate reaches through `uv run`.
    Anything else the gate shells out to is UNGOVERNED and must pin its own version at the call.

and the check reads the justfile, walks `check`'s recipe closure, and classifies every command
line in it. Today exactly one line is ungoverned — `uvx ty@0.0.73 check`, spelled twice — and if
someone adds `npx`, a bare `uvx`, or a global binary to the gate tomorrow, this fails naming it
rather than going quietly one entry short.

Three checks, and each is a different KIND of claim:

1. **The interpreter is the pinned one.** `.python-version` against the running `python`. A
   version string, because that is exactly what drifted.
2. **Every ungoverned tool in the gate pins a version at its call site, the spellings agree, and
   the pin TAKES.** `uvx ty@0.0.73 --version` must answer 0.0.73 — a pin nobody resolves is a
   claim, not a pin.
3. **tdom's vendored patch is doing its job on this interpreter** — a BEHAVIOURAL probe, not a
   version string, because the version was never the question: 0.1.17 and upstream's later 0.1.18
   have byte-identical `parser.py` and both fail on 3.14.7 unpatched (`infra/tdom/PIN.txt`). The
   probe is the one-liner from that record, plus the malformed case the first attempted fix broke:

       html(t'<g class="{k}"></g>')   must render      # 3.14.7 unpatched: ValueError
       html(t'<')                     must raise       # the wrong fix rendered `&lt;` instead

   This is NOT a coverage claim — the suite covers tdom in fifteen tests. It is a DIAGNOSIS
   ACCELERATOR. That break presented as fifteen unrelated failures in graphlayout and dashboard,
   and the first diagnosis blamed the wrong subsystem entirely. One named line here is cheaper
   than that hour.

What this does NOT check, said plainly because a gate's grammar is its domain: anything inside
`uv.lock` (the lock is the record, and `uv run` enforces it); the `infra/*/PIN.txt` artifact
hashes (those gate a FETCH, and their own recipes verify them at fetch time — re-verifying ten
records on every `just check` would buy latency and no signal); and any toolchain a recipe
OUTSIDE `check`'s closure uses.

    just preflight
"""

import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
JUSTFILE = REPO / "justfile"
PYTHON_VERSION = REPO / ".python-version"

GATE = "check"
"""The recipe whose closure is this gate's domain. Named once: `just check` is what CI and every
session runs, so what it reaches is what has to be deterministic."""

_RECIPE = re.compile(r"^(?P<name>[a-z][a-z0-9-]*)(?P<params>[^:=\n]*):(?!=)(?P<deps>.*)$")
"""A justfile recipe header at column 0. `(?!=)` keeps `name := value` assignments out."""

_LOCKED = re.compile(r"(^|\s|;|&&|\|\|)uv\s+run\s")
"""Governed by `uv.lock`. Env-var prefixes (`FOO=bar uv run …`) and `&&` chains still match."""

_PINNED_UVX = re.compile(r"\buvx\s+(?P<tool>[A-Za-z0-9._-]+)@(?P<version>[A-Za-z0-9._-]+)\b")
"""An ungoverned tool that pins its version AT THE CALL. The justfile line is the record — there
is deliberately no second file restating it, because two bookkeepers for one version is the
defect this script exists to catch one level up."""

_BARE_TOOL = re.compile(r"^\s*(?P<tool>uvx|npx|pipx|npm|node|java|cargo|go)\b")
"""Ungoverned runners that reach a registry. Listed as a floor, not as the domain — anything not
matching `_LOCKED` is reported whether or not it is here; these just get a sharper message."""


@dataclass(frozen=True, slots=True)
class Recipe:
    name: str
    deps: tuple[str, ...]
    body: tuple[str, ...] = field(default_factory=tuple)


def parse_recipes(text: str) -> dict[str, Recipe]:
    """The justfile's recipes, by name. Text-parsed rather than asked of `just`, because the
    question is what the file SAYS the gate runs, and a shell-out would need `just` installed to
    tell us whether the toolchain is right."""
    recipes: dict[str, Recipe] = {}
    current: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        if line[:1] in {" ", "\t"} and current is not None:
            body.append(line)
            continue
        if current is not None:
            recipes[current] = Recipe(current, recipes[current].deps, tuple(body))
        current, body = None, []
        if (match := _RECIPE.match(line)) and "#" not in line[: match.start("name")]:
            current = match["name"]
            recipes[current] = Recipe(current, tuple(match["deps"].split()))
    if current is not None:
        recipes[current] = Recipe(current, recipes[current].deps, tuple(body))
    return recipes


def closure(recipes: dict[str, Recipe], root: str) -> list[str]:
    """`root` and everything it depends on, in visit order. A recipe naming a dependency that
    does not exist is left out rather than raising: this script's job is the toolchain, and a
    broken justfile fails at `just` in the next breath with a better message."""
    seen, order, stack = set(), [], [root]
    while stack:
        name = stack.pop(0)
        if name in seen or name not in recipes:
            continue
        seen.add(name)
        order.append(name)
        stack.extend(recipes[name].deps)
    return order


def command_lines(recipes: dict[str, Recipe], names: list[str]) -> list[tuple[str, str]]:
    """(recipe, command) for every real command in the closure — comments and blanks dropped."""
    out: list[tuple[str, str]] = []
    for name in names:
        for raw in recipes[name].body:
            command = raw.strip()
            if not command or command.startswith("#"):
                continue
            out.append((name, command))
    return out


def check_interpreter() -> list[str]:
    """The pinned interpreter is the running one. THE one that drifted."""
    if not PYTHON_VERSION.exists():
        return [".python-version is missing — the interpreter is unpinned and `uv` will float it"]
    pinned = PYTHON_VERSION.read_text().strip()
    running = platform.python_version()
    if running != pinned:
        return [
            f"interpreter drift: .python-version pins {pinned}, this process is {running}. "
            f"A gate is green on the interpreter it ran on — re-run after `uv sync`."
        ]
    return []


def check_lock_is_current() -> list[str]:
    """`uv.lock` still satisfies `pyproject.toml`.

    `uv run` re-locks in silence when the two disagree, so a dependency edit can first
    take effect in CI, on a resolution nobody reviewed — the same shape as an ungoverned
    tool one layer down. `npm ci` is the discipline CLAUDE.md's supply-chain policy names
    for the npm side; this is its Python equivalent, and the repo had none.
    """
    out = subprocess.run(["uv", "lock", "--check"], cwd=REPO, capture_output=True, text=True)
    if out.returncode == 0:
        return []
    return [
        "`uv.lock` does not satisfy `pyproject.toml` — run `uv lock` and review the "
        "resolution it produces, rather than letting `uv run` pick one silently"
    ]


def ungoverned_tools(recipes: dict[str, Recipe]) -> tuple[list[str], set[str]]:
    """DOMAIN ONE — what the gate reaches that `uv.lock` does not govern, and must pin at its call.

    Returns (problems, tool names). A command that is neither `uv run` nor a version-pinned `uvx`
    is the finding: `uv.lock` does not govern it, so the gate can change verdict with no commit.
    """
    problems: list[str] = []
    used: set[str] = set()
    for recipe, command in command_lines(recipes, closure(recipes, GATE)):
        if _LOCKED.search(command):
            continue  # uv.lock is the record
        if match := _PINNED_UVX.search(command):
            used.add(match["tool"])
            continue
        tool = bare["tool"] if (bare := _BARE_TOOL.match(command)) else command.split()[0]
        problems.append(
            f"`{recipe}` runs {tool!r} outside `uv run` and pins no version — "
            f"`uv.lock` does not govern it, so the gate can change verdict with no commit: "
            f"{command}"
        )
    return problems, used


def pin_sites(recipes: dict[str, Recipe], tools: set[str]) -> dict[str, list[tuple[str, str]]]:
    """DOMAIN TWO — every site in the WHOLE justfile that pins one of `tools`, version and recipe.

    Deliberately wider than `ungoverned_tools`' closure, and the two domains are different
    questions. The closure decides what must be pinned; agreement is a property of the file,
    because a standalone recipe outside the gate can still be the one a person runs by hand.
    `typecheck` is exactly that, and the justfile's own comment already demands the two spellings
    match. Measured while mutating this script: scoping agreement to the closure let `lint` say
    0.0.72 and `typecheck` say 0.0.73 with a green preflight and no complaint."""
    sites: dict[str, list[tuple[str, str]]] = {}
    for name, recipe in recipes.items():
        for raw in recipe.body:
            command = raw.strip()
            if command.startswith("#") or not (match := _PINNED_UVX.search(command)):
                continue
            if match["tool"] in tools:
                sites.setdefault(match["tool"], []).append((match["version"], name))
    return sites


def check_pins(sites: dict[str, list[tuple[str, str]]]) -> tuple[list[str], list[tuple[str, str]]]:
    """Each tool is pinned at ONE version across the file, and that pin resolves to itself.

    Returns (problems, what was verified) so the caller can print the domain — a gate that prints
    only failures cannot be told from one whose domain came out empty."""
    problems: list[str] = []
    resolved: list[tuple[str, str]] = []
    for tool, found in sorted(sites.items()):
        versions = {version for version, _ in found}
        if len(versions) > 1:
            spellings = ", ".join(f"{v} (in `{r}`)" for v, r in sorted(found))
            problems.append(
                f"{tool!r} is pinned at two versions — {spellings}. The standalone recipe and "
                f"the gate would disagree about what clean means."
            )
            continue
        version = versions.pop()
        if (actual := resolve(tool, version)) is None:
            problems.append(f"could not resolve {tool}@{version} to ask its version")
        elif not re.search(rf"(^|\s|v){re.escape(version)}(\s|$)", actual):
            problems.append(
                f"{tool} pinned at {version} but `{tool}@{version} --version` answers "
                f"{actual!r} — the pin does not take, so it is a claim rather than a pin"
            )
        else:
            resolved.append((tool, version))
    return problems, resolved


def check_gate_toolchain() -> tuple[list[str], list[tuple[str, str]]]:
    """The two domains above, composed: what must be pinned, then whether the pins hold."""
    recipes = parse_recipes(JUSTFILE.read_text())
    if GATE not in recipes:
        return [f"justfile has no `{GATE}` recipe — this script's domain is derived from it"], []
    problems, used = ungoverned_tools(recipes)
    pin_problems, resolved = check_pins(pin_sites(recipes, used))
    return problems + pin_problems, resolved


def resolve(tool: str, version: str) -> str | None:
    """What the pinned tool says it is, from a SUCCESSFUL run. `None` when it will not run.

    The exit code is checked, and stderr is deliberately not read as an answer. Measured while
    mutating this script: `uvx ty@0.0.999999 --version` fails with *"Distribution not found …
    ty==0.0.999999"* — an error message that QUOTES the version it could not find, so a
    substring test against combined output passes on the one input the test exists to catch."""
    try:
        proc = subprocess.run(
            ["uvx", f"{tool}@{version}", "--version"],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except OSError, subprocess.SubprocessError:
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


def check_tdom_patch() -> list[str]:
    """The vendored patch is doing its job ON THIS INTERPRETER — behaviour, not a version.

    Both halves matter and only one of them is the bug: the well-formed case is what 3.14.7's
    batching `HTMLParser.feed` broke, and the malformed case is what the first attempted fix
    broke while making the well-formed one pass. A probe carrying only the thing you are fixing
    cannot see what you broke to fix it."""
    try:
        from tdom import html
    except ImportError as missing:  # pragma: no cover -- the dependency is locked
        return [f"tdom does not import: {missing}"]

    problems: list[str] = []
    attribute = "x"
    try:
        rendered = html(t'<g class="{attribute}"></g>')
    except Exception as broke:
        return [
            f"tdom fails on a WELL-FORMED template ({type(broke).__name__}: {broke}). This is "
            f"the 3.14.7 batching-feed break; the vendored patch in infra/tdom is not in play. "
            f"It presents as ~15 unrelated failures in graphlayout/dashboard — see "
            f"infra/tdom/PIN.txt before diagnosing anything downstream."
        ]
    if 'class="x"' not in str(rendered):
        problems.append(f"tdom rendered a well-formed template as {rendered!r}")

    try:
        html(t"<")
        problems.append(
            "tdom accepted a MALFORMED template instead of raising — the parse is being flushed "
            "as text, which is what moving `super().close()` (rather than draining) does. The "
            "well-formed half passes under that fix too; see infra/tdom/PIN.txt."
        )
    except ValueError:
        pass
    return problems


def main(argv: list[str]) -> int:
    if len(argv) > 1:
        print(__doc__)
        return 0

    problems = check_interpreter() + check_lock_is_current()
    toolchain, pins = check_gate_toolchain()
    problems += toolchain + check_tdom_patch()

    for problem in problems:
        print(f"preflight: {problem}", file=sys.stderr)
    if problems:
        print(
            f"\npreflight: {len(problems)} problem(s) — the gate would report on a "
            f"toolchain nobody pinned",
            file=sys.stderr,
        )
        return 1

    pinned = ", ".join(f"{tool}@{version}" for tool, version in pins) or "none"
    print(
        f"preflight: python {platform.python_version()} (pinned), "
        f"ungoverned tools in `{GATE}`: {pinned}, tdom patch live"
    )
    report_preconditions()
    return 0


def report_preconditions() -> None:
    """What the gate can and cannot reach, said out loud before it runs — and NEVER fatal.

    A toolchain problem is a lie about what the gate measured, so `main` refuses on one. A missing
    precondition is not a lie; it is a smaller domain, and the honest response is to name the
    domain and continue. A sandbox with no egress and a laptop with the database
    down are both good reasons, and refusing to run 4000 tests over either teaches nothing.

    So this prints and returns, and a green `just check` cannot imply coverage it did not have:
    conformance tests skipping silently, with the whole Absurd half absent from a run that looked
    complete.
    """
    from preconditions import DISCOVERED_AT_USE, probe_all

    have = [p for p in probe_all() if p.available]
    lack = [p for p in probe_all() if not p.available]
    if have:
        print(f"preflight: available — {', '.join(f'{p.name} ({p.detail})' for p in have)}")
    for p in probe_all():
        if p.warning:
            print(f"preflight: WARNING {p.name} — {p.detail}")
            print(f"           {p.warning}")
    for p in lack:
        print(f"preflight: MISSING {p.name} — {p.detail}")
        print(f"           unrun:  {p.covers}")
        print(f"           enable: {p.enable}")
    if lack:
        print(
            f"preflight: {len(lack)} precondition(s) absent — the gate CONTINUES over a smaller "
            f"domain, which is the point of saying so"
        )
    print(f"preflight: {DISCOVERED_AT_USE}")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
