"""Determinism-boundary lint for workflow source (ast-grep-py).

A sync generator makes the boundary a *compiler* guarantee (it cannot `await` I/O);
these rules cover what the compiler cannot see:

| rule                   | requires                                                          |
|------------------------|-------------------------------------------------------------------|
| ``require-yield-from`` | every op goes through ``yield from <wrapper>(...)``; a bare       |
|                        | ``yield`` binds a generator and runs nothing                      |
| ``no-io-import``       | no network, random or db import; I/O happens through a yielded op |
| ``no-nondeterminism``  | no wall-clock or random *call* between yields (``datetime`` may   |
|                        | be imported for its types; ``.now()`` may not be called)          |

A seam-level rule (``--deps``) enforces the dependency direction: the substrate packages never
import the code built on them.

| rule                   | requires                                                          |
|------------------------|-------------------------------------------------------------------|
| ``seam-dep-direction`` | ``effective/`` imports none of ``agent``, ``tui``, ``examples``;  |
|                        | ``agent/`` imports neither ``tui`` nor ``examples``               |

A back-edge turns extracting a package from a ``git mv`` into a refactor.

Use as a library (``check_source`` / ``check_file`` / ``check_deps_file``) or wire
into CI.
"""

import ast
import io
import os
import re
import sys
import tokenize
import tomllib
import typing
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAliasType, TypedDict, assert_never, get_args, get_origin

from ast_grep_py import SgRoot
from pydantic import TypeAdapter, ValidationError

from effective.domain import DomainOp
from effective.keys import DOMAIN_DIRECTIVE, RESERVED_AUTHORITY_TAGS
from effective.keys.frame import unmintable
from effective.keys.grammar import (
    ARITY_SEPARATOR,
    COORDINATE_NAME_SEPARATOR,
    TAG_SEPARATOR,
    TERM_SEPARATOR,
    Domain,
    Hole,
    KeySyntaxError,
    Skeleton,
    SkeletonTerm,
    Splice,
    parse_skeleton,
)
from effective.keys.marker import Index, Name, Ordinal, Run, Subject
from effective.keys.registry import KeyMap, Shape, UnknownTag, _render_skeleton, separated
from effective.ops import WorkflowOp


def config_root() -> tuple[Path, dict[str, Any]] | None:
    """The nearest `pyproject.toml` at or above the cwd with a `[tool.effective]` table, and that
    table. The search stops at the repository root, so a workspace member's own `pyproject.toml`
    does not hide the root's."""
    for directory in (Path.cwd(), *Path.cwd().parents):
        if (pyproject := directory / "pyproject.toml").is_file():
            with pyproject.open("rb") as f:
                tool = tomllib.load(f).get("tool", {})
            if "effective" in tool:
                return directory, tool["effective"]
        if (directory / ".git").exists():
            return None
    return None


class _LazyImport(TypedDict):
    path: str
    module: str
    reason: str


_SHAPES: tuple[tuple[re.Pattern[str], TypeAdapter[Any]], ...] = (
    (
        re.compile(r"_srcs$|^extra_paths|^private_ops_nouns$|_exempt$|^app_tables$"),
        TypeAdapter(list[str]),
    ),
    (re.compile(r"^seam_forbidden$"), TypeAdapter(dict[str, list[str]])),
    (re.compile(r"^not_"), TypeAdapter(dict[str, str])),
    (re.compile(r"^lazy_import_allowed$"), TypeAdapter(list[_LazyImport])),
)
"""The shape each configured name must have, so a string where a list belongs raises rather than
extending nothing."""


def configured(name: str, table: str = "lint") -> Any:
    """`[tool.effective.<table>].<name>`, from `config_root`, checked against `_SHAPES`.

    A tree that holds more than this repository ships (more workflow modules, more packages a seam
    keeps out) declares the extension there, so every tree runs the same lint."""
    found = config_root()
    value = None if found is None else found[1].get(table, {}).get(name)
    if value is None:
        return None
    for pattern, shape in _SHAPES:
        if pattern.search(name):
            try:
                shape.validate_python(value, strict=True)
            except ValidationError as wrong:
                raise TypeError(f"[tool.effective.{table}].{name} has the wrong shape") from wrong
    return value


FORBIDDEN_IMPORT_ROOTS: frozenset[str] = frozenset(
    {
        "httpx",
        "requests",
        "aiohttp",
        "urllib",
        "socket",
        "smtplib",
        "anthropic",
        "openai",
        "random",
        "secrets",
        "sqlalchemy",
        "asyncpg",
        "psycopg",
        "psycopg2",
    }
)
FORBIDDEN_CALLEES: frozenset[str] = frozenset(
    {"datetime.now", "datetime.utcnow", "time.time", "time.monotonic", "time.time_ns"}
)


# The layer authority model as a lint rule. A `@domain_layer` is
# call-scoped: it may forward `yield op` or yield a DomainOp, and it may not reach a
# control-flow `WorkflowOp`, so it cannot suspend or alter control flow. This is the
# static form of the capability confinement the two-seam split buys, keyed on the
# decorator name, which a single shared decorator could not give ast-grep.
def _members(alias: TypeAliasType) -> frozenset[str]:
    """The class names a `type` alias admits, through any union or alias it nests, so a new arm
    joins the lint with the type."""
    return _arms(alias.__value__)


def _arms(value: Any) -> frozenset[str]:
    match get_origin(value) or value:
        case typing.Union:
            return frozenset().union(*map(_arms, get_args(value)))
        case TypeAliasType() as nested:
            return _members(nested)
        case type() as op:
            return frozenset({op.__name__})
        case other:
            raise TypeError(f"an op union arm is a class, a union or an alias, not {other!r}")


WORKFLOW_OP_CTORS: frozenset[str] = _members(WorkflowOp)
DOMAIN_OP_CTORS: frozenset[str] = _members(DomainOp)

# The seam as a lint. Dependency direction is one-way: the substrate packages never import the
# code built on them, so extracting a package is a `git mv` rather than a refactor.
#
# Keys and values are both **dotted module prefixes**, matched by prefix rather than by top-level
# name, which lets one table carry the cross-package rule and the intra-package ones. A
# single-segment key covers its whole package, because `agent` prefixes `agent.bracket`.
#
# A row applies to module `M` iff its key prefixes `M` **and `M` is not itself under any entry of
# that row's forbidden set**. The protected half is named once, positively, and everything else in
# the package is covered by not being it. `effective.coding.runners` may import `gate` because
# `runners` belongs to the half `gate` is in; `effective.coding.verdicts` may not, because it does
# not. A new `effective/coding/prompts.py` is covered on the day it is written, with no edit here.
#
# The inversion is deliberate. Spelling the rule as a list of the light modules ("`specs`,
# `states`, `transition`, `verdicts` must stay importable without `gate`, `runners` or `edits`")
# leaves a domain enumerated as a list, which is always one entry short.
SEAM_FORBIDDEN: dict[str, frozenset[str]] = {
    "effective": frozenset({"agent", "tui", "examples"}),
    "agent": frozenset({"tui", "examples"}),
    # An example is built on the substrate, so it shows what an adopter can import.
    "examples": frozenset({"tui"}),
    # The trampoline is generic over the state type; the coding table is one of three
    # embodiments that ride it (`tests/test_machine_second_embodiment.py`,
    # `tests/test_machine_non_command_predicate.py`). An edge this way makes "generic"
    # unfalsifiable.
    "effective.machine": frozenset({"effective.coding"}),
    # The yielding half must stay importable without the handler-side half, which carries
    # ast-grep, jedi and a ruff config resolved from the filesystem. A re-export such as
    # `CODING_TOOLS` from `coding/specs.py` would make the yielding half import the tool half.
    "effective.coding": frozenset(
        {"effective.coding.gate", "effective.coding.runners", "effective.coding.edits"}
    ),
}
SEAM_FORBIDDEN |= {
    package: SEAM_FORBIDDEN.get(package, frozenset()) | frozenset(more)
    for package, more in (configured("seam_forbidden") or {}).items()
}

# Why each forbidden prefix is forbidden, keyed by the TARGET rather than by the (owner, target)
# pair, because that is the grain the reason actually has. A rule that reports "this is forbidden"
# without saying why teaches people to reach for the escape.
SEAM_REASONS: dict[str, str] = {
    "tui": "a UI; the substrate renders a run without one (`effective.runread`)",
    "effective.coding": "the trampoline is generic; an embodiment may not be imported by it",
    "effective.coding.gate": "the handler-side half carries ast-grep, jedi and a ruff config",
    "effective.coding.runners": "the handler-side half carries ast-grep, jedi and a ruff config",
    "effective.coding.edits": "the handler-side half carries ast-grep, jedi and a ruff config",
}
_PRIVATE_REASON = "code built on the substrate; dependencies point from it to the substrate"

# A broad `except` in an op-layer would swallow the suspend signal the durable
# handler raises through the layer's `yield` — breaking
# suspend-from-mid-layer. An op-layer must catch only the specific errors it means
# to handle (e.g. `retry` catches its transient type). Provider-neutral: we forbid
# the over-broad catch rather than naming any one engine's suspend exception.
BROAD_EXCEPTS: frozenset[str] = frozenset({"", "Exception", "BaseException"})

# The role-partitioned source sets: the single source of truth that both the justfile `lint`
# recipe and an agent's post-edit lint gate read, so the three role-dispatched rules cannot drift
# between them.
#
# The agent gate is a subset of `just lint`. `lint_post_edit` runs `check_source`,
# `check_layer_source` and `check_deps_source`; the rest of `just lint` (the key rules,
# `--sql-templates`, `--role-coverage`, ruff, `ty`) it never runs. What holds by construction is
# that these three read the same sets from here.
#
# `just lint` dispatches BY ROLE:
# `check_source` (determinism-boundary) runs ONLY on workflow-role files;
# `check_layer_source` (layer-authority) ONLY on layer-role files (the two sets are
# disjoint, curated); `check_deps_source` runs whole-tree (role-independent, already
# path-keyed via `seam_forbidden_for`). The CLI defaults its file list to these
# when none is given, so the justfile passes no hand-list.
WORKFLOW_ROLE_SRCS: tuple[str, ...] = (
    "src/effective/react.py",
    "src/effective/interrupts.py",
    "src/effective/smol.py",
    "src/effective/compose.py",
    "src/agent/debug.py",
    "src/effective/improve.py",
    "src/agent/bench_sweep.py",
    "src/agent/permit_tuning.py",
    "src/effective/code.py",
    "src/effective/machine/trampoline.py",
    "src/effective/machine/specs.py",
    # The mechanical judges. They yield no op, but they are RE-EXECUTED on replay and they pick
    # the next state, so a nondeterministic one diverges the walk.
    "src/effective/coding/verdicts.py",
    # The judgment processor: `judge` re-runs it on replay, so it must build the same op each time.
    "src/effective/judgment.py",
    "src/effective/prose/runners.py",
    # Beside `coding/verdicts.py` and for its reason: `verdict_for_verify` yields nothing and is
    # RE-EXECUTED on replay, and it picks the next state: a nondeterministic one diverges the
    # walk. `prose/runners.py` also yields ops directly (`await_event`, `call_tool`), so it is
    # workflow-role on both criteria rather than on the honorary one.
    "src/effective/skills.py",
    "src/effective/combinators.py",
    "src/effective/search.py",
    "src/effective/spawning.py",
    "src/examples/coder/machine.py",
    "src/examples/deep_research/research.py",
    "src/examples/startup/asking.py",
    "src/examples/startup/incident.py",
    "src/examples/startup/launch.py",
    "src/examples/startup/voice.py",
    "tests/_funnel.py",
    "tests/_cart.py",
    "tests/_mcts.py",
    "tests/_coding.py",
    "examples/dashboard_demo.py",
    "examples/first_workflow.py",
    "examples/smol_door.py",
    "examples/tui_demo.py",
)
WORKFLOW_ROLE_SRCS += tuple(configured("workflow_role_srcs") or ())
# COVERAGE BOUNDARY: the layer-authority lint scans only these files and keys on a
# `@op_layer`/`@domain_layer` *decorated def* at the def site. A service authored outside them
# (tests/, an adopter's tree) or as a bound-object `__call__` (the tunable-service pattern) is
# not statically scanned; it relies on `serve`'s runtime marker check and the `run_metered` shape
# guard. Extend the list, or key on the `serve` import, when an out-of-tree or bound-object
# service appears.
LAYER_ROLE_SRCS: tuple[str, ...] = (
    "src/effective/layers.py",
    "src/effective/cost.py",
    "src/effective/govern.py",
    "src/effective/permission.py",
    "src/effective/telemetry.py",
    "src/effective/tape.py",
    "examples/hooks_as_layers.py",
)
LAYER_ROLE_SRCS += tuple(configured("layer_role_srcs") or ())
"""The layer-authority gate's domain. `--layer-coverage` proves it complete, so the gate
maintains this list and a reader need not remember it."""


def _in_role_set(path: str | Path, role_set: tuple[str, ...]) -> bool:
    """Whether ``path`` names a file in ``role_set`` — matched on the repo-relative
    suffix, so an absolute path or a bare ``src/...`` both resolve."""
    p = str(path).replace("\\", "/")
    return any(p == s or p.endswith("/" + s) for s in role_set)


def is_workflow_role(path: str | Path) -> bool:
    """True iff ``check_source`` (the determinism-boundary lint) applies to ``path``
    under `just lint`'s role dispatch."""
    return _in_role_set(path, WORKFLOW_ROLE_SRCS)


def is_layer_role(path: str | Path) -> bool:
    """True iff ``check_layer_source`` (the layer-authority lint) applies to ``path``
    under `just lint`'s role dispatch."""
    return _in_role_set(path, LAYER_ROLE_SRCS)


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str
    line: int
    text: str
    filename: str = "<workflow>"

    def __str__(self) -> str:
        return f"{self.filename}:{self.line}: [{self.rule}] {self.message} — {self.text!r}"


def _import_root(text: str) -> str:
    text = text.strip()
    if text.startswith("from "):
        module = text[len("from ") :].split(" import")[0]
    elif text.startswith("import "):
        module = text[len("import ") :].split(",")[0].split(" as ")[0]
    else:
        return ""
    return module.strip().split(".")[0]


def _callee(call_text: str) -> str:
    return call_text.split("(", 1)[0].strip()


def check_source(source: str, filename: str = "<workflow>") -> list[Violation]:
    return list(_source_violations(source, filename))


_TERMINAL_STATEMENTS = ("return_statement", "raise_statement")


def _unreachable(node: Any) -> bool:
    """Is this statement dead code — does a preceding sibling in its own block return or raise?

    Narrow on purpose. It answers the one question the determinism rules actually need — *can this
    ever execute?* — and answers it structurally, from the syntax tree, rather than by trusting a
    comment or a filename. It is not a general reachability analysis: a `while True` above, an
    `if` that always holds, a `sys.exit()` are all reachable as far as this is concerned, and that
    is the right side to be conservative on.

    It exists for the **generator marker** — the `yield` after a `return` that a pure function
    carries so it satisfies an `Effect`/`Judge` type. Flagging that was a false positive with
    teeth: it is what kept `effective/coding/verdicts.py`, whose judges pick the next state on
    replay, out of the determinism gate entirely.

    One consequence, accepted rather than overlooked: a genuinely wrong `yield SomeOp()` sitting
    after a `return` also stops being reported. It is dead code, so it cannot breach the boundary
    this rule guards — the finding there is *unreachable statement*, which is a different rule's
    job and not one worth widening this one to fake."""
    stmt = node.parent()
    if stmt is None:
        return False
    return any(sibling.kind() in _TERMINAL_STATEMENTS for sibling in stmt.prev_all())


def _source_violations(source: str, filename: str) -> Iterator[Violation]:
    root = SgRoot(source, "python").root()

    for node in root.find_all(kind="yield"):
        text = node.text()
        if not text.startswith("yield from") and not _unreachable(node):
            yield Violation(
                "require-yield-from",
                "perform ops with `yield from <wrapper>(...)`, not a bare `yield`",
                node.range().start.line + 1,
                text,
                filename,
            )

    for kind in ("import_statement", "import_from_statement"):
        for node in root.find_all(kind=kind):
            root_mod = _import_root(node.text())
            if root_mod in FORBIDDEN_IMPORT_ROOTS:
                yield Violation(
                    "no-io-import",
                    f"workflows must not import {root_mod!r}; do I/O through a yielded op",
                    node.range().start.line + 1,
                    node.text(),
                    filename,
                )

    for node in root.find_all(kind="call"):
        callee = _callee(node.text())
        if callee in FORBIDDEN_CALLEES or callee.startswith("random."):
            yield Violation(
                "no-nondeterminism",
                f"{callee}(...) is nondeterministic; pass the value in or yield an op",
                node.range().start.line + 1,
                node.text(),
                filename,
            )


def _yielded_callee(yield_text: str) -> str:
    """The constructor name a `yield`/`yield from` invokes, or '' for a bare forward."""
    body = yield_text.removeprefix("yield from ").removeprefix("yield ").strip()
    return body.split("(", 1)[0].strip()


def _except_type(except_text: str) -> str:
    """The caught type of an `except` clause: '' for a bare `except:`, else the
    type expression (e.g. 'Exception', 'TransientError', '(A, B)')."""
    head = except_text.split(":", 1)[0].removeprefix("except").strip()
    return head.split(" as ", 1)[0].strip()


def check_layer_source(source: str, filename: str = "<layer>") -> list[Violation]:
    return list(_layer_violations(source, filename))


def declares_layer(source: str) -> bool:
    """Does this CONTENT declare a layer? The content-side twin of `is_layer_role(name)`.

    Keyed on the same signal `_layer_violations` dispatches on — a `@op_layer`/`@domain_layer`
    decorated def — for the same reason `yields_effect_op` exists: a file the coding machine
    writes has no name in `LAYER_ROLE_SRCS`, so a name-keyed gate cannot reach it."""
    root = SgRoot(source, "python").root()
    for dec in root.find_all(kind="decorated_definition"):
        decorators = " ".join(d.text() for d in dec.find_all(kind="decorator"))
        if "domain_layer" in decorators or "op_layer" in decorators:
            return True
    return False


def _layer_violations(source: str, filename: str) -> Iterator[Violation]:
    """Layer-authority lint, keyed on the layer decorator.

    - ``domain-layer-no-control-flow``: a ``@domain_layer`` must not yield a
      ``WorkflowOp`` constructor; it cannot suspend or alter control flow.
    - ``op-layer-alphabet``: an ``@op_layer`` must not yield a bare ``DomainOp``
      constructor (that would bypass ``Step`` and the durable grain); forward via
      ``yield op`` or yield a ``WorkflowOp``.
    - ``op-layer-no-broad-except``: an ``@op_layer`` must not catch broadly
      (bare ``except:`` / ``Exception`` / ``BaseException``); a broad catch would
      swallow the durable suspend signal that propagates through the layer's
      ``yield``. Catch only the specific error you handle.
    """
    root = SgRoot(source, "python").root()
    for dec in root.find_all(kind="decorated_definition"):
        decorators = " ".join(d.text() for d in dec.find_all(kind="decorator"))
        is_domain = "domain_layer" in decorators
        is_op = "op_layer" in decorators and not is_domain
        if not (is_domain or is_op):
            continue
        for node in dec.find_all(kind="yield"):
            callee = _yielded_callee(node.text())
            if is_domain and callee in WORKFLOW_OP_CTORS:
                yield Violation(
                    "domain-layer-no-control-flow",
                    f"a @domain_layer must not yield a control-flow op ({callee}); "
                    "domain layers are call-scoped and cannot suspend",
                    node.range().start.line + 1,
                    node.text(),
                    filename,
                )
            elif is_op and callee in DOMAIN_OP_CTORS:
                yield Violation(
                    "op-layer-alphabet",
                    f"an @op_layer must not yield a bare DomainOp ({callee}); "
                    "it would bypass Step and the durable grain — forward `yield op`",
                    node.range().start.line + 1,
                    node.text(),
                    filename,
                )
        if is_op:
            for node in dec.find_all(kind="except_clause"):
                if _except_type(node.text()) in BROAD_EXCEPTS:
                    yield Violation(
                        "op-layer-no-broad-except",
                        "an @op_layer must not catch broadly (bare/Exception/BaseException); "
                        "it would swallow the suspend signal — catch only your specific error",
                        node.range().start.line + 1,
                        node.text().split(":", 1)[0],
                        filename,
                    )


def _directive_flags(spec_text: str) -> tuple[str, bool | None]:
    """The (role, cache) a channel ``format_spec`` declares, for the static cache
    lint. Role defaults to ``user`` (the processor's default); cache is ``True`` for
    ``cache``, ``False`` for ``nocache``, ``None`` if the spec sets neither."""
    role = "user"
    cache: bool | None = None
    for tok in (t.strip() for t in spec_text.lstrip(":").split(";") if t.strip()):
        if tok == "cache":
            cache = True
        elif tok == "nocache":
            cache = False
        elif tok.startswith("role="):
            role = tok.removeprefix("role=").strip()
    return role, cache


# The independence lint's static slice: a resolved
# value written *visibly* into a t-string interpolation. `Done(`/`Repair(`
# constructed in place, or a `.resolve(...)` call with a positional argument
# (pathlib's `.resolve()` / `.resolve(strict=True)` stay clean: a keyword-only
# call is not a response resolution). Conservative, t-strings only; the
# authoritative check is the render-time `IndependenceError`.
_RESOLUTION_CTOR_RE = re.compile(r"^\{?\s*(?:Done|Repair)\(")
_RESOLVE_CALL_RE = re.compile(r"\.resolve\(\s*(?![A-Za-z_][A-Za-z0-9_]*\s*=)[^)\s]")
_TSTRING_PREFIX_RE = re.compile(r"""^[rR]?[tT][rR]?['"]""")


def check_channel_source(source: str, filename: str = "<channels>") -> list[Violation]:
    return list(_channel_violations(source, filename))


def _channel_violations(source: str, filename: str) -> Iterator[Violation]:
    """The channel lints, static and single-literal. Within one t-string literal, flag:

    | breach       | written visibly                                                      |
    |--------------|----------------------------------------------------------------------|
    | cache order  | an explicit ``cache`` after an explicit ``nocache`` in one role      |
    | independence | a resolution in place: ``Done(…)``, ``Repair(…)``, ``.resolve(arg)`` |

    Each reasons only about one **t-string** literal; f-strings are not channel templates and
    are not scanned. Cross-``Template`` composition and dynamically threaded values are caught
    by the authoritative render-time checks, ``CacheOrderError`` and ``IndependenceError``.
    """
    root = SgRoot(source, "python").root()
    for string_node in root.find_all(kind="string"):
        if not _TSTRING_PREFIX_RE.match(string_node.text()):
            continue  # only t-strings are channel templates
        seen_volatile: dict[str, bool] = {}
        for interp in string_node.find_all(kind="interpolation"):
            inner = interp.text()
            if _RESOLUTION_CTOR_RE.match(inner) or _RESOLVE_CALL_RE.search(inner):
                yield Violation(
                    "independence",
                    "an interpolation renders a resolved value — a prior "
                    "resolution re-entering a render is a turn boundary "
                    "(control axis), not composition",
                    interp.range().start.line + 1,
                    interp.text(),
                    filename,
                )
            spec = interp.find(kind="format_specifier")
            if spec is None:
                continue
            role, cache = _directive_flags(spec.text())
            if cache is True and seen_volatile.get(role):
                yield Violation(
                    "volatile-last-cache",
                    f"a `cache` segment follows a `nocache` one in role {role!r}; "
                    "an uncached segment poisons every downstream cache point",
                    interp.range().start.line + 1,
                    interp.text(),
                    filename,
                )
            elif cache is False:
                seen_volatile[role] = True


def check_channel_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_channel_source(path.read_text(), filename=str(path))


def check_deps_source(source: str, filename: str, forbidden: frozenset[str]) -> list[Violation]:
    return list(_deps_violations(source, filename, forbidden))


def _under(module: str, prefix: str) -> bool:
    """Is ``module`` the prefix itself or something inside it?

    Segment-wise, so ``effective.coding`` does not match ``effective.codingx`` — the false
    positive a bare ``startswith`` would have."""
    return module == prefix or module.startswith(f"{prefix}.")


def _relative_base(path: Path) -> str:
    """The package a relative import inside ``path`` counts levels down from.

    A regular module resolves ``from .`` to its PARENT package; a package's own ``__init__.py``
    resolves it to that package. `_module_path` maps both to a dotted name but loses which one it
    was, so the discrimination is made here, from the filename."""
    module = _module_path(path)
    if path.name == "__init__.py":
        return module
    return module.rpartition(".")[0]


def _imported_modules(text: str, base: str) -> list[str]:
    """Every module an import statement names, fully dotted, with relative imports RESOLVED.

    Three things `_import_root` does not do, each a hole in the rule that reads it: it truncates
    to the first segment (so no intra-package edge is expressible), it returns ``""`` for
    ``from .states import X`` (so a relative import matches no forbidden set and goes unpoliced),
    and it inspects only the first name of ``import a, b``."""
    text = text.strip()
    if text.startswith("from "):
        spec = text[len("from ") :].split(" import")[0].strip()
        level = len(spec) - len(spec.lstrip("."))
        if not level:
            return [spec]
        # `from . import x` inside `a.b.c` is `a.b`; each extra dot climbs one more. A climb past
        # the top-level package is invalid Python ("attempted relative import beyond top-level
        # package"), so it cannot appear in a module that imports — resolving it to SOMETHING
        # would invent a module name and then report against it.
        if level - 1 > base.count("."):
            return []
        climbed = base.rsplit(".", level - 1)[0] if level > 1 else base
        tail = spec[level:]
        return [f"{climbed}.{tail}" if tail else climbed]
    if text.startswith("import "):
        return [
            name
            for raw in text[len("import ") :].split(",")
            if (name := raw.split(" as ")[0].strip())
        ]
    return []


def _deps_violations(source: str, filename: str, forbidden: frozenset[str]) -> Iterator[Violation]:
    """Seam dependency-direction lint: flag any import landing under a prefix in ``forbidden``
    (the set the owning module may not reach)."""
    if not forbidden:
        return
    base = _relative_base(Path(filename))
    root = SgRoot(source, "python").root()
    for kind in ("import_statement", "import_from_statement"):
        for node in root.find_all(kind=kind):
            for imported in _imported_modules(node.text(), base):
                hit = next((f for f in sorted(forbidden) if _under(imported, f)), None)
                if hit is None:
                    continue
                yield Violation(
                    "seam-dep-direction",
                    f"must not import {hit!r} ({SEAM_REASONS.get(hit, _PRIVATE_REASON)})",
                    node.range().start.line + 1,
                    node.text(),
                    filename,
                )


def _owning_package(path: Path) -> str:
    """The top-level package a file belongs to: the path segment after the last
    ``src/`` (so an absolute path under a repo *named* ``effective`` is not
    mis-owned), or the first segment when there is no ``src`` anchor."""
    parts = path.parts
    srcs = [i for i, p in enumerate(parts) if p == "src"]
    if srcs and srcs[-1] + 1 < len(parts):
        return parts[srcs[-1] + 1]
    return parts[0] if parts else ""


def seam_forbidden_for(path: Path) -> frozenset[str]:
    """The forbidden import prefixes for a file: the UNION of every `SEAM_FORBIDDEN` row that
    applies to it, empty if none does.

    A row applies when its key prefixes the file's own module and the file is not itself part of
    the half that row protects. The union rather than a longest match is deliberate: a longest
    match would let `effective.machine`'s intra-package row displace `effective`'s
    cross-package row, so adding a narrow rule would quietly delete a wide one."""
    module = _module_path(path)
    out: set[str] = set()
    for owner, forbidden in SEAM_FORBIDDEN.items():
        if not _under(module, owner):
            continue
        if any(_under(module, f) for f in forbidden):
            continue  # the file IS part of the half this row protects
        out |= forbidden
    return frozenset(out)


def check_deps_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_deps_source(path.read_text(), str(path), seam_forbidden_for(path))


# --- the lazy-import rule (`--lazy-imports`) --------------------------------------------
#
# An import inside a function body is almost always a smell: a deferred stdlib import buys
# nothing, and a cycle it hides is a cycle in the design. The allowlist makes every warranted lazy
# import visible in one place with a stated reason, so "is this one warranted?" is answered by
# reading a table rather than by re-deriving the cycle.
#
# Keyed by (path below `src/`, imported ROOT module) rather than by line, so the entry survives
# the function moving. A reason is required: an entry with an empty reason is itself a violation,
# because a silent allowlist is the thing this rule exists to prevent.
LAZY_IMPORT_ALLOWED: dict[tuple[str, str], str] = {
    # 1. Circular-import break — the module we need already imports us.
    ("effective/govern.py", "effective"): "op_key: handlers.base imports api, which govern needs",
    (
        "effective/ops.py",
        "effective",
    ): "is_engine_internal: checkpoints imports handlers.base, which imports ops",
    # 2. Optional / dev-group-only SDKs — not in the runtime's lean dependency set.
    (
        "effective/interpreters/openai.py",
        "openai",
    ): "heavy optional SDK; only the model-caller paths need it",
    ("agent/skillsbench.py", "yaml"): "dev-group dep; bench tooling only",
    # 3. An SDK name kept private to its one use.
    ("effective/handlers/absurd.py", "absurd_sdk"): "a PRIVATE SDK name; kept at its one use",
}
LAZY_IMPORT_ALLOWED |= {
    (entry["path"], entry["module"]): entry["reason"]
    for entry in configured("lazy_import_allowed") or ()
}
"""Every lazy import that is warranted, and why. Adding an entry is a review decision.

| category                 | warrant                                         |
|--------------------------|-------------------------------------------------|
| circular-import break    | the module needed already imports this one      |
| optional or dev-only SDK | outside the runtime's lean dependency set       |
| SDK name kept to one use | an SDK-internal name stays at the site using it |
"""


def _enclosing_function(node: Any) -> Any | None:
    """The nearest `function_definition` ancestor, or None if the node is at module level."""
    cur = node.parent()
    while cur is not None:
        if cur.kind() == "function_definition":
            return cur
        cur = cur.parent()
    return None


def check_lazy_imports_source(source: str, filename: str = "<module>") -> list[Violation]:
    return list(_lazy_import_violations(source, filename))


def _lazy_import_violations(source: str, filename: str) -> Iterator[Violation]:
    key_path = _path_below_src(Path(filename))
    root = SgRoot(source, "python").root()
    for kind in ("import_statement", "import_from_statement"):
        for node in root.find_all(kind=kind):
            if _enclosing_function(node) is None:
                continue
            module = _import_root(node.text())
            if LAZY_IMPORT_ALLOWED.get((key_path, module)):
                continue
            yield Violation(
                "lazy-import",
                f"import of {module!r} inside a function body. If it breaks a cycle, is an "
                f"optional/dev-only SDK, or is a prod-only side-effecting dep, add "
                f'`("{key_path}", "{module}"): "<reason>"` to `LAZY_IMPORT_ALLOWED`; '
                f"otherwise hoist it to module level",
                node.range().start.line + 1,
                node.text().replace("\n", " "),
                filename,
            )


def _path_below_src(path: Path) -> str:
    """The allowlist key: the path under the last `src/`, so an entry is stable across the
    absolute/relative spellings a caller might pass."""
    parts = path.parts
    srcs = [i for i, p in enumerate(parts) if p == "src"]
    return "/".join(parts[srcs[-1] + 1 :]) if srcs else str(path)


def check_lazy_imports_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_lazy_imports_source(path.read_text(), str(path))


# --- the canonical-read rule (`--ledger-reads`) -----------------------------------------
#
# One predicate fences the counterfactual: a hypothetical row is a fork's, so every projection
# reads `WHERE NOT hypothetical`. A query that omits it lets a fork's row win a last-event-wins
# fold and reach a downstream projection, and a convention fails exactly where a raw query sits
# beside an ORM one that carries the predicate. So a SQL string that reads FROM ledger must
# mention `hypothetical`, or say why not.
#
# Deliberately dumb: a substring check over string literals rather than a SQL parse. It cannot be
# fooled into a false PASS by anything a real query does, and its false FAILs are cheap to
# silence with the escape hatch below. The SQLModel path is covered by the same rule via
# `select(LedgerEntry)`, whose `.where(col(LedgerEntry.hypothetical)...)` mentions it too.
_LEDGER_READ_RE = re.compile(r"\bfrom\s+ledger\b", re.IGNORECASE)
_LEDGER_ESCAPE = "lint: ledger-read-not-canonical"
"""Put this in a comment on the query's line to opt out — for a read that DELIBERATELY sees
hypothetical rows (a fork's own marginal, an admin view of a counterfactual). It must name a
reason after the tag; the point is a visible, reviewable decision, not a silent omission."""


def check_ledger_reads_source(source: str, filename: str = "<sql>") -> list[Violation]:
    return list(_ledger_read_violations(source, filename))


def _ledger_read_violations(source: str, filename: str) -> Iterator[Violation]:
    """Flag a SQL string literal that reads `FROM ledger` without mentioning `hypothetical`.

    Scans **string literals** (plain, f-, and t-strings alike — all three build SQL here) and
    joins each literal with its implicitly-concatenated neighbours, because a multi-line query
    puts `FROM ledger` and the `WHERE` clause in *different* literals. The unit is therefore
    the whole concatenation group, which is also the unit a reader reasons about.
    """
    lines = source.splitlines()
    root = SgRoot(source, "python").root()
    for group in root.find_all(kind="concatenated_string"):
        yield from _check_sql_text(group.text(), group, lines, filename)
    for node in root.find_all(kind="string"):
        if (parent := node.parent()) is not None and parent.kind() == "concatenated_string":
            continue  # already covered as part of its group
        yield from _check_sql_text(node.text(), node, lines, filename)


def _check_sql_text(text: str, node: Any, lines: list[str], filename: str) -> Iterator[Violation]:
    if not _LEDGER_READ_RE.search(text) or "hypothetical" in text:
        return
    start = node.range().start.line
    end = node.range().end.line
    # The opt-out may sit anywhere ADJACENT to the query — the line above (the usual
    # `conn.execute(  # lint: ...` spot), inside it, or the line below (`).fetchall()`).
    # A query is a multi-line expression and an author puts the comment where it reads best.
    window = range(max(start - 1, 0), min(end + 2, len(lines)))
    if any(_LEDGER_ESCAPE in lines[i] for i in window):
        return  # an explicit, reasoned opt-out
    yield Violation(
        "ledger-read-not-canonical",
        "a read FROM ledger does not mention `hypothetical`; a fork's row would join the "
        "canonical fold (the canonical view is `WHERE NOT hypothetical`). Add the "
        f"predicate, or opt out with a `# {_LEDGER_ESCAPE}: <reason>` comment",
        start + 1,
        text.splitlines()[0][:90],
        filename,
    )


def check_ledger_reads_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_ledger_reads_source(path.read_text(), filename=str(path))


# --- the SQL-boundary rule (`--sql-templates`) -------------------------------------------
#
# An f-string that builds an IDENTITY, SQL, or a PROMPT is a defect, because at those three seams
# the structure IS the safety property. This is that rule
# with teeth for the SQL seam, scoped BY POSITION (the first argument of an `.execute(...)`), not
# by f-string-ness — an f-string rendering a LIKE *pattern* one line above is fine, and common.
#
# Two shapes are refused:
#   - `sql-params-split` — `.execute("… ? …", (a, b))`. The old DB-API form. Not an injection
#     (the driver still binds), but the structure and its data sit in two arguments a reader has
#     to zip up by eye, and every hole's meaning is positional. `effective.sql.bind` puts them
#     back in one template: `.execute(*bind(t"… {a} … {b}"))`.
#   - `sql-built-by-formatting` — an f-string, `%`, `.format`, or `+` in the SQL position. This
#     one IS the Bobby Tables shape: the delimiter/data distinction is destroyed before the
#     driver ever sees the statement.
#
# A single static literal is untouched: a query with no data to separate has made no structural
# decision to lose. `executemany` is out of scope — it binds many parameter sets, which is a
# different contract than `bind` composes.
_SQL_ESCAPE = "lint: sql-not-templated"
"""Opt out with a reason — for a call whose first argument is genuinely not a statement this
processor can compose (a driver-specific form, a schema script)."""


def _is_sql_call(call: Any) -> bool:
    return (fn := call.field("function")) is not None and fn.text().endswith(".execute")


def _call_arguments(call: Any) -> list[Any]:
    args = call.field("arguments")
    return [] if args is None else [a for a in args.children() if a.kind() not in ("(", ")", ",")]


def _formats_a_string(node: Any) -> bool:
    """Whether this SQL-position expression was BUILT rather than written: an f-string anywhere
    inside it, a `%`/`+` on a string, or a `.format(` call. A t-string is not "built" — its holes
    survive to the processor, which is the entire point."""
    if any("f" in start.text().lower() for start in node.find_all(kind="string_start")):
        return True
    if node.kind() == "binary_operator" and node.text().lstrip()[:1] in ("'", '"'):
        return True
    return (
        node.kind() == "call"
        and node.field("function") is not None
        and (node.field("function").text().endswith(".format"))
    )


def check_sql_templates_source(source: str, filename: str = "<sql>") -> list[Violation]:
    return list(_sql_template_violations(source, filename))


def _sql_template_violations(source: str, filename: str) -> Iterator[Violation]:
    lines = source.splitlines()
    root = SgRoot(source, "python").root()
    for call in root.find_all(kind="call"):
        if not _is_sql_call(call) or not (args := _call_arguments(call)):
            continue
        line = call.range().start.line
        window = range(max(line - 1, 0), min(call.range().end.line + 2, len(lines)))
        if any(_SQL_ESCAPE in lines[i] for i in window):
            continue
        first = args[0]
        excerpt = first.text().splitlines()[0][:90]
        if _formats_a_string(first):
            yield Violation(
                "sql-built-by-formatting",
                "SQL built by string formatting — the delimiter/data distinction is destroyed "
                "before the driver sees it (Bobby Tables). Compose the statement as a t-string: "
                '`.execute(*bind(t"… {value} … {table:i}"))`',
                line + 1,
                excerpt,
                filename,
            )
        elif len(args) > 1 and first.kind() in ("string", "concatenated_string"):
            yield Violation(
                "sql-params-split",
                "SQL and its parameters are split across two arguments — the structure and its "
                "data have to be zipped up by eye, and each hole's meaning is positional. "
                'Compose them in one template: `.execute(*bind(t"… {value} …"))`',
                line + 1,
                excerpt,
                filename,
            )


def check_sql_templates_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_sql_templates_source(path.read_text(), filename=str(path))


# --- the key-composition rule (`--key-composition`) ----------------------------------------
#
# A prefix concatenated onto a name is judged by the TYPE OF THE RIGHT OPERAND: a frame atom is a
# render backend, a finished identity is composition. So the rule is scoped BY POSITION; one
# phrased "no f-strings near keys" flags the correct backend (an error message, a SQL `LIKE`
# pattern) and drowns.
#
# It asks one question, *is a value that was BUILT being consumed as an IDENTITY?*, and only in
# the argument slots below, which are its whole domain:
#
#   - The render backends pass by construction. `_key_segment`'s `f"{item.value}"` and
#     `scope_prefix`'s `f"{text}{TERM_SEPARATOR}"` sit *inside* the processor, below the
#     decision, and never in an identity slot; `tests/test_lint.py` pins both by name.
#   - An untyped `str` name is a `ty` error at the seam, since `AwaitEvent.name` and
#     `TaskContext.await_event` take a `Key`. What is left for a syntactic rule is a finished
#     identity concatenated by hand and a template's leading positions restated by hand, in the
#     positions where the substrate still takes a `str`, chiefly `Step.name`.
_KEY_ESCAPE = "lint: key-composition"
"""Opt out with a reason — for a name that is genuinely not an identity this composer owns (a
FOREIGN grammar's, e.g. the vendored SDK's `$awaitEvent:` checkpoint spelling)."""

_IDENTITY_CALLS = ("step", "Step")
"""Calls whose FIRST positional argument is an op identity. `step(...)` is the author surface and
`Step(...)` the op itself; `.step(...)` on a ctx matches too, since the check is on the callee's
trailing name."""

_IDENTITY_KEYWORDS = ("idempotency_key", "event_id")
"""Keyword slots that take an identity wherever they appear. `name=` is deliberately NOT here: it
is the most overloaded keyword in the tree (a task name, a tool name, a span name), so it is
honored only on the calls above, where it IS the op key."""


def check_key_composition_source(source: str, filename: str = "<keys>") -> list[Violation]:
    return list(_key_composition_violations(source, filename))


def _identity_slots(call: Any) -> Iterator[Any]:
    """The argument nodes of `call` that are consumed as an identity — the rule's whole domain."""
    fn = call.field("function")
    if fn is None:
        return
    callee = fn.text().rsplit(".", 1)[-1]
    args = _call_arguments(call)
    if callee in _IDENTITY_CALLS and args and args[0].kind() != "keyword_argument":
        yield args[0]
    for arg in args:
        if arg.kind() != "keyword_argument":
            continue
        key = arg.field("name")
        value = arg.field("value")
        if key is None or value is None:
            continue
        if key.text() in _IDENTITY_KEYWORDS or (
            key.text() == "name" and callee in _IDENTITY_CALLS
        ):
            yield value


def _composed_bindings(source: str) -> dict[str, int]:
    """Names bound to a BUILT string, mapped to the line that built them — ONE hop of dataflow.

    **Why the rule reaches past the argument slot.** A value is often assigned first and passed
    second (`action_key = f"…"` on one line, `step(action_key, …)` on the next, as in `code.py`),
    and a slot-only rule is blind to that shape.

    **One hop, module-wide, no scope analysis: a deliberate ceiling.** It over-matches
    only when a name bound to an f-string somewhere is *coincidentally* the same name used in an
    identity slot elsewhere, which the `# lint: key-composition` escape settles per site; it
    under-matches on two hops, an f-string through a helper's return, or a name bound in a
    comprehension. A dataflow-complete version wants types, and the seams that HAVE types
    (`Step.name` is the last identity slot still typed `str`) should get them instead: that is the
    structural fix this rule stands in for.
    """
    bound: dict[str, int] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:  # pragma: no cover - the caller's own parse reports it
        return bound
    for node in ast.walk(tree):
        targets = (
            node.targets
            if isinstance(node, ast.Assign)
            else [node.target]
            if isinstance(node, ast.AnnAssign)
            else []
        )
        value = getattr(node, "value", None)
        if value is None or not _ast_is_built(value):
            continue
        for target in targets:
            if name := _identifier(target):
                bound[name] = value.lineno
    return bound


def _ast_is_built(node: ast.expr) -> bool:
    """The `ast` twin of `_formats_a_string`: an f-string, a `+` on a string, or `.format(`."""
    if isinstance(node, ast.JoinedStr):
        return True
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return any(
            # lint: totality(foreign) — a compound test over `ast.*`, a hierarchy with no
            # union to declare. This rule's own domain, which is the honest place to say so.
            isinstance(side, ast.Constant) and isinstance(side.value, str)
            for side in (node.left, node.right)
        ) or any(isinstance(side, ast.JoinedStr) for side in (node.left, node.right))
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "format"
    )


def _escape_window(lines: list[str], at: int) -> Iterator[int]:
    """The reported line (1-indexed) and the CONTIGUOUS comment block directly above it.

    Walking the block rather than a fixed ±N is what lets a reason be as long as it needs to be:
    the two-arity opt-outs in `code.py`/`skills.py` each take four lines to say why a conversion
    is a durable-bytes migration, and a fixed window would have silently stopped honoring them."""
    yield at - 1
    i = at - 2
    while i >= 0 and lines[i].lstrip().startswith("#"):
        yield i
        i -= 1


def _key_composition_violations(source: str, filename: str) -> Iterator[Violation]:
    lines = source.splitlines()
    composed = _composed_bindings(source)
    root = SgRoot(source, "python").root()
    for call in root.find_all(kind="call"):
        for slot in _identity_slots(call):
            # Either the slot holds the composition, or it holds a NAME that was composed —
            # the one hop `_composed_bindings` buys.
            built_at = composed.get(slot.text()) if slot.kind() == "identifier" else None
            if not _formats_a_string(slot) and built_at is None:
                continue
            # Report at the line that BUILT the name, not the line that consumed it: that is
            # where the fix goes, and for a one-hop the consuming line shows only `action_key,`.
            at = built_at if built_at is not None else slot.range().start.line + 1
            excerpt = lines[at - 1].strip()[:90] if built_at is not None else slot.text()
            # The escape window hangs off the REPORTED line, not off the call: the reason belongs
            # where the fix would go, and these reasons run to a paragraph (the two-arity ones
            # do), so it reads the line itself plus the comment block above it.
            if any(_KEY_ESCAPE in lines[i] for i in _escape_window(lines, at)):
                continue
            yield Violation(
                "key-composition",
                "an op identity BUILT by string formatting — the composer is what makes a key "
                "injective, and a hand-rolled one is invisible to the key registry, so nothing "
                "can decode it back to its fields or catch the day it collides. Compose it: "
                '`compose_key(t"tag:{field}")` (take `.stored()` where a `str` is still '
                "required), or `Key.prefixed` if you are applying a frame to a finished identity",
                at,
                excerpt.splitlines()[0][:90],
                filename,
            )


def check_key_composition_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_key_composition_source(path.read_text(), filename=str(path))


# --- the role-coverage rule (`--role-coverage`) --------------------------------------------
#
# `check_source` is the gate the whole replay model rests on, and it runs on a CURATED list. A
# curated list is only as good as the last person who remembered it, and a file left off it keeps
# the gate green.
#
# This makes the list PROVABLE: a file that yields an effect op through a wrapper it imported is
# workflow-role, and must be in `WORKFLOW_ROLE_SRCS` or say why not. The same move as
# deriving `RESERVED_AUTHORITY_TAGS` from the `AuthorityTag` declarations rather than curating it.
#
# **Why this is a coverage check where `--deps` scans the whole tree.** Determinism is a
# property of a FUNCTION rather than of a file, and the two live together:
# `src/agent/contrastbench.py` builds fixtures with `random` at lines 146-347 and authors its
# workflow at line 1003. Scanning that file whole reports 7 violations and none of them is real.
# Until `check_source` is function-scoped, membership stays a judgment, and the gate's job is to
# make sure the judgment was MADE.
_EFFECT_MODULES = ("effective", "effective.api", "effective.combinators", "effective.react")
"""The SEED of the effect-module domain: where a wrapper is defined.

Each entry is an exact `from {m} import` spelling, so as a whole domain the list would enumerate
spellings of one referent and always be one short. The property is transitive: a module that
yields an effect op IS an effect module, so `effective.improve` is one, and
`agent/bench_sweep.py`'s `yield from improve(...)` authors a workflow through it.
`_effect_module_closure` computes the domain, seeded from here."""
NOT_WORKFLOW_ROLE: dict[str, str] = {
    "src/effective/fork.py": "an interpreter of recorded traces, not a workflow: it yields ops "
    "to drive someone else's generator",
    "src/agent/contrastbench.py": "a bench harness whose workflow is one function among fixture "
    "builders; file-scoped determinism rules cannot separate them yet",
}
"""Files that yield effect ops and are deliberately NOT workflow-role. A reason, not a bare
entry: the point is a decision on the record, and an unexplained exemption is the curated list
this rule replaces."""


def _module_path(path: Path) -> str:
    """The dotted module a file is imported as — `src/effective/improve.py` -> `effective.improve`.

    Derived from the path below `src/` so it matches the `from {m} import` spelling a sibling
    writes, and a package `__init__.py` maps to the package itself (`agent/__init__.py` ->
    `agent`), which is what a re-export is imported from.

    A file under no `src/` root — a script, a research tree, a `tmp_path` fixture — falls back to
    its STEM, which is how a sibling in the same directory imports it. That is an approximation
    and it errs the safe way: two stems could collide and widen the closure by one module, where
    the alternative (no name at all) would silently drop an intermediary from it."""
    if "src" not in path.parts:
        return path.stem.removesuffix("__init__") or path.parent.name
    key = _path_below_src(path).removesuffix(".py")
    return key.removesuffix("/__init__").replace("/", ".")


def _effect_module_closure(sources: dict[str, str]) -> tuple[str, ...]:
    """The effect-module domain as a FIXPOINT over the files being scanned.

    Seeded with `_EFFECT_MODULES` (where the wrappers are defined), then any scanned module whose
    own source yields an op from an already-known effect module joins the set, until nothing
    changes. That is the property the rule actually wants — *a module that yields an effect op is
    an effect module* — rather than the list of spellings that kept going one short.

    It reaches what no enumeration could: `agent/bench_sweep.py` writes `yield from improve(...)`,
    and `effective.improve` is an effect module only because it yields `gather`/`scoped` itself.

    Its domain is the SCANNED FILES, and saying so is the point of this paragraph — the closure
    can only see modules in `paths`. `just lint` passes its source roots,
    so an intermediary outside those roots is invisible here, exactly as it is to every other
    file-scanning rule. That is a bound to state, not a bug to hide."""
    known = set(_EFFECT_MODULES)
    pending = dict(sources)
    while True:
        found = {m for m, source in pending.items() if _first_effect_yield(source, tuple(known))}
        if not found:
            return tuple(sorted(known))
        known |= found
        pending = {m: s for m, s in pending.items() if m not in found}


def check_role_coverage(paths: Iterable[str | Path]) -> list[Violation]:
    """Every file that yields an effect op is workflow-role, exempt, or a violation."""
    out: list[Violation] = []
    scanned = [Path(p) for p in paths]
    readable: dict[Path, str] = {}
    for p in scanned:
        try:
            readable[p] = p.read_text()
        except OSError:
            continue
    modules = _effect_module_closure({_module_path(p): s for p, s in readable.items()})
    for p, source in readable.items():
        key = _path_below_src(p)
        rel = str(p)
        if rel.startswith("tests/") or "/tests/" in rel:
            continue  # tests author throwaway workflows inline; scanning them buys noise
        if not (line := _first_effect_yield(source, modules)):
            continue
        if _in_role_set(rel, WORKFLOW_ROLE_SRCS):
            continue
        if any(rel.endswith(k) or key.endswith(k) for k in NOT_WORKFLOW_ROLE):
            continue
        out.append(
            Violation(
                "role-coverage",
                f"{rel} yields an effect op but is not in `WORKFLOW_ROLE_SRCS`, so the "
                f"determinism-boundary lint never sees it. Add it, or add it to "
                f"`NOT_WORKFLOW_ROLE` with the reason it is not a workflow",
                line,
                "yield from <effect wrapper>(...)",
                rel,
            )
        )
    return out


# --- the layer-coverage rule (`--layer-coverage`) -------------------------------------------
#
# `--layers` scans a CURATED list of layer-role files, so a layer authored anywhere else would be
# invisible to it, and a broad `except` there would swallow the durable suspend signal unchecked.
# This rule closes the domain by composing the two statements of "is a layer":
# `is_layer_role(name)`, by file name, and `declares_layer`, by content. A file that declares a
# layer is layer-role, exempt in `NOT_LAYER_ROLE` with a reason, or a violation.
NOT_LAYER_ROLE: dict[str, str] = {}
"""Files that declare a layer and are deliberately NOT layer-role. A reason, not a bare entry —
the same contract as `NOT_WORKFLOW_ROLE`, and empty today because no such file exists."""


def check_layer_coverage(paths: Iterable[str | Path]) -> list[Violation]:
    """Every file that declares a layer is layer-role, exempt, or a violation."""
    out: list[Violation] = []
    for p in (Path(x) for x in paths):
        try:
            source = p.read_text()
        except OSError:
            continue
        rel = str(p)
        if rel.startswith("tests/") or "/tests/" in rel:
            continue  # tests declare throwaway layers inline; scanning them buys noise
        if not declares_layer(source):
            continue
        if _in_role_set(rel, LAYER_ROLE_SRCS):
            continue
        key = _path_below_src(p)
        if any(rel.endswith(k) or key.endswith(k) for k in NOT_LAYER_ROLE):
            continue
        out.append(
            Violation(
                "layer-coverage",
                f"{rel} declares an `@op_layer`/`@domain_layer` but is not in "
                f"`LAYER_ROLE_SRCS`, so the layer-authority lint never sees it. Add it, or add "
                f"it to `NOT_LAYER_ROLE` with the reason it is not layer-role",
                _first_layer_line(source),
                "@op_layer / @domain_layer",
                rel,
            )
        )
    return out


def _first_layer_line(source: str) -> int:
    """The line of the first layer decorator, so the violation points at the declaration."""
    for n, line in enumerate(source.splitlines(), start=1):
        if line.lstrip().startswith(("@op_layer", "@domain_layer")):
            return n
    return 1


def _imported_names(text: str) -> set[str]:
    """The names an `import_from_statement` binds, parenthesized or not.

    The strip ORDER matters: brackets, then whitespace, then alias. Stripping whitespace while the
    opening paren is attached turns a parenthesized import's FIRST name into `"\\n    step"`, which
    matches no callee, while the middle names survive and hide the defect."""
    names = text.split(" import ", 1)[1]
    return {
        stripped
        for n in names.split(",")
        if (stripped := n.strip("()\n\t ").split(" as ")[-1].strip())
    }


def _first_effect_yield(source: str, modules: tuple[str, ...] = _EFFECT_MODULES) -> int:
    """The line of the first `yield`/`yield from <wrapper>(...)` whose wrapper was IMPORTED from
    an effect module, or 0 if there is none.

    `modules` defaults to the SEED, which is right for a single-source caller like
    `effective.coding.gate` (it has one file's content and no tree to close over).
    `--role-coverage` passes the closure `_effect_module_closure` computed from the files it is
    scanning, so a workflow reached through an intermediary is visible there.

    **Both yield forms.** A bare `yield step(...)` is the shape `require-yield-from` exists to
    catch; a test for `yield from` alone would keep a file whose only defect is that one out of the
    role set, so the determinism lint would never see it.

    Imported, not merely named, is what keeps `effective.api` itself out: it DEFINES the wrappers
    and bare-`yield`s the op constructors, which is the one place that is correct. **The IMPORT
    test carries that exclusion**: api.py imports from
    `effective.ops` / `.domain` / `.keys`, none of them an `_EFFECT_MODULES` spelling, so the bare
    yield form needs no carve-out beside it
    (`test_widening_to_a_bare_yield_still_excludes_the_wrapper_layer`)."""
    root = SgRoot(source, "python").root()
    imported: set[str] = set()
    for node in root.find_all(kind="import_from_statement"):
        text = node.text()
        if any(f"from {m} import" in text for m in modules):
            imported |= _imported_names(text)
    if not imported:
        return 0
    # Longest prefix first: `yield from` is not a bare `yield`, and both are workflow-role.
    for node in root.find_all(kind="yield"):
        text = node.text()
        for prefix in ("yield from ", "yield "):
            if text.startswith(prefix):
                callee = text.removeprefix(prefix).split("(", 1)[0].strip()
                if callee in imported:
                    return node.range().start.line + 1
                break
    return 0


def yields_effect_op(source: str) -> bool:
    """Does this CONTENT author a workflow? The content-side twin of `is_workflow_role(name)`.

    One operational definition, two callers. `--role-coverage` asks it of a file on disk to prove
    the curated role set is complete; `effective.coding.gate` asks it of content the machine just
    wrote, which has no name anyone curated. Sharing the definition is what keeps the two from
    drifting into disagreeing about what a workflow is."""
    return _first_effect_yield(source) != 0


# --- the working-note rule (`--working-notes`) --------------------------------------------
#
# A working note is a transient: it records what you were mid-way through, and it is discharged
# before the commit the way a red test is fixed before the commit. Both halves: dated notes are
# allowed in COMMENTS and never reach a commit; a docstring may never carry one at all, because a
# docstring says what a thing IS and a reader cannot tell a stale promise from a current contract.
#
# So this gate refuses outright rather than ageing them. The DATE is required anyway, because an
# undated note cannot be triaged by anyone but its author, and because a note that does reach a
# commit (amended in, or approved) should say how old it is on sight.
_NOTE_TAGS = ("TO" + "DO", "FIX" + "ME", "XX" + "X", "WI" + "P")  # split: do not match own source
_DATED_NOTE = re.compile(rf"\b({'|'.join(_NOTE_TAGS)})\((\d{{4}}-\d{{2}}-\d{{2}})\)")
_BARE_NOTE = re.compile(rf"\b({'|'.join(_NOTE_TAGS)})\b")
_NOTE_ESCAPE = "lint: working-note"
"""Opt out on the offending line — for a file whose SUBJECT is these tags (a rubric fixture, this
rule's own tests). Name a reason after the tag; a silent hatch is how a gate stops meaning
anything. A docstring's hatch goes on its `def`/`class` line, which is where the violation is
reported."""


def check_working_notes_source(source: str, filename: str = "<module>") -> list[Violation]:
    return list(_working_note_violations(source, filename))


def _working_note_violations(source: str, filename: str) -> Iterator[Violation]:
    # The line itself, and the one above it — a `def` too long to carry a trailing comment
    # (ruff's line limit) puts the hatch on the line before, which is where a reader looks anyway.
    hatched = {line for line, text in _comment_lines(source) if _NOTE_ESCAPE in text}
    exempt = hatched | {line + 1 for line in hatched}
    for line, text in _comment_lines(source):
        if line in exempt:
            continue
        if match := _BARE_NOTE.search(text):
            dated = _DATED_NOTE.search(text)
            yield Violation(
                "working-note",
                f"{match.group(1)} working note in a committed comment. Discharge it before "
                f"committing — it is a transient, like a red test"
                if dated
                else f"undated {match.group(1)} in a comment. Date it "
                f"`{match.group(1)}(YYYY-MM-DD):` so it can be triaged, then discharge it "
                f"before committing",
                line,
                text.strip(),
                filename,
            )
    for line, text in _docstrings(source):
        if line in exempt:
            continue
        if match := _BARE_NOTE.search(text):
            yield Violation(
                "working-note",
                f"{match.group(1)} in a DOCSTRING. A docstring says what a thing IS; a reader "
                f"cannot tell a stale promise from a current contract. Move the note to a dated "
                f"comment, or the design question to an ADR",
                line,
                text.strip().replace("\n", " ")[:80],
                filename,
            )


def _comment_lines(source: str) -> Iterator[tuple[int, str]]:
    """Comment tokens only — a string literal mentioning one of these tags is not a note."""
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT:
                yield token.start[0], token.string
    except tokenize.TokenError, IndentationError, SyntaxError:
        return


def _docstrings(source: str) -> Iterator[tuple[int, str]]:
    """Every module/class/function docstring, by AST — not every string literal."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, holders) and (doc := ast.get_docstring(node, clean=False)):
            # The HOLDER's line, not the string's: that is where a reader looks, and it gives
            # the opt-out somewhere to sit (a docstring cannot carry a trailing comment).
            yield getattr(node, "lineno", 1), doc


def check_working_notes_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_working_notes_source(path.read_text(), str(path))


# --- the public-prose rule (`--public-prose`) ----------------------------------------------
#
# `effective/` and `agent/` are the substrate, so their prose is read by strangers who have none
# of our context. Two categories of true-but-wrong detail keep reaching it, and the
# reason they are hard to self-catch is that both are FACTUAL and are usually how the author found
# the thing they are documenting: our own deployment's moving parts, and one cloud vendor's product
# names. Both are provenance. Provenance belongs in the commit message; a docstring gets the
# property that holds for any adopter.
#
# Scoped BY PATH rather than by a phrase denylist over the whole tree: a deployment's own scripts
# SHOULD name its moving parts and its vendor, because that is what they are about. The rule is
# "substrate prose names no deployment and no vendor", a statement about two directories.
#
# Deliberately a small list of NOUNS rather than a general "is this operational?" judgment: a gate
# is bounded by what it scans, and a short list that never false-fires is worth more than a broad
# one nobody trusts. A deployment adds the nouns of its own moving parts here; this tree has none.
_PRIVATE_OPS_NOUNS: tuple[str, ...] = tuple(configured("private_ops_nouns") or ())
# A first list including `S3` and `BigQuery` produced hits that were mostly legitimate: a vendor
# noun that is also a PROTOCOL or DIALECT name, or an analogy. A gate whose escape hatch is the
# common case has stopped meaning anything, so the list is only services that can name nothing BUT
# where something is deployed.
_VENDOR_NOUNS = ("Cloud Logging", "Cloud Run", "Cloud Scheduler", "Secret Manager", "Cloud Build")
_PUBLIC_PROSE_ROOTS = ("src/effective", "src/agent")
_PUBLIC_PROSE_ESCAPE = "lint: public-prose"
"""Opt out on the offending line, with a reason. The legitimate case is a seam that genuinely
names a foreign dialect it must interoperate with (a vendor's event shape, say). That is a
protocol name, not a claim about where this code runs."""


def check_public_prose_source(source: str, filename: str = "<module>") -> list[Violation]:
    """Refuse our-deployment and single-vendor nouns in the substrate's prose."""
    if not any(r in str(filename).replace("\\", "/") for r in _PUBLIC_PROSE_ROOTS):
        return []
    out: list[Violation] = []
    for line, text in _prose_lines(source):
        if _PUBLIC_PROSE_ESCAPE in text:
            continue
        for noun, kind in [(n, "our deployment") for n in _PRIVATE_OPS_NOUNS] + [
            (n, "a single vendor") for n in _VENDOR_NOUNS
        ]:
            if noun in text:
                out.append(
                    Violation(
                        "public-prose",
                        f"{noun!r} names {kind} in substrate prose. A stranger reading this "
                        "has no such context, and Effective is not one deployment or one vendor. "
                        "State the property that holds for any adopter; put the provenance in "
                        f"the commit message. Opt out with `{_PUBLIC_PROSE_ESCAPE}: <reason>` "
                        "when the name is a foreign PROTOCOL this seam interoperates with.",
                        line,
                        text.strip()[:90],
                        filename,
                    )
                )
                break
    return out


def _prose_lines(source: str) -> Iterator[tuple[int, str]]:
    """Every comment line and every docstring line — the prose surface, not the code."""
    yield from _comment_lines(source)
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return
    holders = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    for node in ast.walk(tree):
        if isinstance(node, holders) and (doc := ast.get_docstring(node, clean=False)):
            base = getattr(node, "lineno", 1)
            for offset, text in enumerate(doc.splitlines()):
                yield base + offset, text


def check_public_prose_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_public_prose_source(path.read_text(), str(path))


# --- the authority-namespace rule (`--authority-tags`) -----------------------------------
#
# NOTE: examples in this file spell the tag in CAPS so the scanner does not match its own source.
# `event_name` keeps author-supplied await names out of the substrate's authority namespaces by
# refusing a reserved set of leading tags. The await axis is where that refusal has to happen:
# a step name composes under its own arm and is disjoint by construction, but an await name is
# delivered as written, and Absurd delivers events by name across the whole queue — so a
# namespace the substrate parks on but has not reserved lets an author's await consume the
# substrate's own approval. A hand-curated set drifts out from under that, which is what makes
# the set DERIVED: every tag declared `AuthorityTag` must be reserved, so a new authority
# namespace cannot be introduced without `event_name` learning to fence it. Separation logic —
# the regions are provably disjoint rather than disjoint-by-inspection. The set itself is
# `keys.RESERVED_AUTHORITY_TAGS`; this rule is what keeps it equal to the declarations.
# A tag that arrives as an interpolation (`t"{FORK_SCOPE}:..."` / `t"{Tag(X)}:..."`) is a runtime
# value, so resolve it the ONE bounded way a simple tool honestly can:
# a same-file module-level string constant, or an inline literal. Regex is the right instrument
# for THIS — a module constant is line-shaped — where it was the wrong one for template holes
# (nested braces / format specs / quotes), which `_hole_expressions` now walks with ast-grep.
# Anything more dynamic is reported as UNREGISTRABLE, never silently skipped: a registry with an
# invisible hole is worse than one that admits the hole.
_MODULE_CONST_RE = re.compile(
    r'^([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=\s*(?:(\w+)\()?["\']([a-z0-9_-]+)["\']', re.MULTILINE
)
"""A module-level constant, capturing (NAME, constructor-or-None, literal). The constructor is
what tells the authority rule whether a namespace DECLARED itself as authority-bearing
(`AuthorityTag("govern")`) — the distinction that a plain "the substrate mints it" test cannot
make once every identity composes through `compose_key`."""


def _module_consts(text: str) -> dict[str, tuple[str | None, str]]:
    return {m.group(1): (m.group(2), m.group(3)) for m in _MODULE_CONST_RE.finditer(text)}


def check_authority_tags(paths: Iterable[str | Path]) -> list[Violation]:
    """`RESERVED_AUTHORITY_TAGS` must equal the set of `AuthorityTag` declarations.

    Checked BOTH ways, which is what makes the set derived rather than curated:

    - a namespace composed from an `AuthorityTag` but NOT reserved → an author-supplied await
      name could park on an approval / grant / park, so `event_name` must fence it;
    - a reserved authority prefix that NO `AuthorityTag` declares → the reserved set has grown a
      stale entry, or the namespace it names is gone and the fence guards nothing.

    Author and domain namespaces (`tool:`, `review:`, `extracted:`, …) are deliberately NOT
    required to be reserved: they are the names authors are *supposed* to own. Every identity
    composes through `compose_key`, so "the substrate mints it" cannot tell the two kinds apart;
    the declaration does (see `AuthorityTag`).

    **It reads through `_skeleton_of`, the grammar's own parser** — one reader of the grammar,
    inside the module whose whole job is to check against it. `_shape_of` survives beside it only
    for `_authority_declarations`, which wants the leading static and not a parse.
    """
    out: list[Violation] = []
    seen: set[str] = set()
    for path in paths:
        path = Path(path)
        text = path.read_text()
        declared = _module_consts(text)
        consts = {name: literal for name, (_ctor, literal) in declared.items()}
        authority = {literal for _n, (ctor, literal) in declared.items() if ctor == "AuthorityTag"}
        seen |= authority
        for line, node in _compose_key_templates(text):
            tag, _skeleton, _fields, _roles, error = _skeleton_of(node, consts)
            if error is not None:
                # The GRAMMAR refused the template — `a:b:{n}` (a term carries one `:`), a hole
                # glued to literal text, an empty coordinate. The parser's own
                # message says which, and says it better than this rule could.
                out.append(Violation("unregistrable-tag", error, line, node.text(), str(path)))
                continue
            if tag is None:
                out.append(
                    Violation(
                        "unregistrable-tag",
                        "`compose_key` takes its namespace from a `Tag(...)` this scanner cannot "
                        "resolve (it reads inline literals and same-file module constants only), "
                        "or from a leading splice, which names no namespace statically — so the "
                        "namespace can be neither registered nor reserved. Use a literal static, "
                        "or a same-file module-level constant",
                        line,
                        node.text(),
                        str(path),
                    )
                )
                continue
            if tag in authority and tag not in RESERVED_AUTHORITY_TAGS:
                out.append(
                    Violation(
                        "unreserved-authority-tag",
                        f"namespace {tag + ':'!r} is declared `AuthorityTag` — its names ARE the "
                        f"authorization — but it is not in `RESERVED_AUTHORITY_TAGS` "
                        f"(handlers/base.py), so an author-supplied await name could park on an "
                        f"approval / grant / park. Reserve it",
                        line,
                        node.text(),
                        str(path),
                    )
                )
    for reserved in RESERVED_AUTHORITY_TAGS:
        if reserved not in seen:
            out.append(
                Violation(
                    "stale-reserved-authority-tag",
                    f"{reserved!r} is reserved in `RESERVED_AUTHORITY_TAGS` but no "
                    f"`AuthorityTag` declares it — the fence guards a namespace nothing mints. "
                    f"Either declare it at its composition site or drop the reservation",
                    0,
                    reserved,
                    "handlers/base.py",
                )
            )
    return out


# --- the KEY REGISTRY / source map (`--key-registry`) ------------------------------------
#
# One pass over the `compose_key(t"...")` literals records each template's SHAPE (namespace tag,
# hole source-expressions, file:line). Two jobs from one artifact: two variants of one tag that
# `separated` cannot tell apart are a collision, and the recorded field NAMES are what let
# `registry.explain` decode a key back into named bindings and point at its producing line.
#
# Static, not runtime: file:line comes for free, production pays nothing, and a collision fails
# the gate. The scanner is deliberately simple: see `_MODULE_CONST_RE` for the one bounded
# resolution it does (an inline literal or a same-file module constant) and the `unregistrable-tag`
# violation it raises rather than silently skipping anything more dynamic.
_TAG_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_ROLES: dict[str, str] = {cls.__name__: cls.role for cls in (Index, Name, Ordinal, Run, Subject)}
"""Marker name -> the role it declares, read off the classes so the scan and the runtime cannot
disagree about what `Run` means."""

_MARKERS = ("Segment", "Key", "Tag", *_ROLES)


def _holes(string_node: Any) -> tuple[tuple[str, str], ...]:
    """Each interpolation as `(source expression, marker)`, via ast-grep's `interpolation` nodes.

    Structure, not a regex: a hole can nest braces, carry a format spec, or contain quoted
    strings, and a brace-matching regex is wrong on all three. (Regex stays fine for line-shaped
    things like the module-constant scan below — this is the case where it is genuinely too weak.)

    The marker constructors are peeled off the expression and the OUTERMOST is returned beside it:
    `Run(task_id)` reads as `("task_id", "Run")`, and `Index(Name(lane))` as `("lane", "Index")`.
    The field is keyed by the author's NAME, so a decode answers in the author's vocabulary, and a
    re-wrapped value leaves no wrapper in the name a reader is shown. The outermost marker is what
    the line declares, which is what a reader sees there.

    **The marker read here is the one written INLINE.** A hole spelled `{r}` after
    `r = Run(task_id)` reads as `("r", "")`, and `_bound_roles` is where that assignment is
    resolved, one level and no further."""
    out: list[tuple[str, str]] = []
    for interpolation in string_node.find_all(kind="interpolation"):
        text = interpolation.text().strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1].strip()
        if (spec := interpolation.find(kind="format_specifier")) is not None:
            text = text.removesuffix(spec.text()).rstrip().removesuffix(":")
        found = ""
        while peeled := next(
            (m for m in _MARKERS if text.startswith(m + "(") and text.endswith(")")), ""
        ):
            text = text[len(peeled) + 1 : -1].strip()
            found = found or peeled  # the OUTERMOST marker is the one the line declares
        out.append((text, found))
    return tuple(out)


def _hole_expressions(string_node: Any) -> tuple[str, ...]:
    return tuple(expression for expression, _marker in _holes(string_node))


def _hole_roles(string_node: Any, bound: dict[str, str] | None = None) -> tuple[str, ...]:
    """The role each hole declares, `""` where none is declared.

    `bound` is `_bound_roles`' one-level resolution, so a hole spelled `{gate}` after
    `gate = Name(self.gate)` declares `name`."""
    bound = bound or {}
    return tuple(
        _ROLES.get(marker) or bound.get(expression, "")
        for expression, marker in _holes(string_node)
    )


def _bound_roles(source: str) -> dict[str, str]:
    """Local name -> the role a marker assigned to it declares, for the whole file.

    **The same bounded resolution `_module_consts` does for a `Tag`.** `govern`'s template binds
    its markers a line early on purpose: a hole's source expression becomes the registry's FIELD
    NAME, so `Name(self.gate)` written inline would register the field as `self.gate`. Reading the
    assignment keeps the field reading `gate` and the role declared.

    What the file says about a name, and what the hole then declares:

    | assignments to the name, file-wide | the hole declares |
    |---|---|
    | `gate = Name(...)` | `name` |
    | `gate, run_id = Name(...), Run(...)` | `name`, `run` |
    | `v = Name(...)` in one function, `v = Run(...)` in another | nothing |
    | `x = role_for(...)`, a helper returning a role | nothing |
    | none | nothing |

    Row three is the ambiguity: the file offers two readings and the ground to choose between them
    is outside this scan, so it declines, exactly as `_shape_of` declines a tag it cannot settle.
    Row four is the depth limit. Both leave the coordinate undeclared, which is a state the
    coverage gate reports."""
    seen: dict[str, str] = {}
    ambiguous: set[str] = set()
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target, value in _assigned_pairs(node):
            match value:
                case ast.Call(func=ast.Name(id=marker)) if marker in _ROLES:
                    role = _ROLES[marker]
                    if seen.setdefault(target, role) != role:
                        ambiguous.add(target)
                case _:
                    pass
    return {name: role for name, role in seen.items() if name not in ambiguous}


def _assigned_pairs(node: ast.Assign) -> Iterator[tuple[str, ast.expr]]:
    """`(name, value)` for a simple assignment and for the tuple form `a, b = X(...), Y(...)`."""
    for target in node.targets:
        match (target, node.value):
            case (ast.Name(id=name), value):
                yield name, value
            case (ast.Tuple(elts=targets), ast.Tuple(elts=values)) if len(targets) == len(values):
                for element, value in zip(targets, values, strict=True):
                    if name := _identifier(element):
                        yield name, value
            case _:
                pass


def _compose_key_templates(source: str) -> Iterator[tuple[int, Any]]:
    """Every `compose_key(t"...")` **call** in `source`, as (line, literal-source-text).

    An AST walk, not a regex over raw text: this module and several docstrings contain
    illustrative `compose_key(t"TAG:...")` snippets, and a text scan cannot tell a call from
    prose: it would register `...` and `TAG:...` as namespaces. ast-grep is already a dependency
    here, so reading actual call nodes costs nothing and removes the whole class."""
    root = SgRoot(source, "python").root()
    for call in root.find_all(kind="call"):
        function = call.field("function")
        # `rsplit(".", 1)[-1]`, matching `_identity_slots`' convention one screen up, so a dotted
        # `keys.compose_key(...)` is seen too. This helper feeds three rules (`--key-registry`,
        # `--key-borrowing`, `--key-literals`), so a blind spot here is shared three ways. An
        # alias (`import compose_key as ck`) still defeats it, which is what an AST rule is
        # bounded by; the store gate is the backstop that is not.
        if function is None or function.text().rsplit(".", 1)[-1] != "compose_key":
            continue
        arguments = call.field("arguments")
        if arguments is None:
            continue
        # A template may be written as ADJACENT t-string literals — legal in 3.14, and the
        # natural way to wrap a long one. tree-sitter gives that group a `concatenated_string`
        # node, so read the group when there is one and fall back to a lone `string` otherwise.
        # Reading only the lone string under-reports a wrapped template's shape (3 fields of
        # `govern`'s 5); ast-grep sees the implicit t-string concatenation, which is the reason to
        # read structure rather than text.
        group = arguments.find(kind="concatenated_string")
        if group is not None and _TSTRING_PREFIX_RE.match(group.text()):
            yield group.range().start.line + 1, group
            continue
        for string_node in arguments.find_all(kind="string"):
            text = string_node.text()
            if _TSTRING_PREFIX_RE.match(text):
                yield string_node.range().start.line + 1, string_node
                break


def _leading_static(string_node: Any) -> str:
    """The static text before the template's first hole — the candidate namespace tag.

    Read from the `string_content` node rather than by indexing past a quote character, which
    mis-parses a single-quoted template as the tag `t\'event2` and a triple-quoted one as
    `""event3`, and drops both with no registration and no violation. `string_content` is
    quote-form independent, so the gate's soundness does not rest on `ruff format` keeping the
    repo double-quoted.

    For adjacent literals only the FIRST segment can carry the tag; a later segment's leading text
    sits after a hole and is therefore interior."""
    node = string_node
    if node.kind() == "concatenated_string":
        first = next(iter(node.find_all(kind="string")), None)
        if first is None:
            return ""
        node = first
    for child in node.children():
        match child.kind():
            case "string_content":
                return child.text()
            case "interpolation":
                return ""  # the template opens with a hole — a `Tag`, or unregistrable
    return ""


def _shape_of(string_node: Any, consts: dict[str, str]) -> tuple[str | None, tuple[str, ...]]:
    """The (tag, field-expressions) of one `compose_key` template.

    The tag is either a leading static (`t"event:{name}"`) or a leading `Tag` interpolation
    (`t"{FORK_SCOPE}:{child}:{id}"`, the wrapper pattern, which keeps a namespace to ONE
    spelling). For the interpolated form this scanner resolves the bounded case a simple tool
    honestly can: an inline literal, or a same-file module-level constant — which covers
    `FORK_SCOPE = Tag("hyp")` whether it is written bare or as `Tag(FORK_SCOPE)`. Anything more
    dynamic returns `None` and is reported as `unregistrable-tag`, never silently skipped."""
    fields = _hole_expressions(string_node)
    head = _leading_static(string_node)
    if head:
        return (head.rstrip(":") or None), fields
    if not fields:
        return None, ()
    lead = fields[0]  # markers already peeled, so `Tag(FORK_SCOPE)` and `FORK_SCOPE` both arrive
    resolved = lead.strip("\"'") if lead[:1] in "\"'" else consts.get(lead)
    return resolved, fields[1:]


_HOLE = "\x00"
"""A sentinel standing in for one interpolation while the template is read as a flat pattern.
Chosen because `compose_key`'s own guards make it unreachable in a real template: a static is the
author's literal text and a NUL there would be a different problem entirely."""


def _resolved_tag(hole: Hole, holes: list[str], consts: dict[str, str]) -> str | None:
    """The tag a `Tag`-valued hole names, for the bounded case a static reader can honestly settle:
    an inline literal, or a same-file module-level constant. `None` means "cannot say", which the
    caller reports rather than guesses at."""
    expression = holes[hole.index] if hole.index < len(holes) else ""
    return expression.strip("\"'") if expression[:1] in "\"'" else consts.get(expression)


def _skeleton_of(
    string_node: Any, consts: dict[str, str], bound: dict[str, str] | None = None
) -> tuple[str | None, Skeleton | None, tuple[str, ...], tuple[str, ...], str | None]:
    """`(tag, skeleton, fields, roles, error)`: the structure of one `compose_key` template.

    `roles` runs parallel to `fields`: the role each coordinate declares, `""` where the author
    declared none.

    **The parse is `grammar.parse_skeleton`**, the reader `compose_key` uses. The registry is a
    source map only if it describes what the composer mints, and a second reader would disagree
    with the composer about where a field ends.

    A COMPOUND static prefix (`t"a:b:{n}"`) is unregistrable because the grammar refuses it: a
    term carries one `:`. A variant is declared visibly, by a literal coordinate
    (`t"a:{x},seg,{n}"`)."""
    parts: list[str | Hole] = []
    _flatten_pattern(string_node, parts)
    holes = list(_hole_expressions(string_node))
    roles = list(_hole_roles(string_node, bound))
    try:
        skeleton = parse_skeleton(parts)
    except KeySyntaxError as exc:
        return None, None, (), (), str(exc)
    lead = skeleton.elements[0]
    if isinstance(lead, Splice):
        # A leading hole STANDING ALONE: `t"{APPROVE};{placed_key(op)}"`. The statics cannot tell
        # a `Tag` naming a zero-arity qualifier from a `Key` contributing terms, and `Splice` says
        # so; the composer resolves it by the value's TYPE (`keys._fill`), and a static reader
        # resolves it by the NAME, the same bounded case the `{TAG}:{x}` branch below handles.
        # Left unresolved, the highest-stakes namespace in the tree would have no registered event
        # shape, and `explain("approve;r1;tool:charge-card")` would name the TOOL producer with
        # `approve` demoted to a frame atom.
        resolved = _resolved_tag(lead.hole, holes, consts)
        if resolved is None:
            return None, None, (), (), None
        lead = SkeletonTerm(resolved)  # a zero-arity qualifier, as `keys._fill` mints at runtime
        skeleton = Skeleton((lead, *skeleton.elements[1:]))
    tag = lead.tag
    if isinstance(tag, Hole):
        # A leading `Tag` interpolation — resolved to its module constant, and CONSUMED, so the
        # remaining holes line up with the fields the skeleton left open.
        resolved = _resolved_tag(tag, holes, consts)
        if resolved is None:
            return None, None, (), (), None
        tag = resolved
        # RESOLVE it INTO the skeleton, so the shape carries the same leading term the composer
        # mints. Left as a `Hole` it renders as an interpolation everywhere the skeleton is walked
        # — `label()`, the borrowing witness, a test's synthetic key — and each of those would
        # count one more hole than the shape has fields.
        skeleton = Skeleton((SkeletonTerm(tag, lead.coordinates), *skeleton.elements[1:]))
    order = [index for index in _hole_order(skeleton) if index < len(holes)]
    fields = tuple(holes[index] for index in order)
    return tag, skeleton, fields, tuple(roles[index] for index in order), None


def _hole_order(skeleton: Skeleton) -> list[int]:
    """Every hole's interpolation index, in the order a decode binds them — which is the order
    they appear, EXCEPT that a leading `Tag` is consumed as the namespace rather than a field."""
    out: list[int] = []
    for position, element in enumerate(skeleton.elements):
        match element:
            case Splice(hole):
                out.append(hole.index)
            case SkeletonTerm(tag, coordinates):
                if isinstance(tag, Hole) and position > 0:
                    out.append(tag.index)
                for coordinate in coordinates:
                    out.extend(a.index for a in coordinate.atoms if isinstance(a, Hole))
            case unreachable:
                assert_never(unreachable)  # pragma: no cover - `ty` proves this arm dead
    return out


def _flatten_pattern(node: Any, out: list[str | Hole]) -> None:
    """Append this template's statics and `_HOLE` markers, IN ORDER, recursing through implicit
    concatenation.

    Adjacent literals are one template to Python and two `string` nodes to the parser
    (`govern.py`'s park name is written across two lines). A non-recursive walk sees no
    `string_content` there and silently registers nothing.

    **The `domain=` directive rides along**, read off the `format_specifier` node — the reason to
    put a splice's admissible domain in the `format_spec` rather than in a table the composer
    consults. One declaration in one template, and the composer and this scanner read it the same
    way, so a registered shape cannot describe a laxer splice than the composer enforces."""
    for child in node.children():
        match child.kind():
            case "string_content":
                out.append(child.text())
            case "interpolation":
                out.append(
                    Hole(sum(1 for o in out if isinstance(o, Hole)), _declared_domain(child))
                )
            case "string":
                _flatten_pattern(child, out)


def _declared_domain(interpolation: Any) -> Domain | None:
    """The `domain=` an interpolation declares, or `None` — the static half of `keys._directives`.

    Silent on anything it does not recognize: a malformed spec is `compose_key`'s to refuse, with
    the value in hand and a message that can name it. A scanner that guessed here would report a
    shape the composer never mints."""
    if (spec := interpolation.find(kind="format_specifier")) is None:
        return None
    directive, _marked, value = spec.text().lstrip(":").partition(COORDINATE_NAME_SEPARATOR)
    if directive != DOMAIN_DIRECTIVE:
        return None
    try:
        return Domain(value)
    except ValueError:
        return None


def _tag_of(head: str, holes: list[str], consts: dict[str, str]) -> str | None:
    """The namespace, from a leading static or a leading `Tag` interpolation (which is CONSUMED
    from `holes`, so the caller's slot walk starts after it)."""
    if head != _HOLE:
        return head or None
    lead = holes.pop(0) if holes else ""
    return lead.strip("\"'") if lead[:1] in "\"'" else consts.get(lead)


def build_key_registry(paths: Iterable[str | Path]) -> tuple[list[Shape], list[Violation]]:
    """`registry_of` over files."""
    return registry_of({str(p): Path(p).read_text() for p in paths})


def registry_of(sources: Mapping[str, str]) -> tuple[list[Shape], list[Violation]]:
    """Scan named SOURCES for `compose_key` templates; return the registered VARIANTS and any
    collisions. The name is what a site is reported as, so a caller with files passes their paths.

    A namespace may own several variants — it is a tagged union — admitted only when `separated`
    proves each new one disjoint from every variant already registered under that tag. That is
    separation logic; a one-arity-per-tag rule is a proxy for it that refuses admissible unions
    (`code:`, `skill:`)."""
    variants: dict[str, list[Shape]] = {}
    problems: list[Violation] = []
    for path, text in sources.items():
        declared = _module_consts(text)
        consts = {name: literal for name, (_ctor, literal) in declared.items()}
        bound = _bound_roles(text)
        for line, node in _compose_key_templates(text):
            tag, skeleton, fields, roles, error = _skeleton_of(node, consts, bound)
            site = f"{path}:{line}"
            if error is not None:
                problems.append(Violation("unslotted-segment", error, line, node.text(), path))
                continue
            if tag is None or skeleton is None or not _TAG_RE.match(tag):
                continue  # unregistrable — `check_authority_tags` reports it; not duplicated here
            fresh = Shape(skeleton=skeleton, fields=fields, roles=roles, site=site, sites=(site,))
            known = variants.setdefault(tag, [])
            # SAME variant, written twice — one protocol at two sites, differing only in the field
            # NAMES its authors chose. Record the extra site; nothing to separate.
            same = next((v for v in known if _same_variant(v, fresh)), None)
            if same is not None:
                merged, conflict = _merged_roles(same, fresh)
                if conflict:
                    problems.append(
                        Violation("coordinate-role-conflict", _ROLE_CONFLICT, line, conflict, path)
                    )
                    continue
                known[known.index(same)] = replace(same, roles=merged, sites=(*same.sites, site))
                continue
            clashes = [(v, why) for v in known for ok, why in (separated(v, fresh),) if not ok]
            if clashes:
                prior, why = clashes[0]
                problems.append(
                    Violation(
                        "tag-variants-not-separated",
                        f"namespace {tag + ':'!r} would own two variants whose languages are not "
                        f"provably disjoint: {prior.label()} at {prior.site} and "
                        f"{fresh.label()} here ({why}). A key could satisfy both, so a decode "
                        f"cannot say which produced it, and an authorization under one name "
                        f"could answer the other. Separate them with a literal at a fixed "
                        f"coordinate (`tag:{{a}},seg,{{b}}` vs `tag:{{a}},action,{{b}}`), or "
                        f"give one its own tag",
                        line,
                        node.text(),
                        path,
                    )
                )
                continue
            known.append(fresh)
    return [v for group in variants.values() for v in group], problems


_ROLE_CONFLICT = (
    "one variant of this namespace is minted at two sites that declare DIFFERENT roles for one "
    "coordinate. A role describes the language rather than the producer, so a projection folding "
    "stored keys of this shape would drop the coordinate for one site's keys and keep it for the "
    "other, from bytes that cannot tell them apart. Agree on the role, or separate the shapes"
)


def _merged_roles(a: Shape, b: Shape) -> tuple[tuple[str, ...], str]:
    """The roles of one variant written at two sites, as `(merged, witness)`.

    A site that declares a role contributes it; a site that declares none inherits, so the tree
    converts one file at a time and still registers the declared answer. Two sites declaring
    DIFFERENT roles for one coordinate is the collision: the shape is what a decode dispatches on,
    so a coordinate cannot mean two things under one template. The witness is empty when they
    agree, and names the coordinate and both readings when they do not."""
    merged: list[str] = []
    for index, (mine, theirs) in enumerate(zip(a.roles, b.roles, strict=False)):
        if mine and theirs and mine != theirs:
            field = b.fields[index] if index < len(b.fields) else str(index)
            return (), " ".join(
                ["{" + field + "}", repr(theirs), "here, but", repr(mine), "at", a.site]
            )
        merged.append(mine or theirs)
    return tuple(merged), ""


def _same_variant(a: Shape, b: Shape) -> bool:
    """One variant written twice: the same SKELETON, allowing different field NAMES.

    Names are documentation attached to the decode; the shape is what a reader counts. This is
    what keeps `budget-grant:{run_id},{trips}` and `budget-grant:{run_id},{i}` one variant with
    two sites rather than a collision. Comparing the skeletons directly is the whole test now —
    the old version hand-rolled a signature over slots and a tail, which is a third reading of
    the same structure."""
    return a.skeleton == b.skeleton


NOT_COORDINATE_ROLE_SCANNED: dict[str, str] = configured("not_coordinate_role_scanned") or {}
"""Path prefixes this rule does not scan, each with the decision that put it there.

A gate's domain is what it scans, so an exclusion is a sentence on the record rather than a quiet
skip. The shipped table is empty: every shipped tree is scanned, and a tree that ships more
declares its exclusions in `[tool.effective.lint]`."""

_UNDECLARED_COORDINATE = (
    "this coordinate declares no role, so a projection reading the composed key back off a tape "
    "has no ground to keep it or drop it. Say what it means at the mint: `Run` identifies an "
    "execution, `Index` counts repetitions of one position, `Ordinal` numbers a distinct "
    "position of its kind, `Name` selects a distinct position, `Subject` carries the domain's "
    "own value"
)


# --- assembled Python source (`--python-codegen`) -------------------------------------------
#
# **Nothing here hard-codes the assembly of Python.** The coding machine does emit Python, and
# that is the product: the source comes from a model and arrives as data, so no template in this
# tree spells a `def`. A metaprogram would be the other legitimate reason, the way `dataclasses`
# builds an `__init__` from a class's fields. This project has none, so the rule holds until one
# arrives with a case to make.
#
# What it leaves is fixtures, where a SOURCE LITERAL is what a scanner wants: a reader reads it as
# Python and `ruff format` checks it, while an assembled one is one indirection from what it
# claims to be and invisible to every tool that reads Python.

_ASSEMBLED_SOURCE = re.compile(
    r"\bdef \w*\s*\(|\bdef \w*$|\bclass \w+\s*(?:\(|:[ \t]*\n)|\bclass \w*$"
    r"|^[ \t]*from [\w.]+ import ",
    re.M,
)
"""A literal that DECLARES something, which is what a generated module always does.

Shaped like syntax, because a keyword alone is ordinary prose: an `"import of {module!r} inside a
function body"` diagnostic and a `"class pattern with no resolvable class name"` refusal both read
as source to a looser pattern, and both are messages. A fragment ENDING at `def ` counts, since
that is where an interpolated name goes and `f"def {name}(op):"` is the shape this rule is for."""

_HAS_LINES = "\n"
"""The second half of the test, and the half that separates a generated module from a message
about one. Source has lines; `` f"no `def {name}` under tests/" `` does not, and it is a
diagnostic in this very file."""

NOT_PYTHON_CODEGEN_SCANNED: dict[str, str] = {
    "tests/test_lint.py": "3 sites to convert to source literals",
    "tests/test_op_key_injectivity.py": "4 sites to convert to source literals",
    "tests/test_exec_tier.py": "2 sites to convert to source literals",
    "tests/test_artifact_addressing.py": "1 site to convert to source literals",
}
NOT_PYTHON_CODEGEN_SCANNED |= configured("not_python_codegen_scanned") or {}
"""Files this rule does not scan, each with what sits behind it.

The exclusion is per FILE, so a new assembly in one of these is not reported. That is the blind
spot it buys. Each entry carries a task rather than a reason, because each is work to do.
`src/`, `examples/` and `scripts/` are at zero and stay there."""

_ASSEMBLED = (
    "this builds Python source by interpolation, and the project generates no Python. A scanner "
    "wants a source LITERAL, which reads as Python and which `ruff format` can check; assembling "
    "one hides the fixture from every tool that reads Python. Parametrize over whole literals "
    "instead of over the pieces"
)


def check_python_codegen(paths: Iterable[str | Path]) -> list[Violation]:
    """Python source may be written, never assembled."""
    out: list[Violation] = []
    for path in _expanded(paths):
        if str(path) in NOT_PYTHON_CODEGEN_SCANNED:
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        # By LINE, because a chained `a + b + c` is one expression and `ast.walk` meets each of
        # its `BinOp`s, which reported the same assembly twice.
        seen: set[int] = set()
        for node in ast.walk(tree):
            match node:
                case ast.JoinedStr() | ast.BinOp(op=ast.Mod() | ast.Add()):
                    pass
                case ast.Call(func=ast.Attribute(attr="format")):
                    pass
                case _:
                    continue
            if node.lineno in seen:
                continue
            literals = [
                inner.value
                for inner in ast.walk(node)
                if isinstance(inner, ast.Constant) and isinstance(inner.value, str)
            ]
            if any(_ASSEMBLED_SOURCE.search(text) for text in literals) and any(
                _HAS_LINES in text for text in literals
            ):
                seen.add(node.lineno)
                out.append(Violation("python-codegen", _ASSEMBLED, node.lineno, "", str(path)))
    return out


def check_coordinate_roles(paths: Iterable[str | Path]) -> list[Violation]:
    """`check_coordinate_roles_source` over files, minus the trees below."""
    return [
        violation
        for path in _expanded(paths)
        if not any(str(path).startswith(prefix) for prefix in NOT_COORDINATE_ROLE_SCANNED)
        for violation in check_coordinate_roles_source(path.read_text(), str(path))
    ]


def check_coordinate_roles_source(source: str, filename: str = "<keys>") -> list[Violation]:
    """Every interior COORDINATE declares what it means.

    A coordinate's role is the half of a key its bytes cannot carry: the text says where the
    coordinates end, and only the declaration says which of them identifies an execution. That is
    what lets a projection over stored keys drop a run id and keep a gate name in one term, so an
    undeclared coordinate is one no projection can rule on.

    **What counts as a coordinate here.** A hole filling an atom inside a term. The namespace tag
    is not one; an integer literal carries no identity to declare; and a SPLICE contributes whole
    terms, so its value is a `Key` carrying roles of its own.

    A hole whose marker this scan cannot see is reported rather than assumed roleless. A role
    bound one line early resolves; anything deeper is a coordinate whose meaning nothing states."""
    out: list[Violation] = []
    consts = {name: literal for name, (_ctor, literal) in _module_consts(source).items()}
    bound = _bound_roles(source)
    for line, node in _compose_key_templates(source):
        _tag, skeleton, fields, roles, error = _skeleton_of(node, consts, bound)
        if skeleton is None or error is not None:
            continue  # unregistrable or unslotted: `--key-registry` owns both
        spliced = {
            element.hole.index for element in skeleton.elements if isinstance(element, Splice)
        }
        order = [index for index in _hole_order(skeleton)]
        for position, (field, role) in enumerate(zip(fields, roles, strict=True)):
            if role or position >= len(order) or order[position] in spliced:
                continue
            if field.lstrip("-").isdigit():
                continue
            out.append(Violation("coordinate-role", _UNDECLARED_COORDINATE, line, field, filename))
    return out


def check_key_registry(paths: Iterable[str | Path]) -> list[Violation]:
    """The gate: no namespace may be owned by two template shapes."""
    _, problems = build_key_registry(paths)
    return problems


# --- borrowed namespaces (`--key-borrowing`) -----------------------------------------------
#
# ASK WHAT THE GATE SCANS. `--key-registry` reads `KEY_REGISTRY_SRCS` and nothing else, so a
# `compose_key(t"fold:{2}")` in `tests/` mints a second variant of a namespace the registry
# describes at one, invisible to the rule that would refuse it one directory over, and a key it
# mints does not decode against its own registry.
#
# **Widening `--key-registry` to scan `tests/` is the wrong fix.** Most of what it would report
# is not a defect:
#
#   - a test may invent its OWN namespaces, and does: `ns:`, `q:`, `x:` are scaffolding used at
#     several shapes ON PURPOSE, in the very file that specifies what separation means;
#   - `t"x:{'p'}{'q'}"` is an *argument* to `pytest.raises`, a bad template composed to prove it
#     is refused. A gate that flags it is flagging the test for testing;
#   - a test that SPECIALIZES a production shape with a literal (`budget-grant:{RUN}:0` against
#     `budget-grant:{run_id},{trip}`) writes the same language, narrower.
#
# And it would put test fixtures in `build/key-registry.json`, which is the SOURCE MAP: a key
# found in production would be decodable to a test's line. The registry describes what production
# mints; that is the artifact's whole job.
#
# So the rule is the containment the sentence above implies: **a test may not compose a tag
# that production registers, at a shape that tag cannot accept.**
# Its own tags are its business.

KEY_REGISTRY_SRCS: tuple[str, ...] = (
    "src/effective",
    "src/agent",
    "src/examples",
    "examples",
)
KEY_REGISTRY_SRCS += tuple(configured("key_registry_srcs") or ())
"""What "production mints" MEANS, in one place, because two rules now depend on the answer.

`--key-registry` builds the source map from these and `--key-borrowing` asks which tags they own.
Read from a constant rather than repeated in two justfile lines: the two rules disagreeing about
the boundary is exactly the drift that would make the second one vacuous."""

_BORROW_WITNESS = "x"
"""The atom substituted for a hole when building a witness key. Any delimiter-free name works."""


def _expanded(paths: Iterable[str | Path]) -> list[Path]:
    """Directories to their `.py` files, files verbatim — `main`'s expansion, reusable from a
    rule that resolves its own default path set instead of receiving one."""
    out: list[Path] = []
    for entry in paths:
        path = Path(entry)
        out.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    return out


def _witness(shape: Shape) -> str:
    """One key the template would mint, with every hole filled by a placeholder atom.

    The check is by LANGUAGE rather than by skeleton, and the difference is the 9 false positives
    named above: `budget-grant:{RUN},0` and `budget-grant:{run_id},{trip}` are different skeletons
    — one holds a literal where the other holds a hole — while the first mints keys the second
    accepts. Comparing structures reports a collision; comparing languages reports the truth."""
    return _render_skeleton(shape.skeleton, lambda: _BORROW_WITNESS)


def check_key_borrowing(
    paths: Iterable[str | Path], registry_paths: Iterable[str | Path] | None = None
) -> list[Violation]:
    """Refuse a template that claims a PRODUCTION namespace at a shape that namespace cannot mint.

    `registry_paths` defaults to `KEY_REGISTRY_SRCS` — the same set `--key-registry` builds from,
    read from one constant so the two rules cannot disagree about what "production" means.

    **An empty `paths` RAISES**, for the reason `_resolve_role_default` gives: a whole-set rule
    that scans nothing reports success, and a gate that reports success for files it never opened
    is worse than no gate. This one is especially prone to it, because its subject (`tests`) is
    not the default any other mode uses."""
    # Expanded here as well as in `main`, which is idempotent on files and is what makes
    # `check_key_borrowing(["tests"])` work from Python. Without it the rule is total only for
    # callers that happen to have run `main`'s expansion first, and a directory raises
    # `IsADirectoryError` from inside a gate — found by this rule's own ratchet test.
    scanned = _expanded(paths)
    if not scanned:
        raise ValueError(
            "check_key_borrowing: no paths to scan. This rule takes its subject explicitly "
            "(`--key-borrowing tests`) because it is the one rule whose domain is NOT `src/`; "
            "with an empty list it would return no violations and read as green."
        )
    registered, _ = build_key_registry(_expanded(registry_paths or KEY_REGISTRY_SRCS))
    variants: dict[str, list[Shape]] = {}
    for shape in registered:
        variants.setdefault(shape.tag, []).append(shape)

    out: list[Violation] = []
    for path in scanned:
        path = Path(path)
        if str(path) in NOT_KEY_BORROWING_SCANNED:
            continue
        text = path.read_text()
        consts = {name: literal for name, (_ctor, literal) in _module_consts(text).items()}
        for line, node in _compose_key_templates(text):
            tag, skeleton, fields, roles, error = _skeleton_of(node, consts)
            if error is not None or tag is None or skeleton is None or tag not in variants:
                continue  # a tag production does not own is this file's own business
            mine = Shape(skeleton=skeleton, fields=fields, roles=roles)
            key = _witness(mine)
            if any(v.decode(key) is not None for v in variants[tag]):
                continue
            shapes = " or ".join(v.label() for v in variants[tag])
            sites = ", ".join(sorted({s for v in variants[tag] for s in v.sites}))
            out.append(
                Violation(
                    "key-borrowing",
                    f"namespace {tag!r} is production's ({shapes}, minted at {sites}), and this "
                    f"template composes {mine.label()} — a key that namespace cannot accept "
                    f"({key!r} decodes against no registered variant). A test may invent its own "
                    f"tags freely; borrowing one production owns puts a second shape under it "
                    f"where `--key-registry` cannot see it. Use the registered shape, or a tag of "
                    f"your own.",
                    line,
                    node.text(),
                    str(path),
                )
            )
    return out


def _string_literals(source: str) -> list[tuple[int, str]]:
    """Every string literal's `(line, content)` — read from the AST, never by regex.

    `string_content` is quote-form independent, which is the same reason `_leading_static` reads
    it: a regex over source has to guess at quoting, and this repo has already paid for that once
    (a single-quoted template mis-parsed as the tag `t\'event2`).

    **F-STRINGS ARE OUT OF DOMAIN, deliberately.** Their `string_content` nodes are FRAGMENTS —
    `f"review:{mid}"` yields `review:`, a term ending in a separator with no coordinates — so
    validating them as whole keys reports every one of them as malformed. Measured before the
    exclusion: 730 violations, 113 of them the single fragment `review:`. An f-string BUILDING an
    identity is a real finding, and it has its own rule (`--key-composition`, scoped by position);
    this one is about literals that claim to BE a key."""
    root = SgRoot(source, "python").root()
    out: list[tuple[int, str]] = []
    for node in root.find_all(kind="string"):
        if node.text()[:2].lower().startswith(("f", "rf", "t")):
            continue
        for content in node.find_all(kind="string_content"):
            out.append((content.range().start.line + 1, content.text()))
    return out


def check_forged_joins(paths: Iterable[str | Path]) -> list[Violation]:
    """A key may not be rebuilt by joining rendered TERMS with the term separator.

    `ParsedKey.render` is not `";".join(t.render() ...)`, and the difference is one term: a
    FOREIGN term that wraps our key joins its payload with `:`, because that is what the vendored
    SDK wrote and a checkpoint name is a lookup. A hand-rolled join emits `$awaitEvent;review:m1`
    where the engine wrote `$awaitEvent:review:m1`: a DIFFERENT key that still parses, so no
    assertion downstream can see the substitution and no round-trip test reddens. The defect is
    invisible to its own output, which is why a gate holds it rather than careful reading.

    **Scoped by POSITION rather than by spelling.** A rule matching a join whose argument contains
    a `.render()` call catches `join(t.render() for t in terms)` and misses the shape that matters:
    terms appended to a list in a loop, then the bare NAME joined. So **any `TERM_SEPARATOR.join`
    is a violation unless it is one of the designated string joins below.** None of them holds a
    `Term` to lose a field from:

    | designated join | rejoins                         |
    |-----------------|---------------------------------|
    | a text split    | what it split                   |
    | a template      | its skeleton parts              |
    | a projection    | terms it split as text          |

    **What it does NOT scan, because a gate's domain is its grammar.** An f-string that interleaves
    a rendered term with the separator by hand (`f"{head.render()}{TERM_SEPARATOR}"`, live in
    `counterfactual.py` and `handlers/absurd.py`) is the same shape and is invisible here. Both
    are safe by CONSTRUCTION: each renders the first term of a key our own composer minted, which
    can never be a wrapping foreign term. The same spelling over a term
    SEQUENCE would re-create the defect, and nothing would see it."""
    out: list[Violation] = []
    for path in _expanded(paths):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            match node:
                case ast.Call(
                    func=ast.Attribute(value=ast.Name(id="TERM_SEPARATOR"), attr="join")
                ) if (str(path), _enclosing_def(tree, node)) not in DESIGNATED_JOINS:
                    out.append(
                        Violation(
                            "forged-join",
                            "a key rebuilt by joining rendered terms with the term separator "
                            "loses a FOREIGN term's `:` join, emitting bytes no producer wrote "
                            "— a different key that still parses. Build a `ParsedKey` and call "
                            "`.render()`, which is the one renderer that knows the difference.",
                            node.lineno,
                            ast.unparse(node)[:80],
                            str(path),
                        )
                    )
    return out


DESIGNATED_JOINS: frozenset[tuple[str, str]] = frozenset(
    {
        # rejoins what a TEXT split produced; off-language input has no terms to lose
        ("src/effective/keys/frame.py", "_past_frames_text"),
        # joins a SKELETON's rendered parts — a template with its holes shown, not a key
        ("src/effective/keys/registry.py", "_render_skeleton"),
        # rejoins a PROJECTION's parts, at both of its joins: one re-forms the tail it is about
        # to hand to `_peel`, the other the walk's own output. Neither can forge a separator,
        # because the `.join` sees only OUR terms: a wrapping foreign head is peeled first
        # (`_peel`) and its bytes are sliced from the source with its separator, so a vendor's `:`
        # never reaches this join. `test_a_frame_survives_a_payload_no_variant_claims` pins that.
        # A projection's output often still parses (54 of the 82 banked names, at `drop=(Index,)`),
        # so "it is not a key" is not the reason and never was.
        ("src/effective/keys/registry.py", "_project_walk"),
    }
)
"""The joins that hold strings rather than terms. Function-scoped, so a second join added to one
of these files is a violation rather than an inherited exemption."""


def _enclosing_def(tree: ast.AST, node: ast.AST) -> str:
    """The name of the function `node` sits in, or `""` at module level."""
    for candidate in ast.walk(tree):
        if isinstance(candidate, ast.FunctionDef | ast.AsyncFunctionDef):
            end = getattr(candidate, "end_lineno", None) or candidate.lineno
            if candidate.lineno <= getattr(node, "lineno", 0) <= end:
                return candidate.name
    return ""


def check_key_literals(
    paths: Iterable[str | Path], registry_paths: Iterable[str | Path] | None = None
) -> list[Violation]:
    """A key-shaped STRING LITERAL must be one production could actually mint.

    `--key-borrowing`'s sibling, and its complement: that rule scans `compose_key` TEMPLATES, so a
    literal that never composes is invisible to it, and a bare literal in an assertion is exactly
    that.

    **`just lint` runs it over `src` only, and the asymmetry is a measurement rather than a
    preference.** `src` reads 0; `tests/` reads 111, most of them frame+payload composites this
    rule does not model, so gating them would be a backlog wearing a gate's clothes. Say "tests
    unscanned", never "tests clean". The two reserved CallTool names are kebab (`code-execute`,
    `skill-disclose`), so they do not resemble keys.

    Scoped by **production ownership**: a literal whose
    leading tag is one the registry owns is making a claim about a production namespace; a literal
    under a tag production does not own is the test's own business and is never read.

    Three questions, in the order a defect gets cheaper to explain:

    1. is it in the LANGUAGE at all (`approve;front-door:a:b` is not);
    2. does it DECODE against a registered variant of that tag;
    3. is it REACHABLE: an arm's payload is an ADDRESS, so `event;step:q` is well-formed,
       decodes, and is unmintable.

    **What it does NOT scan, said plainly because a gate's domain is its grammar.** A literal that
    ENDS in a separator is a prefix (`startswith("tool:")`, a frame under construction), and no
    key can end in one, so the exclusion is structural rather than a guess. A literal containing
    `%` is a SQL LIKE pattern or a %-format, and `%` is in no atom's charset.

    **And what it still does NOT catch.** A splice is typed `Key`, so a payload that is merely
    the WRONG SHAPE for its position still decodes: `hyp:r-fork;reviewed;m1` (three terms where
    `fork_scoped` mints two). The splice's `domain=` closes the address-vs-identity half of that;
    the remaining half wants a `domain=minted`, which only a rule holding the registry (this one)
    can evaluate, and which is not built."""
    scanned = _expanded(paths)
    if not scanned:
        raise ValueError(
            "check_key_literals: no paths to scan. Like `--key-borrowing`, this rule takes its "
            "subject explicitly because its domain is NOT `src/`; empty, it would read as green."
        )
    registered, _ = build_key_registry(_expanded(registry_paths or KEY_REGISTRY_SRCS))
    variants: dict[str, list[Shape]] = {}
    for shape in registered:
        variants.setdefault(shape.tag, []).append(shape)
    # The SAME reader an operator gets. A literal may be FRAMED (`rec:0;step;tool:a` — a `scoped`
    # frame around a real op key), and a frame is not a variant of `rec:`, so asking one shape at
    # a time reports legitimate keys as unmintable. `explain` peels frames before it dispatches,
    # and a rule that answered the same question a different way would be a second reader of the
    # grammar.
    registry = KeyMap.from_shapes(registered)

    out: list[Violation] = []
    for path in scanned:
        path = Path(path)
        text = path.read_text()
        for line, literal in _string_literals(text):
            if TAG_SEPARATOR not in literal and TERM_SEPARATOR not in literal:
                continue  # a bare word is not a key claim, whatever it is spelled like
            if literal.endswith((TAG_SEPARATOR, TERM_SEPARATOR, ARITY_SEPARATOR)):
                continue  # a PREFIX (`startswith("tool:")`, a frame being built), not a key
            if "%" in literal:
                continue  # a SQL LIKE pattern or a %-format — `%` is in no atom's charset
            owner = literal.split(TAG_SEPARATOR)[0].split(TERM_SEPARATOR)[0]
            if owner not in variants:
                continue  # a tag production does not own is this file's own business
            problem = _unmintable(literal, variants[owner], registry)
            if problem is None:
                continue
            out.append(
                Violation(
                    "key-literal",
                    f"{literal!r} claims the production namespace {owner!r}, and {problem}. A "
                    f"literal under a tag production owns is a claim about what the substrate "
                    f"mints; ask the minter for it instead of spelling it, or use a tag of your "
                    f"own.",
                    line,
                    literal,
                    str(path),
                )
            )
    return out


def _unmintable(literal: str, shapes: list[Shape], registry: KeyMap) -> str | None:
    """Why `literal` could not have been minted, or `None` if it could.

    Two questions, and this module owns neither reader. **Is it in the language and is its arm
    structure reachable** is `grammar.unmintable`. **Does some production account for it** is
    `KeyMap.explain`, which is what an operator holding the key would run — and which unframes
    first, so a `scoped` or gather frame around a real op key answers the way it should.

    `shapes` is kept for the MESSAGE only: naming the variants of the leading tag is what makes a
    violation actionable, and `explain` reports the whole map."""
    if (why := unmintable(literal)) is not None:
        return why
    try:
        registry.explain(literal)
    except UnknownTag, ValueError:
        return (
            f"no registered production accounts for it "
            f"({' or '.join(shape.label() for shape in shapes)})"
        )
    return None


NOT_KEY_BORROWING_SCANNED: dict[str, str] = {}
"""Files excluded from `--key-borrowing`, each with a reason rather than a bare entry — the same
contract as `NOT_TERMINAL_HOLE_SCANNED`.

**Empty, and that is the measurement rather than an oversight.** The rule keyed on production
ownership rather than on f-string-ness or on shape, which is what let the exclusion list stay
empty: the three files that would have needed an entry under the naive rule
(`test_op_key_injectivity.py`'s deliberate `ns:`/`x:` collisions, `test_combinators.py`'s `q:`)
are all composing tags production does not own, so the rule never reaches them."""


# --- the terminal-hole ratchet (`--terminal-holes`) ----------------------------------------
#
# A *marker discipline*: a static nudge that `tests/` say what kind of value lands at a template's
# terminal hole. The safety property is the composer's: a value must be a well-formed atom in
# EVERY position, and `Atom.of` refuses it otherwise. A green here is not the injectivity
# guarantee, which is the grammar's.
#
# **What it scans.** `compose_key(t"…")` calls whose template's LAST child is an `interpolation`,
# which is narrower than "the last hole": `t"skill:{name},activate"` ends in a static and is not
# scanned. It cannot see types, so it makes exactly two structural exceptions (a marker-wrapped
# hole, an integer literal) and asks for a stated reason for the rest.
_TERMINAL_ESCAPE = "lint: terminal-hole"

NOT_TERMINAL_HOLE_SCANNED: dict[str, str] = {
    "tests/test_op_key_injectivity.py": "the file that SPECIFIES what the composer refuses, so a "
    "bare terminal here is the subject rather than a straggler. Most of its unwrapped terminals "
    "are refusal fixtures — `t\"ns:{'a:b'}\"`, `{'a;b'}`, `{'x;y'}` — which CANNOT be wrapped, "
    "because a `Segment` refuses them one layer earlier and the test would then be asserting the "
    "wrong refusal. Per-site escapes would restate the module docstring at every site in the one "
    "file where it is already written down",
}
"""Files excluded from `--terminal-holes`, each with a reason rather than a bare entry: the same
contract as `NOT_WORKFLOW_ROLE`, and for the same purpose, a decision on the record.

**Say "13 unscanned", never "clean".** Measured 2026-10-04 by running the rule on the file:
`check_terminal_holes_source(Path("tests/test_op_key_injectivity.py").read_text())` reports
**13**.

That is the blind spot the exclusion buys, and it is the whole file: a regression added there
would not be reported. Some of them are delimiter-bearing refusal fixtures that cannot be wrapped
(a `Segment` refuses them one layer earlier, so wrapping would assert the wrong refusal); the rest
could take a per-site `# lint: terminal-hole` escape.
The trade is small, because the composer refuses a non-atom in every position at runtime
regardless."""


def _terminal_interpolation(string_node: Any) -> Any | None:
    """The template's last child if that child is a hole, else `None`.

    Read from the node's children rather than by matching a trailing `}` in the text: a template
    may end with a static that itself contains braces, and adjacent literals put the final child
    in the LAST segment of a `concatenated_string`."""
    node = string_node
    if node.kind() == "concatenated_string":
        node = next(reversed(list(node.find_all(kind="string"))), None)
        if node is None:
            return None
    last = None
    for child in node.children():
        if child.kind() in ("string_content", "interpolation"):
            last = child
    return last if last is not None and last.kind() == "interpolation" else None


def check_terminal_holes_source(source: str, filename: str = "<keys>") -> list[Violation]:
    return list(_terminal_hole_violations(source, filename))


def _terminal_hole_violations(source: str, filename: str) -> Iterator[Violation]:
    lines = source.splitlines()
    for line, string_node in _compose_key_templates(source):
        interpolation = _terminal_interpolation(string_node)
        if interpolation is None:
            continue
        text = interpolation.text().strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1].strip()
        if (spec := interpolation.find(kind="format_specifier")) is not None:
            text = text.removesuffix(spec.text()).rstrip().removesuffix(":")
        if any(text.startswith(m + "(") and text.endswith(")") for m in _MARKERS):
            continue  # the marker validated at construction — the whole point
        if text.lstrip("-").isdigit():
            continue  # an int cannot carry a delimiter, so it needs no wrapper anywhere
        window = (lines[i] for i in _escape_window(lines, line) if i < len(lines))
        if any(_TERMINAL_ESCAPE in text_line for text_line in window):
            continue
        yield Violation(
            "terminal-hole",
            f"the terminal hole {{{text}}} is unwrapped, and this rule cannot see its type. "
            f"Wrap it: Segment({text}). If it is already an `int` or a `Key`, it needs no "
            f"wrapper; say so with a `# {_TERMINAL_ESCAPE}` comment above. (The composer "
            f"refuses a non-atom in EVERY position, so this is a marker discipline rather than "
            f"the safety property.)",
            line,
            string_node.text().splitlines()[0][:90],
            filename,
        )


def check_terminal_holes_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    if str(path) in NOT_TERMINAL_HOLE_SCANNED:
        return []
    return check_terminal_holes_source(path.read_text(), filename=str(path))


# --- the authority-SCOPE rule (`--authority-scopes`) --------------------------------------
#
# `--authority-tags` above asks *is this namespace fenced?*; this asks whether a namespace's
# DECLARED reach is one the substrate can honor. A `Scope.SETTLEMENT` declaration is a request
# of the walk: `handlers.base.placing` applies `Key.occurrence` to every SETTLEMENT await, in
# every walk, pinned by `test_the_walk_coordinates_EVERY_settlement_await` and its ACCRUAL twin
# in `test_op_key_injectivity.py`. What the walk cannot discharge:
#
# | declaration                                          | verdict                               |
# |------------------------------------------------------|---------------------------------------|
# | unreadable here (a tag behind a name, a conditional) | reported, never skipped               |
# | no `scope`                                           | names the missing axis                |
# | `QUALIFIED`, wrapping a `str`                        | refused: it launders a hand-rolled    |
# |                                                      | name into the wrapping namespace      |
#
# The blind spot: a tag DECLARED in one module and COMPOSED in another is not resolved, since
# `_shape_of` reads same-file module constants only. Every authority tag in this tree is declared
# and composed in one module, so a green covers same-module compositions.
def _authority_declarations(source: str) -> Iterator[tuple[int, str | None, str | None]]:
    """Every ``AuthorityTag(...)`` declaration, as ``(line, tag, scope-member-or-None)``.

    ast-grep rather than `_MODULE_CONST_RE`: that regex is shared with the registry's tag
    resolution and is line-shaped by design, where a declaration now carries a keyword whose value
    is a dotted name. Reading the call node also means a declaration that wraps across lines is
    seen, the same fragility class `_compose_key_templates` guards against."""
    root = SgRoot(source, "python").root()
    for call in root.find_all(kind="call"):
        function = call.field("function")
        if function is None or function.text() != "AuthorityTag":
            continue
        arguments = call.field("arguments")
        if arguments is None:
            continue
        line = call.range().start.line + 1
        tag = next(
            (node.text().strip("\"'") for node in arguments.children() if node.kind() == "string"),
            None,
        )
        if tag is None:
            # An inline literal or nothing. A tag behind a name (`AuthorityTag(NAME, …)`) or a
            # keyword (`value="t"`) is UNRESOLVED, and unresolved is reported, never skipped:
            # the contract `_shape_of` keeps one section up.
            yield line, None, None
            continue
        keyword = next(
            (
                kw
                for kw in arguments.children()
                if kw.kind() == "keyword_argument"
                and (name := kw.field("name")) is not None
                and name.text() == "scope"
            ),
            None,
        )
        if keyword is None:
            yield line, tag, None
            continue
        value = keyword.field("value")
        # `Scope.X` and nothing else. `rsplit('.', 1)[-1]` over an arbitrary expression read the
        # last member of a CONDITIONAL — so `Scope.SETTLEMENT if x else Scope.ACCRUAL` resolved to
        # the PERMISSIVE value and the namespace went silently unchecked. An attribute node on
        # `Scope` is the only shape this can honestly resolve; anything else is `?`.
        resolved = (
            value.text().split(".", 1)[1]
            if value is not None
            and value.kind() == "attribute"
            and value.text().startswith("Scope.")
            else "?"
        )
        yield line, tag, resolved


def _enclosing_annotations(node: Any) -> dict[str, str]:
    """Parameter annotations visible at `node` — every enclosing `def`, innermost first.

    Walks OUT rather than stopping at the innermost function, because a composition inside a
    closure reads its outer function's parameters and stopping early called those unannotated.
    Reads `default_parameter` as well as `typed_parameter`: a defaulted annotated parameter is a
    different tree-sitter kind, and matching only the first spelled `name: Key` and rejected
    `name: Key = None`."""
    annotations: dict[str, str] = {}
    enclosing = node
    while (enclosing := enclosing.parent()) is not None:
        if enclosing.kind() != "function_definition":
            continue
        parameters = enclosing.field("parameters")
        for parameter in parameters.children() if parameters is not None else ():
            if parameter.kind() not in ("typed_parameter", "typed_default_parameter"):
                continue
            name = next(iter(parameter.children()), None)
            annotation = parameter.field("type")
            if name is not None and annotation is not None:
                annotations.setdefault(name.text(), annotation.text())
    return annotations


def _marker_wrapped_holes(string_node: Any) -> set[str]:
    """The field names whose hole is written `Segment(x)` / `Key(x)` / `Tag(x)`.

    `_hole_expressions` peels the marker off, because a decoded key should be keyed by the field's
    NAME. Here the wrapper is exactly what matters — it validates at construction, so the hole is
    safe whatever the underlying parameter is annotated — so the raw text is re-read rather than
    inferred from the peeled name."""
    out: set[str] = set()
    for interpolation in string_node.find_all(kind="interpolation"):
        text = interpolation.text().strip()
        if text.startswith("{") and text.endswith("}"):
            text = text[1:-1].strip()
        for marker in _MARKERS:
            if text.startswith(marker + "(") and text.endswith(")"):
                out.add(text[len(marker) + 1 : -1].strip())
                break
    return out


_KEY_ANNOTATIONS = ("Key", "keys.Key", '"Key"', "'Key'")
"""How `Key` may be spelled in an annotation. A qualified or stringised form is legal Python and
was being rejected; the alternative — resolving names — is not this scanner's job."""


def _wrapped_hole_is_a_key(source: str, tag: str, consts: dict[str, str]) -> bool | None:
    """Does every ``compose_key`` under `tag` keep author text out of its holes?

    ``None`` when this file composes nothing under `tag` — "no opinion".

    **Over every hole.** The obligation is a property of the template, and a `Key` is legal in
    an interior hole, spliced as its own terms. So at least one hole resolves to a `Key`, and
    none resolves to a bare `str`.

    A hole that is not a parameter of an enclosing `def` — an attribute (`self.name`, which is the
    shape `govern:`'s own reference implementation uses), a call, a module-scope local — is
    UNRESOLVED and returns no verdict rather than a false one. That is a real hole in the check
    and it is the reason the message says "annotate the wrapped parameter" rather than claiming
    proof.

    A hole written `Segment(x)` is safe whatever `x` is annotated: the marker validates at
    construction, which is the whole point of the marker family. So the raw hole text is read
    alongside the peeled field name — reading only the peeled name reddened `fork:`, whose
    `child_run_id` is `Segment`-wrapped in the template and `str` at the parameter."""
    verdicts: list[bool] = []
    for _line, node in _compose_key_templates(source):
        shape_tag, fields = _shape_of(node, consts)
        if shape_tag != tag or not fields:
            continue
        annotations = _enclosing_annotations(node)
        wrapped = _marker_wrapped_holes(node)
        resolved = {
            field: annotations[field]
            for field in fields
            if field in annotations and field not in wrapped
        }
        if not resolved:
            continue  # nothing this scanner can read — no verdict, not a green and not a red
        verdicts.append(
            any(a in _KEY_ANNOTATIONS for a in resolved.values())
            and not any(a == "str" for a in resolved.values())
        )
    return all(verdicts) if verdicts else None


def check_authority_scopes(paths: Iterable[str | Path]) -> list[Violation]:
    """Every `AuthorityTag` must declare a `Scope`, and a declared scope must hold.

    - **`SETTLEMENT`** — nothing to check HERE, and that is a change rather than an omission. The
      coordinate is applied by the walk for every such namespace (`handlers.base.placing`), so the
      declaration is a request the substrate services rather than a promise the namespace keeps.
      See the section comment for what was removed and why.
    - **`QUALIFIED`** — the namespace wraps another composed key, so the occurrence question
      recurses. Checkable only if the wrapped hole is a `Key`; a `str` there launders a
      hand-rolled name into the wrapping namespace.
    - **`ACCRUAL`** — nothing to check. That is deliberate: it is the *named* opt-out, and being
      named at the declaration site is the entire difference between it and a comment.

    Read the section comment above for what this does NOT prove."""
    out: list[Violation] = []
    sources = {Path(p): Path(p).read_text() for p in paths}
    for path, text in sources.items():
        for line, tag, scope in _authority_declarations(text):
            if tag is None or scope == "?":
                out.append(
                    Violation(
                        "unresolvable-authority-declaration",
                        "this `AuthorityTag(...)` declaration cannot be read statically, so its "
                        "namespace is neither checked nor visibly unchecked. Write the tag as an "
                        "inline string literal and the scope as a plain `Scope.MEMBER`; a name, "
                        "a keyword-passed tag, or a conditional scope all resolve to nothing "
                        "here",
                        line,
                        "AuthorityTag(...)",
                        str(path),
                    )
                )
                continue
            if scope is None:
                # Unreachable through a `ty`-clean tree — `scope` is a required keyword — so this
                # covers the untyped consumer and says which axis is missing rather than echoing
                # a TypeError.
                out.append(
                    Violation(
                        "undeclared-authority-scope",
                        f"`AuthorityTag({tag!r})` declares no `scope`. State how far one answer "
                        f"in this namespace reaches: `Scope.SETTLEMENT` (one op-occurrence), "
                        f"`Scope.ACCRUAL` (a whole run/generation, on purpose) or "
                        f"`Scope.QUALIFIED` (it wraps another key)",
                        line,
                        f"AuthorityTag({tag!r})",
                        str(path),
                    )
                )
                continue
            if scope == "QUALIFIED":
                verdicts = [
                    v
                    for source in sources.values()
                    for v in (_wrapped_hole_is_a_key(source, tag, _consts_of(source)),)
                    if v is not None
                ]
                if verdicts and not all(verdicts):
                    out.append(
                        Violation(
                            "qualified-wraps-an-untyped-name",
                            f"namespace {tag + ':'!r} is declared `Scope.QUALIFIED`, which says "
                            f"the occurrence question RECURSES to the key it wraps — but its "
                            f"wrapped hole is not typed `Key`, so there is nothing to recurse "
                            f"into and a hand-rolled name can be laundered into this namespace. "
                            f"Annotate the wrapped parameter `Key` (as `hyp:`'s composer does)",
                            line,
                            f"AuthorityTag({tag!r}, scope=Scope.QUALIFIED)",
                            str(path),
                        )
                    )
    return out


def _consts_of(source: str) -> dict[str, str]:
    return {name: literal for name, (_ctor, literal) in _module_consts(source).items()}


def check_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_source(path.read_text(), filename=str(path))


def check_layer_file(path: str | Path) -> list[Violation]:
    path = Path(path)
    return check_layer_source(path.read_text(), filename=str(path))


def _resolve_role_default(role_set: tuple[str, ...]) -> list[str]:
    """Turn a role set into a concrete default file list. Most entries are real
    repo-relative paths (``src/...``) used verbatim. A few are **suffix-only** — a
    workspace-member file whose source lives under ``examples/<member>/src/<pkg>/...``
    but whose role-set entry is the short package suffix (``<pkg>/workflow.py``) so
    ``_in_role_set`` matches BOTH that real path and the tests' virtual
    ``examples/<pkg>/workflow.py`` paths. Resolve such an
    entry to its real member location so ``just lint`` still lints the file.

    **An entry that resolves to nothing RAISES.** Contributing zero files in silence is the shape
    where a gate reports success for a file it never opened: rename or delete the target and the
    determinism boundary stops being checked there, with `just lint` still green. The list is the
    source of truth for two readers (this CLI and an agent's lint gate), so an entry that names
    nothing is a broken gate."""
    out: list[str] = []
    for entry in role_set:
        if Path(entry).exists():
            out.append(entry)
            continue
        members = sorted(Path("examples").glob(f"*/src/{entry}"))
        if not members:
            raise FileNotFoundError(
                f"role-set entry {entry!r} resolves to no file: it is neither a repo-relative "
                f"path nor a workspace-member suffix under `examples/*/src/`. The gate would "
                f"otherwise skip it silently. Fix the path, or drop the entry."
            )
        out.extend(str(m) for m in members)
    return out


# --- the totality census (`--totality`) -----------------------------------------------------
#
# A CENSUS first and a gate second, and the order is the point. A totality sweep needs to know
# which `isinstance` sites are dispatch over a closed union — and the obvious instrument, ranking
# files by `grep -c isinstance`, measures the wrong thing: `lint.py` tops that ranking at 18 and
# nearly every one of its sites is `isinstance(node, ast.Call)`, a third-party hierarchy with no
# declared union, where `assert_never` cannot be written at all.
#
# So this classifies rather than counts:
#
#   dispatch                 the tested TYPE is a member of a declared union   -> convert
#   duplicated-enumeration   N sites deriving one index space by one filter    -> build it once
#   filter                   one arm SELECTED in a comprehension's `if`        -> leave
#   guard                    `not isinstance(...)` over a raise                -> leave
#   foreign                  no declared union to close (`ast.*`, `dict`, …)   -> out of reach
#
# **Two limits, stated because a gate is bounded by what it SCANS.**
#
#   - `filter` and `duplicated-enumeration` are not separable syntactically. Filters form a
#     duplicated enumeration when each derives an index space another site indexes into, which is
#     semantic. This rule promotes on TEXTUAL identity (two or more `ast.unparse`-identical
#     predicates in one file), so it misses the same defect written with different variable
#     names. A `filter` verdict means "not promoted"; it does not mean "checked and cleared".
#   - Union membership is read from `type X = A | B` aliases and from union ANNOTATIONS, both
#     repo-wide. An `X | None` optional is not a union for this purpose — dispatching on a
#     two-arm optional is an `if x is None`, not a totality question.
_TOTALITY_CATEGORIES = (
    "selection",  # one arm of N chosen; the others do nothing
    "guard",  # a boundary check that REJECTS — it raises
    "filter",  # a comprehension or loop filter
    "coercion",  # the `else` is a constructor, not an arm
    "foreign",  # a third-party hierarchy with no union to declare
    "blocked",  # the subject is `Any` and the fix is elsewhere
    "total",  # already exhaustive, spelled some other way
    "deferred",  # a SCHEDULE, not a classification — must name what unblocks it
)
"""The only categories an escape may claim.

An escape that takes FREE PROSE validates nothing: `"churn"` becomes a category and `"excluded
while the schema is revised"`, a schedule, passes as a classification.

**What the checker proves:**

    An escape must name one token from the fixed vocabulary. The checker validates the token and
    the presence of a reason; it does not prove that the category describes the site. `blocked` is
    reserved for an `Any` subject whose type must be repaired elsewhere. `deferred` records a
    schedule and must name an objective condition that retires it. Every category remains a review
    claim until a category-specific check exists.

The fixed set has no category for *I would rather not*, so the only way out of a real dispatch is
to convert it. A recognized token still silences the gate whether or not it is true."""

_TOTALITY_ESCAPE = "lint: totality"
"""Opt out with a CATEGORY and a reason — a site read and ruled not-work.

Necessary because `dispatch` is a candidate list, not a work list: deciding whether an
`isinstance` is a total dispatch needs the subject's type, and every cheap proxy leaks. Without a
way to record "read this, it is a selection", every pass re-derives the same 15 judgments and the
baseline stops distinguishing *unexamined* from *examined and declined*.

Same shape as `lint: key-composition`, including the contiguous-comment window, so a reason may be
as long as it needs to be."""

_TOTALITY_IN_SCOPE = (
    "open-match",
    "dispatch",
    "duplicated-enumeration",
    "unclosed-match",
    "unannotated-match",
)
"""The populations a totality sweep converts. The other three are reported and not flagged."""


@dataclass(frozen=True)
class TotalitySite:
    """One `isinstance` call, classified. The census row."""

    filename: str
    line: int
    text: str
    population: str
    tested: str
    types: tuple[str, ...]


def _union_attributes(tree: ast.AST) -> set[str]:
    """Attribute names carrying a DECLARED, non-`Any` annotation — `SkeletonTerm.tag: str | Hole`,
    and equally `self._phase: Phase`, where the union hides behind an alias.

    Separate from `_union_members` because an attribute access is where precision is won and lost.
    `element.tag` is dispatch: the class says the attribute is `str | Hole`, so a `match` over it
    can be closed. `interpolation.value` is not: the stdlib types it `Any`, and `assert_never` on
    an `Any` fires unconditionally (plan item 3a-bis) — the check would be noise, not a proof.
    Both read as `<name>.<attr>` and only the declaration tells them apart."""
    named = {
        target.id
        if isinstance(target := node.target, ast.Name)
        else target.attr
        if isinstance(target, ast.Attribute)
        else ""
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and not _is_any(node.annotation)
    }
    return named - {""}


def _declared_returns(tree: ast.AST) -> set[str]:
    """Functions whose RETURN type is written and is not `Any`.

    `match inspect_only(probe, grants):` is not an unannotated subject — the callee says
    `-> Inspection`, and `ty` reads it. Treating every call as unknowable put 25 sites in the
    `unannotated-match` bucket of which most needed no annotation at all; the rule was looking
    only at names and attributes, so the commonest subject shape in this tree fell through."""
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.returns is not None
        and not _is_any(node.returns)
    }


def _union_members(tree: ast.AST) -> set[frozenset[str]]:
    """Every multi-arm union in `tree`, as its SET of arm names.

    Both sources are needed and neither subsumes the other. `SkeletonElement = SkeletonTerm |
    Splice` is the declared form the design writes down; `items: list[str | Interpolation]` is the
    form PEP 750's own pair takes here, which has no alias and is dispatched on four times.

    **Sets and not per-member arities**, which was the first shape and was wrong. Keeping each
    member's smallest arity asks "what is the tightest union this TYPE belongs to", and a decision
    dispatches one union, not several: `match op: case Step(…) | case Scoped(…)` named two arms
    whose minima were 10 and 3, took the 3, and read as covering a three-arm union it was not
    dispatching at all. The right question is which union CONTAINS every arm the decision names,
    and the smallest of those is the bar it has to clear."""
    unions: set[frozenset[str]] = set()
    for node in ast.walk(tree):
        match node:
            case (
                ast.TypeAlias(value=ast.expr() as annotation)
                | ast.AnnAssign(annotation=ast.expr() as annotation)
                | ast.arg(annotation=ast.expr() as annotation)
                | ast.FunctionDef(returns=ast.expr() as annotation)
            ):
                unions |= {frozenset(arms) for arms in _union_arm_sets(annotation)}
    return unions


def _all_arms(unions: set[frozenset[str]]) -> set[str]:
    """Every name that appears as an arm of any declared union."""
    return {name for union in unions for name in union}


def _covering_arity(named: set[str], unions: set[frozenset[str]]) -> int | None:
    """The size of the smallest declared union containing every arm in `named`, or `None`."""
    containing = [u for u in unions if named <= u]
    return min((len(u) for u in containing), default=None)


def _flatten_union(node: ast.expr) -> list[ast.expr]:
    """The arms of a `|` chain, left-nested, as expressions."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _flatten_union(node.left) + _flatten_union(node.right)
    return [node]


def _arm_name(arm: ast.expr) -> str | None:
    """One arm's type name — `Step[Any]` is `Step`, `None` is not an arm at all."""
    match arm:
        case ast.Name(id="None"):
            return None
        case ast.Name(id=name):
            return name
        case ast.Subscript(value=ast.Name(id=name)):
            return name
        case ast.Attribute(attr=name):
            return name
        case _:
            return None


def _union_arm_sets(node: ast.expr) -> list[set[str]]:
    """Each multi-arm union inside `node`, as its set of arm names, `None` excluded.

    Recurses into subscripts so `list[str | Interpolation]` counts — the union that matters is
    rarely at the top of the annotation. A union left with one arm after dropping `None` is an
    OPTIONAL and contributes nothing: `Key | None` is answered by `is None`, not by a `match`.
    """
    found: list[set[str]] = []
    nested = {
        side
        for inner in ast.walk(node)
        if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.BitOr)
        for side in (inner.left, inner.right)
        if isinstance(side, ast.BinOp) and isinstance(side.op, ast.BitOr)
    }
    for inner in ast.walk(node):
        if not isinstance(inner, ast.BinOp) or not isinstance(inner.op, ast.BitOr):
            continue
        # Only MAXIMAL unions. `A | B | C | D` parses left-nested, so walking every `BinOp` also
        # yields `A | B` and `A | B | C` as unions in their own right — and since the arity kept
        # is the smallest, every member of every wide union would come out at arity 2 and the
        # whole arity rule would collapse to "always dispatch". Caught by the corpus row that
        # asks for one arm of nine, on its first run.
        if inner in nested:
            continue
        # The TOP-LEVEL name of each arm, not every name inside it: `Step[Any] | Gather` has arms
        # `Step` and `Gather`, and walking every `Name` also collected `Any` from the subscript —
        # inflating `WorkflowOp` to ten arms and quietly raising the bar a decision must clear.
        names = {name for arm in _flatten_union(inner) if (name := _arm_name(arm))}
        if len(names) > 1:
            found.append(names)
    return found


def _union_arms(node: ast.expr) -> set[str]:
    """The union arm names anywhere in `node` — the flattened form, where arity does not matter."""
    return {name for arms in _union_arm_sets(node) for name in arms}


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _is_comprehension_filter(call: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """Is `call` inside a comprehension's `if` clause — i.e. SELECTING rather than deciding?

    The `if` clause specifically, not "somewhere in a comprehension". `atom if isinstance(atom,
    Atom) else Atom.of(...)` sits in the comprehension's ELEMENT, where both arms do work, and
    that is a two-arm decision the sweep converts."""
    node: ast.AST | None = call
    while node is not None:
        parent = parents.get(node)
        if isinstance(parent, ast.comprehension) and any(node is test for test in parent.ifs):
            return True
        node = parent
    return False


def _is_refusing_guard(call: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    """`not isinstance(...)` whose enclosing `if` raises — a boundary check, not a dispatch.

    Requires BOTH the negation and the raise. `if not isinstance(x, T): return None` is a
    two-arm decision wearing a guard's syntax, and the sweep should see it."""
    node: ast.AST = call
    negated = False
    while (parent := parents.get(node)) is not None:
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.Not):
            negated = True
        if isinstance(parent, ast.If):
            raises = any(isinstance(inner, ast.Raise) for inner in ast.walk(parent))
            return negated and raises
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return False
        node = parent
    return False


def _is_any(annotation: ast.expr | None) -> bool:
    """Is this annotation `Any` (or absent)? The one annotation that proves nothing."""
    match annotation:
        case None:
            return False
        case ast.Name(id="Any") | ast.Attribute(attr="Any"):
            return True
        case _:
            return False


def _declared_in_scope(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> set[str]:
    """`_declared_names` over the functions ENCLOSING `node`, innermost out — never the module.

    Scoping this per function is not a refinement, it is the difference between working and not.
    Run over the module, `keys.py` reports `spliced` and `value` as declared because they are
    parameters of *other* functions — `_refuse_a_wrong_splice(spliced: Key, …)`,
    `Tag.__new__(value: str)` — and the three marker checks they guard read as dispatch. Names are
    function-scoped; a rule about names has to be too.

    Walks OUT rather than stopping at the innermost `def`, for the same reason
    `_enclosing_annotations` does: a comprehension or closure reads its outer function's
    parameters."""
    declared: set[str] = set()
    scope: ast.AST | None = node
    while scope is not None:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
            declared |= _declared_names(scope)
        scope = parents.get(scope)
    return declared


def _declared_names(tree: ast.AST) -> set[str]:
    """Names bound somewhere a type is DECLARED or inherited — parameters, loop targets,
    annotated assignments.

    A name bound only by a plain `spliced = values[hole.index]` is not here, and that is the
    whole point: its type is whatever the right-hand side was, which at these seams is `Any`.
    **Conservative on purpose, and it costs recall.** A real checker infers through a subscript,
    so `head = items[0]` over `items: list[str | Interpolation]` IS knowable and this rule calls
    it unknown. Under-reporting a ranking instrument is the safer direction: a missed site is
    found by the next pass over the file, where a false one sends a sweep to convert a boundary
    check into an `assert_never` that fires unconditionally."""
    declared: set[str] = set()
    for node in ast.walk(tree):
        match node:
            # An `Any`-annotated parameter is NOT declared for this purpose — `Any` is precisely
            # the blocker (plan item 3a-bis), and counting it made the two handler tables classify
            # differently for no reason but where their `op` came from: `recording._dispatch`
            # takes `op: Any` and `replay._drive` reads it off `gen.send()`. Same problem, same
            # remedy, and the census said two different things.
            case ast.arg(arg=name, annotation=annotation) if not _is_any(annotation):
                declared.add(name)
            case ast.AnnAssign(target=ast.Name(id=name)):
                declared.add(name)
            case ast.For(target=target) | ast.comprehension(target=target):
                declared |= {n.id for n in ast.walk(target) if isinstance(n, ast.Name)}
    return declared


def _tested_type_is_knowable(
    tested: ast.expr,
    union_attributes: set[str],
    declared: set[str],
    returns: set[str] = frozenset(),  # ty: ignore[invalid-parameter-default] - read-only default
) -> bool:
    """Could a type checker say what `tested` is — i.e. would `assert_never` over it be a PROOF?

    The one discrimination that separates the sweep's population from the marker checks around
    it, and the reason the pre-pilot `keys.py` reads as 11 dispatch sites when 8 is the answer.
    Two ways to fail it, and both are `Any` in practice:

    - an attribute access whose attribute is undeclared — `interpolation.value` is `Any` by the
      stdlib's own signature, while `element.tag` is `str | Hole` because `SkeletonTerm` says so;
    - a bare name bound only by assignment, which carries its right-hand side's type and at these
      seams that is `Any` again.

    `assert_never` on an `Any` errors UNCONDITIONALLY (plan item 3a-bis) — a check that always
    fires is noise, not a proof — so a site that fails this test is not a totality site however
    much it looks like one."""
    match tested:
        case ast.Name(id=name):
            return name in declared
        case ast.Attribute(attr=attr):
            # A declared attribute, or a PROPERTY — `fork_ctx.phase` is a
            # `def phase(self) -> Phase` and reads exactly like a field at the call site.
            return attr in union_attributes or attr in returns
        case ast.Call(func=ast.Name(id=callee)) | ast.Call(func=ast.Attribute(attr=callee)):
            return callee in returns
        case ast.YieldFrom(value=inner) | ast.Await(value=inner):
            # `match (yield from step(...)):`: the subject is whatever the delegate
            # returns, and the delegate declares it.
            return _tested_type_is_knowable(inner, union_attributes, declared, returns)
        case ast.Subscript(value=ast.Name(id=name)):
            # `items[0]` over `items: list[str | Interpolation]`. `ty` narrows this and the census
            # could not, which left `_has_leading_tag` — a match that ALREADY ends in
            # `assert_never` — reported as unannotated.
            return name in declared
        case _:
            return False


def _isinstance_types(call: ast.Call) -> tuple[str, ...]:
    """The type names `call` tests against, out of a `A | B`, an `(A, B)`, or a lone name.

    Reads each element's HEAD, not every `Name` inside it. Walking for names recorded
    `isinstance(node, ast.Call)` as testing `ast` — the module — which is wrong on its face and
    put every `ast.*` test in the detector's own body into the census's universe."""
    if len(call.args) < 2:
        return ()
    second = call.args[1]
    elements = (
        second.elts
        if isinstance(second, ast.Tuple)
        else _flatten_union(second)
        if isinstance(second, ast.BinOp)
        else [second]
    )
    return tuple(sorted({name for e in elements if (name := _arm_name(e))}))


def _decision_root(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> ast.AST | None:
    """The whole `if`/`elif` chain (or conditional expression) this test belongs to.

    An `elif` is an `If` in the previous `If`'s `orelse`, so the chain's root is found by climbing
    while that holds. Without this, each `elif isinstance(x, B)` reads as its own one-arm decision
    and a three-arm dispatch counts as three selections."""
    branch: ast.AST | None = None
    current: ast.AST | None = node
    while current is not None:
        parent = parents.get(current)
        if isinstance(parent, (ast.If, ast.IfExp)):
            branch = parent
            break
        if isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            return None
        current = parent
    while isinstance(branch, ast.If):
        parent = parents.get(branch)
        if isinstance(parent, ast.If) and any(branch is arm for arm in parent.orelse):
            branch = parent
            continue
        break
    return branch


def _identifier(node: ast.expr | None) -> str | None:
    """The identifier `node` IS, and `None` when it is any other expression.

    **Whether an expression is a bare NAME is not the question the two type readers ask.**
    `_arm_name` and `_tested_type_is_knowable` read a TYPE, so both fold `ast.Attribute` and
    `ast.Subscript` down to a bare name; here that fold is the error. `compose_key` and
    `keys.compose_key` are one identifier and one attribute access, and a reader that confused
    them would count a dotted call as a bare one.

    Every caller asking it here is what makes their guards one guard."""
    # lint: totality(selection); the `ast.Name` arm is the distinction stated above, not a
    # duplicate of the two type readers.
    match node:
        case ast.Name(id=name):
            return name
        case _:
            return None


def _arms_named(root: ast.AST, tested: str) -> set[str]:
    """Every type this decision names for `tested` — across the chain's `elif`s."""
    return {
        name
        for call in ast.walk(root)
        if isinstance(call, ast.Call)
        and _identifier(call.func) == "isinstance"
        and len(call.args) >= 2
        and ast.unparse(call.args[0]) == tested
        for name in _isinstance_types(call)
    }


def _covers_the_union(named: set[str], arity: int) -> bool:
    """Could a decision naming these arms EXHAUST a union of `arity`, counting the `else`?

    `if isinstance(op, AskLLM):` over the arms of `WorkflowOp` names one and
    leaves the rest unaccounted for: a SELECTION of one case, where converting to a total `match`
    would mean writing eight arms nobody wants. `if isinstance(head, str): … else:` over the
    two-arm `str | Interpolation` names one and the `else` takes the other, so it IS total and the
    `else` is exactly what should become a named arm.

    Hence `arity - 1`: the `else` is worth one arm and no more."""
    return len(named) >= arity - 1


def _classify(
    node: ast.Call,
    parents: dict[ast.AST, ast.AST],
    members: set[frozenset[str]],
    attributes: set[str],
    types: tuple[str, ...],
    returns: set[str],
) -> str:
    """One `isinstance` site's population. The whole rule, in the order the tests are cheapest.

    Order matters and is not arbitrary: syntax first (a comprehension `if`, a negation over a
    raise), because those say what the site DOES regardless of types; then knowability, because a
    site over an `Any` cannot be a totality site whatever its type names look like; and only then
    union membership, which is the positive signal."""
    if _is_comprehension_filter(node, parents):
        return "filter"
    if _is_refusing_guard(node, parents):
        return "guard"
    if not _tested_type_is_knowable(
        node.args[0], attributes, _declared_in_scope(node, parents), returns
    ):
        # A marker check on a value the type system calls `Any`. The TYPE name may well be a union
        # member — `isinstance(spliced, Key)`, where `Key` appears in a union somewhere — but the
        # tested VALUE is untyped, so no `match` over it can be closed.
        return "guard"
    tested = ast.unparse(node.args[0])
    root = _decision_root(node, parents)
    named = _arms_named(root, tested) if root is not None else set(types)
    arity = _covering_arity(set(types), members)
    if arity is None:
        return "foreign"
    return "dispatch" if _covers_the_union(named, arity) else "selection"


def _top_level_classes(pattern: ast.pattern) -> set[str]:
    """The classes an arm names AT ITS TOP LEVEL — never one nested inside a structure.

    `case Atom()` and `case Atom() as a` name `Atom`; `case [Hole() as hole]` and
    `case {"rationale": str()}` name NOTHING, because they dispatch on a sequence and a mapping
    and merely destructure a member on the way through. Walking the whole pattern conflates the
    two, which read `_fill_coordinate`'s sequence match and `permission`'s mapping match as union
    dispatches over `SkeletonAtom` and `str`."""
    match pattern:
        # A class pattern may be QUALIFIED (`ast.Name()`, `model.Left()`), which parses as an
        # `Attribute` rather than a `Name`. Reading only `Name` would leave a dead qualified arm
        # invisible to the shadowed gate.
        case ast.MatchClass(
            cls=ast.Name(id=name) | ast.Attribute(attr=name),
            patterns=positional,
            kwd_patterns=keywords,
        ):
            # A REFINED pattern does not cover its class. `SkeletonTerm(tag=str() as tag)` matches
            # only the terms whose tag is a `str`, so a `SkeletonTerm` with a `Hole` tag falls
            # past it — which is precisely what `registry.Shape.tag`'s `case _: return ""` is for.
            # Same defect as counting a guarded arm, one construction over. A bare capture
            # (`Step(op=inner)`) refines nothing and still covers.
            refined = any(
                not (isinstance(sub, ast.MatchAs) and sub.pattern is None)
                for sub in (*positional, *keywords)
            )
            return set() if refined else {name}
        case ast.MatchAs(pattern=ast.pattern() as inner):
            return _top_level_classes(inner)
        case ast.MatchOr(patterns=alternatives):
            return {name for alt in alternatives for name in _top_level_classes(alt)}
        case _:
            return set()


def _match_arm_classes(node: ast.Match) -> set[str]:
    """The class names this `match` names as arms — how a union dispatch is recognised.

    **A GUARDED arm does not count.** `case Atom(text=text) if text != value.text:` names `Atom`
    and covers only part of it: an `Atom` whose text agrees falls straight through, which in
    `registry._bind` is the intended "matched, nothing to bind" path. Counting it made the census
    call that match exhaustive, and the mechanical `assert_never` pass duly added an arm `ty`
    rejected — `Inferred type of argument is Atom`. The checker caught it; the census had not."""
    return {
        name
        for case in node.cases
        if case.guard is None
        for name in _top_level_classes(case.pattern)
    }


def _never_returning(tree: ast.AST) -> set[str]:
    """Functions declared `-> Never` / `-> NoReturn` — calling one IS a refusal.

    Without this the census calls a delegated refusal a silent fall-through, and it did: both
    handler tables ENDED `case _: refuse_unknown_op(...)` — a `-> Never` helper that existed
    precisely
    so every walk refuses identically. They read as the defect they were written to fix."""
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and isinstance(node.returns, (ast.Name, ast.Attribute))
        and (node.returns.id if isinstance(node.returns, ast.Name) else node.returns.attr)
        in ("Never", "NoReturn")
    }


def _match_closure(node: ast.Match, never_returning: set[str]) -> str:
    """How this `match` ends: `open` (silently), `raising`, `assert_never`, or `no-wildcard`.

    The distinction the census is FOR: an unmatched arm that falls to a default path (an `open`
    ending) mis-dispatches silently, and no `isinstance` count can find it, because there is no
    `isinstance` there to count.

    A wildcard that CALLS a `-> Never` function counts as raising. Reading only for an `ast.Raise`
    node scores a delegated refusal as the defect it prevents."""
    final = node.cases[-1]
    calls = [
        (inner, _identifier(inner.func))
        for inner in ast.walk(final)
        if isinstance(inner, ast.Call) and _identifier(inner.func)
    ]
    names = {name for _call, name in calls}
    # A wildcard and a NAMED wildcard are different questions; conflating them reports
    # `case _: return 0` as having no wildcard at all. `case _` is `MatchAs(name=None)`;
    # `case unreachable` is `MatchAs(name='unreachable')`. Both catch everything; only the second
    # can carry the value into a proof.
    catches_all = isinstance(final.pattern, ast.MatchAs) and final.pattern.pattern is None
    bound = final.pattern.name if catches_all and isinstance(final.pattern, ast.MatchAs) else None
    # **A bound wildcard handed to a `-> Never` function is the same proof `assert_never` is**,
    # and reading only for the name `assert_never` misses it: a helper taking `op: Never` makes
    # `ty` reject the call the day an arm goes missing, while its runtime message names the walk.
    # The binding matters in both directions: passing `op` rather than `unreachable` re-widens the
    # type, and the proof disappears with the suite still green.
    # `bound` is tested BEFORE the comparison rather than beside it. `_identifier` answers `None`
    # for an expression that is not an identifier, and `bound` is `None` for an unnamed wildcard,
    # so `_identifier(argument) == bound` would read `case _: assert_never(f())` as a proof.
    proves = bound is not None and any(
        name in ({"assert_never"} | never_returning)
        and any(_identifier(argument) == bound for argument in call.args)
        for call, name in calls
    )
    if proves:
        return "assert_never"
    if not catches_all:
        return "no-wildcard"
    refuses = names & never_returning or any(isinstance(i, ast.Raise) for i in ast.walk(final))
    return "raising" if refuses else "open"


def _match_population(
    node: ast.Match,
    parents: dict[ast.AST, ast.AST],
    members: set[frozenset[str]],
    attributes: set[str],
    never_returning: set[str],
    returns: set[str],
) -> str | None:
    """This `match`'s population, or `None` when it is not a union dispatch at all.

    **Knowability does NOT exclude a match, and that is the difference from the `isinstance`
    side.** There, an `Any` subject means no `match` over it can be closed, so the site is a
    boundary check. Here the `match` already exists: an `Any` subject changes the REMEDY — a
    refusing wildcard
    instead of `assert_never`, exactly as the plan's step 3 says — and not whether the silent
    fall-through is a finding."""
    # A GUARDED arm over the union takes the fall-through slot. `registry._bind` writes
    # `case Atom(text=text) if text != value.text:` and then lets an agreeing `Atom` fall out of
    # the match entirely — that IS the arm, spelled as an absence. The arity rule counts the
    # else-slot as worth one arm, and here it is already spent, so such a match can never be
    # closed by an added arm. `ty` said as much when the mechanical pass tried: "Inferred type of
    # argument is `Atom`".
    guarded = any(
        case.guard is not None and _top_level_classes(case.pattern) & _all_arms(members)
        for case in node.cases
    )
    named = _match_arm_classes(node)
    arity = _covering_arity(named, members) if named else None
    if guarded or arity is None or not _covers_the_union(named, arity):
        # The same arity rule the `isinstance` side uses, one grammar over: a `match` naming one
        # arm of a wide union and falling to `case _` is a SELECTION, and converting it would mean
        # writing the other arms, which nobody asked for.
        return None
    knowable = _tested_type_is_knowable(
        node.subject, attributes, _declared_in_scope(node, parents), returns
    )
    match _match_closure(node, never_returning):
        case "assert_never" if knowable:
            return "closed-match"
        case "assert_never":
            # The arm is written and the subject is `Any`, so the check compiles and proves
            # NOTHING — `Any` satisfies a `Never` parameter. This is `replay._drive` exactly, and
            # the meter had the same defect it had just found there: it reported closed while
            # deleting an arm produced zero errors. A proof needs a type to be a proof.
            return "unannotated-match"
        case "open":
            return "open-match"
        case _ if knowable:
            return "unclosed-match"
        case _:
            # Refusing at runtime over a subject the checker cannot name. NOT closed — the remedy
            # is the sweep's step 1, tighten the annotation to the union, and only then does
            # `assert_never` become a proof rather than a check that always fires.
            return "unannotated-match"


def _row(
    path: Path,
    lines: list[str],
    node: ast.stmt | ast.expr,
    population: str,
    tested: str,
    types: tuple[str, ...],
) -> TotalitySite:
    """One census row. A module-level function rather than a closure over the file loop — the
    closure form binds `path` and `lines` late, which `ruff`'s B023 is right to refuse."""
    return TotalitySite(
        filename=str(path),
        line=node.lineno,
        text=lines[node.lineno - 1].strip() if node.lineno <= len(lines) else "",
        population=population,
        tested=tested,
        types=types,
    )


def totality_census(paths: Iterable[str | Path]) -> list[TotalitySite]:
    """Classify every `isinstance` site in `paths`. The ranking instrument, and the gate's input.

    Whole-set rather than per-file because union membership is a repo-wide fact: a handler
    dispatching on `WorkflowOp` reads an alias declared in `ops.py`, and a per-file pass would
    call it foreign."""
    files = _expanded(paths)
    trees: dict[Path, ast.Module] = {}
    members: set[frozenset[str]] = set()
    attributes: set[str] = set()
    returns: set[str] = set()
    never_returning: set[str] = set()
    for path in files:
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        trees[path] = tree
        members |= _union_members(tree)
        attributes |= _union_attributes(tree)
        returns |= _declared_returns(tree)
        never_returning |= _never_returning(tree)

    sites: list[TotalitySite] = []
    for path, tree in trees.items():
        parents = _parents(tree)
        lines = path.read_text().splitlines()
        per_file: list[TotalitySite] = []
        for node in ast.walk(tree):
            match node:
                case ast.Call(func=ast.Name(id="isinstance"), args=[tested, _, *_]):
                    types = _isinstance_types(node)
                    per_file.append(
                        _row(
                            path,
                            lines,
                            node,
                            _classify(node, parents, members, attributes, types, returns),
                            ast.unparse(tested),
                            types,
                        )
                    )
                case ast.Match():
                    population = _match_population(
                        node, parents, members, attributes, never_returning, returns
                    )
                    if population is not None:
                        per_file.append(
                            _row(
                                path,
                                lines,
                                node,
                                population,
                                ast.unparse(node.subject),
                                tuple(sorted(_match_arm_classes(node))),
                            )
                        )
        sites.extend(
            site
            for site in _promote_duplicated_enumerations(per_file, tree, parents, members)
            if not any(
                _escape_category(lines[i]) in _TOTALITY_CATEGORIES
                for i in _escape_window(lines, site.line)
                if 0 <= i < len(lines)
            )
        )
    return sites


def _promote_duplicated_enumerations(
    sites: list[TotalitySite],
    tree: ast.Module,
    parents: dict[ast.AST, ast.AST],
    members: set[frozenset[str]],
) -> list[TotalitySite]:
    """A filter written the same way two or more times in one file is one enumeration, N times.

    The fix there is to build it once, not to `match` harder — the population no `isinstance`
    count reveals, and the one the pilot actually found. Promotion is on `ast.unparse` identity of
    the predicate, which is the limit stated at the head of this section."""
    # Only a filter over a UNION can be a duplicated enumeration. Promoting on text alone put 11
    # `isinstance(node, ast.Call)` filters — a foreign hierarchy, in this file's own body — into
    # the sweep's work list, where converting them would mean nothing.
    unions = {name for union in members for name in union}
    predicates = Counter(
        f"{site.tested}|{site.types}"
        for site in sites
        if site.population == "filter" and unions & set(site.types)
    )
    return [
        replace(site, population="duplicated-enumeration")
        if site.population == "filter" and predicates[f"{site.tested}|{site.types}"] > 1
        else site
        for site in sites
    ]


# --- ordered arms (`--ordered-arms`) --------------------------------------------------------
#
# A `match` is an ORDERED decision table, and refinement interacts with that order in two ways.
# Refinement itself is good — `case Interpolation(value=Key() as key)` says exactly which case it
# means — but a refined arm and a broader arm on the SAME class are only correct in one order, and
# the wrong one is silent both times.
#
#   SHADOWED   broad arm FIRST, refined arm after  -> the refined arm is DEAD. Always a defect.
#   ABSORBED   refined arm first, broad arm after  -> correct, and the checker goes BLIND: deleting
#              the refined arm leaves the match exhaustive, so `ty` says nothing.
#
# Deleting an absorbed arm passes `ty`, `just lint`, `--totality` and the suite alike. So an
# absorbed arm is a site where `assert_never` is not the guard and a named test pin has to be, and
# the census cannot say so because it reads the match's closure rather than its arms.
#
# The contract:
#
#     `--ordered-arms` rejects an unguarded refined arm after an unguarded broad arm when both
#     class patterns resolve to the same bare or qualified class name. `ordered_arm_report`
#     reports an unguarded refined arm before its broad arm because deleting that arm remains
#     exhaustive and is invisible to `ty`. Guarded refined arms are outside this report even
#     though their deletion can also change behavior without breaking exhaustiveness; they
#     require their own named behavior tests.
def _arm_class_refinements(pattern: ast.pattern) -> list[tuple[str, bool]]:
    """`[(class, is_refined)]` at an arm's top level, flattening `as` and `A() | B()`.

    Flattening `MatchOr` matters: `case str() | Interpolation():` is the broad arm that absorbs
    `compose_key`'s refined `AuthorityTag` arm, and a rule that reads only bare `MatchClass`
    misses it."""
    match pattern:
        case ast.MatchAs(pattern=ast.pattern() as inner):
            return _arm_class_refinements(inner)
        case ast.MatchOr(patterns=alternatives):
            return [row for alt in alternatives for row in _arm_class_refinements(alt)]
        # A class pattern may be QUALIFIED (`ast.Name()`, `model.Left()`), which parses as an
        # `Attribute` rather than a `Name`. Reading only `Name` would leave a dead qualified arm
        # invisible to the shadowed gate.
        case ast.MatchClass(
            cls=ast.Name(id=name) | ast.Attribute(attr=name),
            patterns=positional,
            kwd_patterns=keywords,
        ):
            refined = any(
                not (isinstance(sub, ast.MatchAs) and sub.pattern is None)
                for sub in (*positional, *keywords)
            )
            return [(name, refined)]
        case _:
            return []


def _ordered_arm_pairs(node: ast.Match) -> Iterator[tuple[str, str, int, int]]:
    """`(kind, class, refined_line, broad_line)` for each refined/broad pair on one class.

    Both arms must be UNGUARDED: a guard already declares its arm partial, so
    `case Key(scope=None) if …` before `case Key()` is neither hazard."""
    rows = [(case, _arm_class_refinements(case.pattern)) for case in node.cases]
    for i, (broad, broad_arms) in enumerate(rows):
        if broad.guard is not None:
            continue
        for cls in {c for c, refined in broad_arms if not refined}:
            for j, (other, other_arms) in enumerate(rows):
                if i == j or other.guard is not None:
                    continue
                if any(c == cls and refined for c, refined in other_arms):
                    kind = "shadowed" if j > i else "absorbed"
                    yield kind, cls, other.pattern.lineno, broad.pattern.lineno


def ordered_arm_report(paths: Iterable[str | Path]) -> list[tuple[str, Violation]]:
    """`[("shadowed" | "absorbed", violation)]` for every refined/broad pair on one class."""
    out: list[tuple[str, Violation]] = []
    for path in _expanded(paths):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        lines = path.read_text().splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Match):
                continue
            for kind, cls, at, broad_at in _ordered_arm_pairs(node):
                consequence = (
                    "it can never match"
                    if kind == "shadowed"
                    else "so deleting it stays exhaustive and `ty` cannot see it; this arm "
                    "needs a named test pin, not `assert_never`"
                )
                out.append(
                    (
                        kind,
                        Violation(
                            f"ordered-arms-{kind}",
                            f"a refined `{cls}` arm sits "
                            f"{'after' if kind == 'shadowed' else 'before'}"
                            f" a broad `{cls}` at line {broad_at} — {consequence}",
                            at,
                            lines[at - 1].strip() if at <= len(lines) else "",
                            str(path),
                        ),
                    )
                )
    return out


def check_ordered_arms(paths: Iterable[str | Path]) -> list[Violation]:
    """The GATE half — shadowed only.

    Absorbed is reported by `ordered_arm_report` and deliberately not gated: it is correct code
    whose checker coverage is weaker than it looks, so the answer is a test pin per site rather
    than a rule everyone silences. Shadowed is dead code and is always wrong."""
    return [v for kind, v in ordered_arm_report(paths) if kind == "shadowed"]


def _escape_category(line: str) -> str | None:
    """The category an escape comment claims — `lint: totality(selection)` -> `"selection"`.

    `None` when the line carries no escape at all. A BARE `lint: totality` returns the empty
    string, which is in no category and therefore does not escape: an uncategorised escape is the
    free-prose hole this replaced, so it must fail rather than fall back to the old behaviour."""
    # An escape IS a comment. Without this, the rule's own docstring — which has to quote the
    # syntax to document it — reads as two malformed escapes, and a rule that cannot describe
    # itself without tripping is one people route around.
    if _TOTALITY_ESCAPE not in line or not line.lstrip().startswith("#"):
        return None
    tail = line.split(_TOTALITY_ESCAPE, 1)[1]
    if not tail.startswith("("):
        return ""
    return tail[1:].split(")", 1)[0].strip()


def check_totality_escapes(paths: Iterable[str | Path]) -> list[Violation]:
    """Every escape names a known category, and a `deferred` one says what unblocks it."""
    out: list[Violation] = []
    for path in _expanded(paths):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if "_TOTALITY_ESCAPE" in line or "_TOTALITY_CATEGORIES" in line:
                continue
            category = _escape_category(line)
            if category is None:
                continue
            reason = line.split(")", 1)[-1].strip(" —-:") if category else ""
            if category not in _TOTALITY_CATEGORIES:
                out.append(
                    Violation(
                        "totality-escape",
                        f"escape category {category!r} is not one of "
                        f"{'/'.join(_TOTALITY_CATEGORIES)} — an escape states a "
                        f"CLASSIFICATION, and free prose is how a preference and a schedule "
                        f"both passed as one",
                        n,
                        line.strip(),
                        str(path),
                    )
                )
            elif category == "deferred" and "until" not in line.lower():
                out.append(
                    Violation(
                        "totality-escape",
                        "a `deferred` escape must say what unblocks it — write "
                        "`until <condition>`, or it becomes permanent by being forgotten",
                        n,
                        line.strip(),
                        str(path),
                    )
                )
            elif len(re.findall(r"[A-Za-z]{3,}", reason)) < 3:
                out.append(
                    Violation(
                        "totality-escape",
                        f"escape category {category!r} carries no reason — the category says "
                        f"WHICH population, the reason says why this site is in it",
                        n,
                        line.strip(),
                        str(path),
                    )
                )
    return out


def check_totality(paths: Iterable[str | Path]) -> list[Violation]:
    """The gate half: one violation per in-scope site. The census is the other half."""
    return [
        Violation(
            "totality",
            f"{site.population}: dispatch on {'|'.join(site.types)} over {site.tested!r} — "
            f"convert to `match` closed by `assert_never`"
            if site.population == "dispatch"
            else f"{site.population}: {site.tested!r} is filtered by the same predicate elsewhere "
            f"in this file — build the enumeration once",
            site.line,
            site.text,
            site.filename,
        )
        for site in totality_census(paths)
        if site.population in _TOTALITY_IN_SCOPE
    ]


# Per-file rules answer "is THIS file ok"; whole-set rules answer a question about the set —
# is one tag's shape consistent across every site, is every workflow in the role set. Two dicts
# rather than a branch per rule, so adding a rule is a row.
_PER_FILE = {
    "--layers": check_layer_file,
    "--channels": check_channel_file,
    "--deps": check_deps_file,
    "--ledger-reads": check_ledger_reads_file,
    "--sql-templates": check_sql_templates_file,
    "--lazy-imports": check_lazy_imports_file,
    "--working-notes": check_working_notes_file,
    "--key-composition": check_key_composition_file,
    "--terminal-holes": check_terminal_holes_file,
    "--public-prose": check_public_prose_file,
}

# --- dangling test citations (`--test-citations`) -------------------------------------------
#
# A citation is a claim, and a citation to a test is one a reader will act on: they go looking.
# A citation dangles as easily while fixing another as while writing fresh prose, so this wants a
# gate rather than care. `just docs-check` gates dangling LINKS; this gates identifiers.

_TEST_CITATION = re.compile(r"``?(test_[A-Za-z0-9_]+)``?|::(test_[A-Za-z0-9_]+)")
"""What this rule SCANS, which is its domain and therefore the honest statement of its reach.

Covered: a `test_*` identifier in single or double backticks (both forms are in this tree), and a
`::test_*` pytest node id. A parametrized citation resolves on its base name, since `[` ends the
match — `test_x[sqlite]` asks about `test_x`.

**NOT covered, deliberately: a bare unquoted `test_x` in prose.** Scanning that reddens ordinary
sentences about naming conventions and costs more than it catches. So this gate does not prove
"no dangling citation anywhere"; it proves it for the marked-up forms, which is what a reader
follows. Widen the regex before widening the claim.
"""

_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+(test_[A-Za-z0-9_]+)", re.M)

ALLOWLIST = Path(__file__).resolve().parents[2] / "scripts" / "test-citation-allowlist.txt"
PRIVATE_ALLOWLIST = ALLOWLIST.with_name("test-citation-private-allowlist.txt")


def _defined_tests(tests_dir: Path) -> set[str]:
    """Every test a citation can resolve to — a function anywhere under `tests/`, or a module."""
    names: set[str] = set()
    for file in tests_dir.rglob("*.py"):
        names.add(file.stem)
        names.update(_TEST_DEF.findall(file.read_text()))
    return names


def _allowlisted() -> set[str]:
    out = set()
    for raw in (
        raw
        for file in (ALLOWLIST, PRIVATE_ALLOWLIST)
        if file.exists()
        for raw in file.read_text().splitlines()
    ):
        if line := raw.split("#", 1)[0].strip():
            out.add(line)
    return out


def _span_parity(line: str) -> bool:
    """Does this line leave an INLINE code span open? A fence delimiter does not count.

    A ``` fence opener has three backticks, an odd count, so counting it as inline code makes the
    joiner swallow the following line, and a node id on the first line of a fenced block fuses
    with the `assert` under it into a name that resolves to nothing.
    """
    stripped = line.lstrip()
    if stripped.startswith("```") or stripped.startswith("~~~"):
        return False
    return line.count("`") % 2 == 1


def _unwrapped(text: str) -> Iterator[tuple[int, str]]:
    """Yield `(line number, logical line)`, rejoining a citation the 99-char wrap split in two.

    **This is why the rule is not line-by-line.** A citation wrapped mid-identifier puts the head
    on one line and the tail on the next, and the head resolves to nothing. A gate that flags
    correct prose is worse than no gate, because the first thing anyone does with one is switch
    it off.

    The state is a running parity carried ACROSS lines. A per-line odd-count test misses the
    middle of a multi-line region: a line can close the span opened above it and open another
    (two backticks, even, and still unbalanced in context).

    Capped at two continuations so a stray backtick swallows a paragraph rather than a file.

    Every physical line is scanned joined-forward, including one that begins inside a span: a
    line can close the span above it and open a NEW citation of its own, which skipping it would
    hide. Duplicates are cheaper than misses, so the caller dedupes by `(file, name)`.
    """
    lines = text.splitlines()
    inside = False  # parity of backticks seen SO FAR — a code span open at end of line
    for i, line in enumerate(lines):
        inside ^= _span_parity(line)
        logical, joins = line, 0
        # Join while the span is still open at end of this logical line.
        open_now = inside
        while open_now and joins < 2 and i + joins + 1 < len(lines):
            joins += 1
            nxt = lines[i + joins]
            logical += nxt.lstrip()
            open_now ^= _span_parity(nxt)
        yield i + 1, logical


def check_test_citations(paths: Iterable[str | Path]) -> list[Violation]:
    """A cited test name must resolve to a test — or be allowlisted as deliberately dead.

    Fails in BOTH directions, which is what keeps the allowlist from becoming furniture: an
    unresolvable citation that is not listed, and a listed name that now resolves (stale). The
    second is the one a baseline usually lacks, and it is why this file can only shrink.
    """
    repo = Path(__file__).resolve().parents[2]
    defined = _defined_tests(repo / "tests")
    allowed = _allowlisted()
    out: list[Violation] = []
    for p in paths:
        path = Path(p)
        seen: set[str] = set()
        for n, line in _unwrapped(path.read_text()):
            for single, nodeid in _TEST_CITATION.findall(line):
                name = single or nodeid
                if name in defined or name in allowed or name in seen:
                    continue
                seen.add(name)  # one report per name per file, at its first line
                out.append(
                    Violation(
                        "dangling-test-citation",
                        f"`{name}` names no test — no `def {name}` under tests/ and no "
                        f"tests/**/{name}.py. Fix the citation, or add it to "
                        f"{ALLOWLIST.name} with a reason if the name is dead on purpose",
                        n,
                        line.strip()[:100],
                        str(path),
                    )
                )
    for name in sorted(allowed & defined):
        out.append(
            Violation(
                "stale-test-citation-allowlist",
                f"`{name}` is allowlisted as deliberately dead but now RESOLVES — drop the entry",
                0,
                ALLOWLIST.name,
                str(ALLOWLIST),
            )
        )
    return out


_WHOLE_SET = {
    "--coordinate-roles": check_coordinate_roles,
    "--python-codegen": check_python_codegen,
    "--key-borrowing": check_key_borrowing,
    "--forged-join": check_forged_joins,
    "--key-literals": check_key_literals,
    "--authority-tags": check_authority_tags,
    "--authority-scopes": check_authority_scopes,
    "--role-coverage": check_role_coverage,
    "--layer-coverage": check_layer_coverage,
    "--test-citations": check_test_citations,
    "--totality": check_totality,
    "--ordered-arms": check_ordered_arms,
    "--totality-escapes": check_totality_escapes,
}


def _run_mode(mode: str, paths: list[str]) -> list[Violation]:
    if whole_set := _WHOLE_SET.get(mode):
        return whole_set(paths)
    if mode == "--key-registry":
        shapes, violations = build_key_registry(paths)
        if not violations:
            out = Path("build/key-registry.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(KeyMap.from_shapes(shapes).to_json())
            tags = len({shape.tag for shape in shapes})
            print("key-registry:", tags, "namespaces,", len(shapes), "variants ->", out)
        return violations
    check = _PER_FILE.get(mode, check_file)
    return [v for p in paths for v in check(p)]


def configured_roots(mode: str, given: list[str]) -> list[str]:
    """The roots a tree adds for `mode`: `extra_paths_<mode>` in `[tool.effective.lint]`, else
    `extra_paths`, resolved against the `pyproject.toml` that names them. A root under one already
    given is left out, and a named root that does not exist raises."""
    own = configured("extra_paths_" + mode.removeprefix("--").replace("-", "_"))
    extra = own if own is not None else configured("extra_paths") or ()
    found = config_root()
    if not extra or found is None:
        return []
    roots: list[str] = []
    for entry in extra:
        path = Path(os.path.relpath(found[0] / entry))
        if not path.exists():
            raise FileNotFoundError(f"[tool.effective.lint] names {entry}, which is absent")
        if not any(path.is_relative_to(r) for r in given):
            roots.append(str(path))
    return roots


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m effective.lint <workflow.py> ...`` — exit 1 on violations."""
    args = argv if argv is not None else sys.argv[1:]
    # DERIVED from the two dispatch dicts (plus the one mode that is neither), so "adding a rule
    # is a row" holds. A rule missing from a hand-list here would fall through to the DEFAULT
    # determinism mode, which runs over files it does not apply to and reports nonsense rather
    # than failing loudly.
    known = (*_PER_FILE, *_WHOLE_SET, "--key-registry")
    mode = next((a for a in args if a in known), "")
    raw = [a for a in args if not a.startswith("--")]
    if raw:
        raw += configured_roots(mode, raw)
    # --deps and --ledger-reads apply to every file in a tree, so accept directories and glob
    # them; a new file is covered automatically rather than needing to join a hand-list.
    paths: list[str] = []
    for p in raw:
        path = Path(p)
        if path.is_dir():
            # `--test-citations` reads prose as well as source: a dangling cite in `docs/` or a
            # skill misleads exactly as much as one in a docstring, and the maintained layer is
            # where it is read on trust.
            globs = ("*.py", "*.md") if mode == "--test-citations" else ("*.py",)
            for pattern in globs:
                paths.extend(str(f) for f in sorted(path.rglob(pattern)))
        else:
            paths.append(p)
    # Role default (the single source of truth): with no explicit files, the
    # determinism-boundary mode lints the workflow-role set and `--layers` the
    # layer-role set — so the justfile carries no hand-list to drift from the gate.
    if not paths and mode == "--key-registry":
        paths = [str(path) for path in _expanded(KEY_REGISTRY_SRCS)]
    elif not paths:
        role = WORKFLOW_ROLE_SRCS if mode == "" else LAYER_ROLE_SRCS if mode == "--layers" else ()
        paths = _resolve_role_default(role)
    violations = _run_mode(mode, paths)
    for v in violations:
        print(v)
    if violations:
        kind = {
            "--layers": "layer-authority",
            "--channels": "channel (volatile-last-cache / independence)",
            "--deps": "seam dependency-direction",
            "--ledger-reads": "canonical ledger-read",
            "--sql-templates": "SQL-boundary (t-string)",
            "--key-composition": "key-composition (an identity built by formatting)",
            "--terminal-holes": "terminal-hole (an unwrapped value in the exempt last position)",
            "--authority-tags": "reserved authority-namespace",
            "--authority-scopes": "authority-scope (a declared reach the namespace does not keep)",
            "--coordinate-roles": "coordinate-role (a coordinate that says nothing)",
            "--python-codegen": "python-codegen (Python source built by interpolation)",
            "--key-registry": "key-registry (one tag, one shape)",
            "--key-borrowing": "key-borrowing (a production namespace at a shape it cannot mint)",
            "--forged-join": "forged-join (a key rebuilt by joining rendered terms)",
            "--key-literals": "key-literal (a spelled key production could not mint)",
            "--lazy-imports": "lazy-import (function-body import)",
            "--working-notes": "working-note (transient in a commit)",
            "--role-coverage": "role-coverage (a workflow the determinism gate never sees)",
            "--layer-coverage": "layer-coverage (a layer the authority gate never sees)",
            "--test-citations": "test-citation (a cited test name that resolves to nothing)",
            "--public-prose": "public-prose (our deployment or one vendor, in substrate prose)",
            "--totality": "totality (dispatch over a closed union, not closed by assert_never)",
            "--ordered-arms": "ordered-arms (a refined match arm that can never match)",
            "--totality-escapes": "totality-escape (an escape with no checked category)",
        }.get(mode, "determinism-boundary")
        print(f"\n{len(violations)} {kind} violation(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
