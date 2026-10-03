"""The runner: tool name to host call, with every name in the table actually served.

**This module is the point of the promotion.** The incumbent had `structural_apply` as a tool
*name* whose only implementations lived in a test and a script: `live_workspace._LOCAL` served four
file tools, `CodeAgentDomain.run` knew four more, and anything else hit
`raise ValueError(f"unknown tool: {name}")`. So the deterministic tail existed and nothing called
it. The rule that no tool's only implementation lives in a test or a script is
discharged by a table where every entry has a production body, and by
`test_every_tool_in_the_table_has_a_runner` refusing to let that drift.

**Opinionated, which means choosing rather than offering.** The incumbent shipped five edit modes
and wired two; the other three were built, unit-tested and unreachable from the durable path. Here
the ladder's default is `SEMANTIC` for a refactor and `STRUCTURAL` for a uniform rewrite, with
`WHOLE_FILE` as the drafting fallback — one constant, `DEFAULT_SCHEME`, states it, and the
alternatives sit beside it as measured options rather than as unresolved seams.

**The workspace is a mapping, not a directory, and reads flow through the tree.** A tool that
touched the real filesystem would make replay a lie: the recorded result would depend on whatever
the disk happened to hold. So the tree is workflow-local state threaded in, every tool is a pure
function of `(tree, args)`, and the only escapes are the deterministic subprocesses the ladder's
upper rungs need — ast-grep, ruff and jedi — which run past the yield boundary where a subprocess
is legitimate.

**Those three PARSE; `run_suite` EXECUTES, and the difference decides what needs isolating.**
pytest imports every module it collects, so a tree a model wrote runs before any assertion does.
Where that happens is `effective.coding.tier`'s question, not this module's: `run_suite` takes a
`tier` and the parsing below cannot tell the arms apart.

The one exception is `semantic_rename`: jedi needs a project on disk to resolve bindings
across files, so the caller supplies a `project_root`. That is the ladder's own rule showing
through — rung 3 needs the whole project, and no amount of API design makes it not need it.
"""

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from effective.coding.edits.semantic import SemanticError, declared_arms, references, rename
from effective.coding.edits.structural import StructuralEdit, StructuralRule, apply_structural
from effective.coding.gate import static_diagnostic, static_merge_gate
from effective.coding.tier import PYTHON, Tier, UnsafeTreePath, run_tree_command, tree_paths
from effective.domain import ToolRefused
from effective.machine.evidence import CommandRun
from effective.react import Tool


class Scheme(StrEnum):
    """The edit-scheme ladder as a choosable value: a tunable seam takes data."""

    TEXT = "text"
    STRUCTURAL = "structural"
    SEMANTIC = "semantic"


DEFAULT_SCHEME = Scheme.SEMANTIC
"""What a refactor uses unless told otherwise.

The default is the TOP rung deliberately. A sandbox makes the nominal scheme cheap and the
structural one impossible, so the path of least resistance is regex-over-code — this repo's named
defect class. Defaulting to semantic means a text edit has to argue for itself."""


class ToolError(RuntimeError):
    """A tool was asked for something it cannot do. Distinct from an unknown tool, which is a
    composition defect rather than a runtime one."""


class UnknownTool(KeyError):
    """No runner serves this name. Raised rather than defaulted: a silently-ignored tool call is
    how an agent appears to work while doing nothing."""


@dataclass(frozen=True)
class Workspace:
    """What one tool call acts on: the tree the call carries, and the project root the semantic
    rung needs.

    Built per call and changed by none. An edit RETURNS the tree it makes, so what a run has
    edited lives in the recorded results, and a domain serving the table keeps nothing between
    calls: a worker resumed on a fresh process is handed the tree the crashed one had
    (`coding.specs.coding_bind`).

    The tree is checked as a write is. A key here came from the seed, which no tool can repair,
    so it raises `UnsafeTreePath` and fails the task, where a key a model writes is refused to
    the model."""

    tree: Mapping[str, str] = field(default_factory=dict)
    project_root: str | None = None

    def __post_init__(self) -> None:
        tree_paths(self.tree)


def _materializable(tree: Mapping[str, str]) -> dict[str, str]:
    """`tree`, or a refusal the model sees: a key that leaves the tree, spells a file twice, or
    names a path another key names would be written wrongly, or crash, when the suite runs."""
    try:
        tree_paths(tree)
    except UnsafeTreePath as unsafe:
        raise ToolError(str(unsafe)) from unsafe
    for name, content in tree.items():
        try:
            content.encode()
        except UnicodeError as unencodable:
            raise ToolError(f"the content of {name!a} has no bytes to write") from unencodable
    return dict(tree)


def _read_file(ws: Workspace, args: Mapping[str, Any]) -> str:
    path = str(args["path"])
    if path not in ws.tree:
        raise ToolError(f"no such file in the workspace: {path}")
    return ws.tree[path]


def _list_dir(ws: Workspace, _args: Mapping[str, Any]) -> list[str]:
    return sorted(ws.tree)


def _write_file(ws: Workspace, args: Mapping[str, Any]) -> dict[str, str]:
    """Returns the RESULTING TREE, not a message, because a workflow learns what changed only
    from a recorded op result: on replay no tool runs, and a crash resumes on a worker holding
    nothing the crashed one built. The tree returned is the one checked, a copy taken once, so
    nothing that changes the tree it was handed can change what it vouches for."""
    path, content = str(args["path"]), str(args["content"])
    written = _materializable({**ws.tree, path: content})
    if (bad := static_diagnostic(content, path)) is not None:
        raise ToolError(bad)
    return written


def _check(ws: Workspace, _args: Mapping[str, Any]) -> str:
    return static_merge_gate(ws.tree) or "clean"


def _structural_apply(ws: Workspace, args: Mapping[str, Any]) -> dict[str, str]:
    """THE RUNNER THE INCUMBENT LACKED. Re-derives the edit from the recorded intent."""
    edit = StructuralEdit(
        path=str(args["path"]),
        rules=[StructuralRule(**rule) for rule in args["rules"]],
    )
    tree = dict(ws.tree)
    if edit.path not in tree:
        raise ToolError(f"no such file in the workspace: {edit.path}")
    content, total, error = apply_structural(tree[edit.path], edit.rules, edit.path)
    if error is not None:
        raise ToolError(error)
    if total == 0:
        raise ToolError(f"no rule matched anything in {edit.path} — the pattern hit 0 sites")
    if content is None:
        raise ToolError("ast-grep produced no output")
    if (bad := static_diagnostic(content, edit.path)) is not None:
        raise ToolError(f"the rewrite does not merge: {bad}")
    return {**tree, edit.path: content}  # the resulting tree, from one copy: see `_write_file`


def _semantic_rename(ws: Workspace, args: Mapping[str, Any]) -> str:
    if ws.project_root is None:
        raise ToolError("semantic_rename needs a project_root — rung 3 resolves across files")
    try:
        edit = rename(
            ws.project_root,
            str(args["path"]),
            int(args["line"]),
            int(args["column"]),
            str(args["new_name"]),
        )
    except SemanticError as e:
        raise ToolError(str(e)) from e
    return edit.diff or "no change"


def _semantic_references(ws: Workspace, args: Mapping[str, Any]) -> list[str]:
    if ws.project_root is None:
        raise ToolError("semantic_references needs a project_root")
    try:
        found = references(
            ws.project_root, str(args["path"]), int(args["line"]), int(args["column"])
        )
    except SemanticError as e:
        raise ToolError(str(e)) from e
    return [f"{r.path}:{r.line}:{r.column}" for r in found]


def _declared_arms(ws: Workspace, args: Mapping[str, Any]) -> list[str]:
    if ws.project_root is None:
        raise ToolError("declared_arms needs a project_root")
    try:
        return list(declared_arms(ws.project_root, str(args["path"]), str(args["alias"])))
    except SemanticError as e:
        raise ToolError(str(e)) from e


_FAILURE_LINE = re.compile(r"^(?:FAILED|ERROR)\s+(?P<node>\S+)", re.MULTILINE)
_COLLECT_CODES = frozenset({2, 3, 4})
"""pytest's vocabulary: 0 passed, 1 tests failed, 2 interrupted, 3 internal, 4 usage, 5 nothing
collected. Only 2/3/4 mean the suite could not do its job — and 5 is deliberately NOT here, since
"no tests collected" is a real answer about the tree rather than a broken environment."""


SUITE_COMMAND: tuple[str, ...] = (
    PYTHON,
    "-m",
    "pytest",
    "-q",
    "--no-header",
    "-p",
    "no:randomly",
    ".",
)
"""The success predicate's command, one place, both tiers.

`.` rather than an absolute path, because each tier materializes the tree into its own root and
runs from it — the host into a temp directory, the container into `/work`. `PYTHON` resolves to
`sys.executable` on the host and to the image's pinned interpreter inside."""


def run_suite(
    tree: Mapping[str, str], *, timeout: int = 120, tier: Tier | None = None
) -> CommandRun:
    """THE SUCCESS PREDICATE. Materialize the tree and run pytest over it.

    Named and exported rather than buried in the table, because it is the one tool whose result a
    verdict is computed from — `verdicts.py` reads this and nothing else for three of the machine's
    states. A deployment may substitute a different command; what it may not do is return a
    judgment instead of a measurement.

    The tree goes to a temp directory, never the real tree: the predicate must not be able to
    change the repo it is measuring.

    **That is a correctness property and it is not isolation**, which is what `tier` is for. This
    runs a tree a MODEL wrote, and pytest imports every module it collects — a `conftest.py` in
    the tree executes at collection, before any test does. On `Tier.HOST` that happens as the user
    driving the machine. `Tier.CONTAINER` puts it in a fresh, egress-denied, mount-free container
    instead (`effective.coding.tier`). The default stays HOST so no existing caller silently
    starts requiring podman; a surface that drives a model to write code passes CONTAINER.

    The parsing below is tier-independent by construction: both arms return the command's own exit
    code and its output, so a verdict cannot tell where the suite ran."""
    exit_code, out = run_tree_command(tree, SUITE_COMMAND, timeout=timeout, tier=tier)
    collection = None
    if exit_code in _COLLECT_CODES:
        collection = (out.strip().splitlines() or ["the suite could not collect"])[-1][:200]
    return CommandRun(
        exit_code=exit_code,
        failures=tuple(m.group("node") for m in _FAILURE_LINE.finditer(out)),
        collection_error=collection,
        output=out[-4000:],
    )


def _run_suite(ws: Workspace, args: Mapping[str, Any]) -> CommandRun:
    """The table's entry — TOTAL over what a caller can have meant by `tree`.

    Three cases. Two arms (`dict(tree) if isinstance(tree, Mapping) else ws.tree`) would answer
    "you named no tree" and "the tree you named is not one" identically, so a caller who asked
    for a specific measurement and got the argument wrong would be told about a different tree,
    with a green exit code and nothing raised.

    The cases differ in what the caller INTENDED, even though the fallback is harmless in one:

    - a named tree is the postamble's case, and a model's mid-loop call once `coding_bind` has
      bound the workspace: the recorded op carries what was measured, and a predicate that
      quietly measured something else would make the ledger's `passed` a claim about a tree it
      did not commit. The model names none itself, since its declaration takes no arguments;
    - no tree is a direct caller meaning the workspace it built. Legitimate, and kept;
    - a malformed tree REFUSES, because the caller had a specific measurement in mind and
      answering about another one is worse than not answering.

    **The third case is removed rather than enumerated.** `SuiteArgs` declares
    `tree: dict[str, str] | None`, so validating against the tool's own declared argument type
    turns "not a mapping" into a refusal at the boundary and leaves a union of exactly two for
    the decision below. A denylist arm for the third shape would restate, by exclusion, what the
    declaration already says.

    Matched on SHAPE rather than truthiness: `{}` is a real request (measure an empty tree, which
    the predicate answers with exit 5 and `CommandRun.green` already refuses to read as passing),
    and an arm testing `if tree:` would turn "measure nothing" into "measure everything"."""
    try:
        named: dict[str, str] | None = SuiteArgs.model_validate(dict(args)).tree
    except ValidationError as malformed:
        raise ToolError(
            f"run_suite was given a `tree` that is not a mapping of path to content — refusing "
            f"rather than measuring the workspace, which would answer about a different tree "
            f"than the one you named: {malformed}"
        ) from malformed
    match named:
        case None:
            return run_suite(_materializable(ws.tree))
        case tree:
            return run_suite(_materializable(tree))


type Runner = Callable[[Workspace, Mapping[str, Any]], Any]

TOOLS: dict[str, Runner] = {
    "read_file": _read_file,
    "list_dir": _list_dir,
    "write_file": _write_file,
    "check": _check,
    "run_suite": _run_suite,
    "structural_apply": _structural_apply,
    "semantic_rename": _semantic_rename,
    "semantic_references": _semantic_references,
    "declared_arms": _declared_arms,
}
"""Every tool the machine can call, and every one has a body here.

The table IS the repertoire's provided side — `provides(Pi)` in the repertoire axis's vocabulary,
though the typed `Tool`/`Provide` pair stays unbuilt. `machine.agent_worker(needs=...)` names a
subset of these keys, and `test_every_tool_in_the_table_has_a_runner` is what stops a state's
declared repertoire and a deployment's served set from drifting apart."""


# --- the table, DECLARED: name, arguments, result, and how a result reads in a transcript ------
#
# `TOOLS` above is the PROVIDED side — what a deployment serves. This is the DECLARED side: what
# each name means, bound once instead of restated at every construction site. The `result`
# column shows why: five of these nine return something that is not a string.
#
# The two columns are read by different consumers. `result` is what the OP
# carries, so the workflow folds it and a replay re-derives it; `observe` is what the TRANSCRIPT
# sees, so it stays short enough to sit in every subsequent prompt. A tree in `content` would be
# spliced into the context on every turn after the edit.


class PathArgs(BaseModel):
    path: str


class NoArgs(BaseModel):
    """A tool that takes nothing. Declared rather than left as a bare `dict`, so a model that
    invents arguments for `list_dir` is told so instead of having them silently dropped: a
    `tree` named for `run_suite` among them, which `coding_bind` would otherwise replace."""

    model_config = ConfigDict(extra="forbid")


class WriteFileArgs(BaseModel):
    path: str
    content: str


class StructuralApplyArgs(BaseModel):
    path: str
    rules: list[dict[str, Any]]


class PositionArgs(BaseModel):
    path: str
    line: int
    column: int


class RenameArgs(PositionArgs):
    new_name: str


class DeclaredArmsArgs(BaseModel):
    path: str
    alias: str


class SuiteArgs(BaseModel):
    """`tree` is optional and its absence MEANS something — see `_run_suite`."""

    tree: dict[str, str] | None = None


def _tree_summary(tree: Mapping[str, str]) -> str:
    """An edit's observation: what the tree now holds, not the tree itself.

    The whole tree would be correct and ruinous — it rides in every prompt after the edit, and
    the model already knows what it just wrote. What it cannot know without being told is whether
    the write landed and what else is there."""
    return (
        f"wrote the file; the workspace now holds {len(tree)} file(s): {', '.join(sorted(tree))}"
    )


def _suite_summary(run: CommandRun) -> str:
    if run.collection_error is not None:
        return f"the suite could not collect: {run.collection_error}"
    failing = ", ".join(run.failures) if run.failures else "nothing"
    return f"exit {run.exit_code}; failing: {failing}"


CODING_TOOLS: dict[str, Tool[Any, Any]] = {
    "read_file": Tool("read_file", PathArgs, str, lambda text: text),
    "list_dir": Tool("list_dir", NoArgs, list[str], lambda names: ", ".join(names) or "(empty)"),
    "write_file": Tool("write_file", WriteFileArgs, dict[str, str], _tree_summary),
    "check": Tool("check", NoArgs, str, lambda verdict: verdict),
    "run_suite": Tool("run_suite", NoArgs, CommandRun, _suite_summary),
    "structural_apply": Tool(
        "structural_apply", StructuralApplyArgs, dict[str, str], _tree_summary
    ),
    "semantic_rename": Tool("semantic_rename", RenameArgs, str, lambda diff: diff),
    "semantic_references": Tool(
        "semantic_references",
        PositionArgs,
        list[str],
        lambda sites: f"{len(sites)} reference(s): {', '.join(sites)}" if sites else "none found",
    ),
    "declared_arms": Tool(
        "declared_arms",
        DeclaredArmsArgs,
        list[str],
        lambda arms: ", ".join(arms) or "no arms declared",
    ),
}
"""The declared side of the same nine names `TOOLS` serves.

Kept in step by `test_the_declared_table_and_the_served_table_are_the_same_names` — two tables
over one set of names is exactly the drift `--role-coverage` and the deleted `StateSpec.needs`
have each taught this repo about, and a declaration that quietly covers eight of nine is worse
than none, because it is read on trust."""


def serve_tool(name: str, args: Mapping[str, Any], *, project_root: str | None = None) -> Any:
    """Dispatch one tool call AT THE OP BOUNDARY, where a refusal has to be a value.

    This is what a deployment's `DomainInterpreter` calls, and it differs from `run_tool` in
    returning the refusal. A `ToolError` raised here would be handler-side: it never crosses the
    yield boundary into the workflow, so no `except` in workflow code can see it, the op never
    completes, and the engine retries a refusal that is deterministic by construction. Measured on
    SQLite with the static gate refusing a bad edit: three identical attempts, zero ledger rows,
    the run dead on the exact answer the edit ladder exists to produce.

    Returned as a `ToolRefused`, the op completes and the answer is checkpointed, so a replay
    re-serves it and a judge gets to see it. `Tool.observe` renders it for a transcript; a
    mechanical judge sees it in the next measurement, which is the referee anyway.

    **`UnknownTool` still raises, and by construction rather than by an exemption**: it is a
    `KeyError`, not a `ToolError`, so it passes this clause untouched. An unserved name is a
    composition defect — the model named something the deployment does not have — and answering it
    with a diagnostic would let a run look busy while doing nothing.

    **The workspace is built here, from the tree the call carries**, so a deployment serving the
    table holds none, and a worker resumed on a fresh process acts on the tree the record
    describes (`coding.specs.coding_bind` binds it). A tree whose key no tool can write raises
    `UnsafeTreePath` rather than returning a refusal: every tree a tool returns has passed that
    check, so an unsafe one is the caller's seed, which no model can repair.

    `run_tool` stays as it is: it is the direct API, it is what the edit tests drive, and "this
    call raises on failure" is the right contract everywhere the caller is not an op."""
    try:
        workspace = Workspace(tree=args.get("tree", {}), project_root=project_root)
        return run_tool(workspace, name, args)
    except ToolError as refused:
        return ToolRefused(refused=True, tool=name, diagnostic=str(refused))


def run_tool(ws: Workspace, name: str, args: Mapping[str, Any]) -> Any:
    """Dispatch one tool call. Unknown names raise — see `UnknownTool`.

    Raises on a tool's refusal too. At an OP boundary that is wrong — see `serve_tool`."""
    runner = TOOLS.get(name)
    if runner is None:
        raise UnknownTool(f"no runner for {name!r}; the table serves {sorted(TOOLS)}")
    return runner(ws, args)
