"""The coder's four tools: read, edit, write and bash, and the suite that judges the work.

Each tool is a function of its arguments, and the project tree is one of them. The worker binds the
tree that the recorded results so far describe, so nothing is kept between calls and a run resumed
on a fresh process sees the same files. `edit` and `write` lint what they would produce and refuse
a change that does not pass; `bash` and the suite run in the pinned container with no network, and
refuse when the image is absent.

An argument model's docstring and field descriptions are what the model reads about its tool.

An observation is a `Template`, and every hole carrying content the tools READ declares `:data`,
so it reaches the model inside a block its own bytes cannot close. The coder's own notices stay
outside that block: a notice inside one is indistinguishable from a line of the file imitating it.
"""

from collections.abc import Callable, Mapping
from string.templatelib import Template
from typing import Any, assert_never

from pydantic import BaseModel, ConfigDict, Field

from effective.channels import render
from effective.coding.edits.text import EditRefused, TextEdit, apply_edits
from effective.coding.gate import static_diagnostic
from effective.coding.runners import run_suite
from effective.coding.tier import (
    Tier,
    TierUnavailable,
    UnsafeTreePath,
    run_tree_command,
    tree_paths,
)
from effective.domain import CallTool, ToolRefused
from effective.machine.trampoline import SUITE_TOOL
from effective.react import Tool
from effective.skills import DISCLOSE_TOOL, SkillRegistry

type Tree = Mapping[str, str]

MAX_LINES = 2000
MAX_BYTES = 50_000
BASH_TIMEOUT = 60
BASH_TIMEOUT_MAX = 300


class ReadArgs(BaseModel):
    """Read a file. Output stops at 2000 lines or 50 KB; use offset and limit for the rest."""

    path: str = Field(description="Path of the file, relative to the project root.")
    offset: int | None = Field(default=None, description="First line to read, counting from 1.")
    limit: int | None = Field(default=None, description="Most lines to read.")


class Replacement(BaseModel):
    """One exact replacement."""

    old_text: str = Field(
        description="Exact text to replace. It must occur once in the original file and must not "
        "overlap another edit's old_text."
    )
    new_text: str = Field(description="The text to put in its place.")


class EditArgs(BaseModel):
    """Edit one file by exact text replacement. Every edits[].old_text is matched against the
    original file, not against the result of an earlier edit."""

    path: str = Field(description="Path of the file, relative to the project root.")
    edits: list[Replacement] = Field(
        description="One or more disjoint replacements. Merge changes to nearby lines into one."
    )


class WriteArgs(BaseModel):
    """Create a file, or replace a whole file's content."""

    path: str = Field(description="Path of the file, relative to the project root.")
    content: str = Field(description="The file's complete new content.")


class BashArgs(BaseModel):
    """Run a shell command in a fresh container over a copy of the project, with no network.
    Returns the exit code and the last 2000 lines or 50 KB of output. Changes the command makes to
    files are discarded; change files with edit or write."""

    command: str = Field(description="The command, run by bash from the project root.")
    timeout: int | None = Field(
        default=None, description="Seconds before the command is stopped: 60 if null, 300 at most."
    )


def text(template: Template) -> str:
    """One template as the bytes a caller sends: the flatten, at the boundary and nowhere else."""
    return "".join(message.content for message in render(template, output=str).messages)


class Viewed(BaseModel):
    """What `read` produced: the file's own text, and the coder's notice about what was shown."""

    model_config = ConfigDict(frozen=True)

    text: str
    notice: str = ""


class Ran(BaseModel):
    """What `bash` produced: the command's exit code and output, and the coder's notice."""

    model_config = ConfigDict(frozen=True)

    exit_code: int
    output: str
    notice: str = ""


class Changed(BaseModel):
    """The project after an edit or a write, and the file that changed."""

    model_config = ConfigDict(frozen=True)

    path: str
    tree: dict[str, str]


class Refused(Exception):
    """A call a tool will not carry out. Its message is what the model reads next."""


def _file(tree: Tree, path: str) -> str:
    if path not in tree:
        raise Refused(f"There is no file {path}. The project has: {', '.join(sorted(tree))}.")
    return tree[path]


def _head(line: str, budget: int) -> str:
    """The first `budget` bytes of `line`, cut at a character boundary."""
    return line.encode()[:budget].decode(errors="ignore")


def read(tree: Tree, args: ReadArgs) -> Viewed:
    lines = _file(tree, args.path).splitlines(keepends=True)
    first = args.offset or 1
    if first < 1 or (args.limit is not None and args.limit < 1) or first > max(len(lines), 1):
        raise Refused(
            f"offset {args.offset} and limit {args.limit} select nothing in {args.path}, which "
            f"has {len(lines)} lines."
        )
    wanted = lines[first - 1 : None if args.limit is None else first - 1 + args.limit]
    shown: list[str] = []
    size = 0
    capped = False
    for line in wanted[:MAX_LINES]:
        length = len(line.encode())
        if size + length > MAX_BYTES:
            # The budget binds INSIDE a line as well as between lines. Taken between them only,
            # one long line carried 62 KB past a 50 KB cap into the prompt and the checkpoint.
            if not shown:
                clipped = t"\n[Line {first} is {length} bytes; showing its first {MAX_BYTES}.]"
                return Viewed(text=_head(line, MAX_BYTES), notice=text(clipped))
            capped = True
            break
        shown.append(line)
        size += length
    last = first - 1 + len(shown)
    if last >= len(lines):
        return Viewed(text="".join(shown))
    limit = t" ({MAX_BYTES // 1000} KB limit)" if capped else t""
    paging = (
        t"\n[Showing lines {first}-{last} of {len(lines)}{limit}. "
        t"Use offset={last + 1} to continue.]"
    )
    return Viewed(text="".join(shown), notice=text(paging))


def _changed(tree: Tree, path: str, content: str) -> Changed:
    edited = {**tree, path: content}
    tree_paths(edited)
    if (diagnostic := static_diagnostic(content, path)) is not None:
        raise Refused(f"{path} was left unchanged: {diagnostic}")
    return Changed(path=path, tree=edited)


def edit(tree: Tree, args: EditArgs) -> Changed:
    replacements = [TextEdit(r.old_text, r.new_text) for r in args.edits]
    return _changed(tree, args.path, apply_edits(args.path, _file(tree, args.path), replacements))


def write(tree: Tree, args: WriteArgs) -> Changed:
    return _changed(tree, args.path, args.content)


def _tail(output: str) -> tuple[str, str]:
    """The end of a command's output within both caps, and the notice about what was cut.

    The notice comes back BESIDE the text rather than prepended to it, so it can stay outside the
    data fence, where output imitating a notice cannot forge one."""
    lines = output.splitlines(keepends=True)
    kept: list[str] = []
    size = 0
    for line in reversed(lines[-MAX_LINES:]):
        length = len(line.encode())
        if size + length > MAX_BYTES:
            if not kept:
                kept.append(line.encode()[-MAX_BYTES:].decode(errors="ignore"))
                size = MAX_BYTES
            break
        kept.append(line)
        size += length
    if len(kept) == len(lines) and size == len(output.encode()):
        return output, ""
    notice = t"[Showing the last {size} bytes of {len(lines)} lines.]\n"
    return "".join(reversed(kept)), text(notice)


def bash(tree: Tree, args: BashArgs) -> Ran:
    timeout = min(args.timeout or BASH_TIMEOUT, BASH_TIMEOUT_MAX)
    command = ["bash", "-c", args.command]
    exit_code, output = run_tree_command(tree, command, timeout=timeout, tier=Tier.CONTAINER)
    shown, notice = _tail(output)
    return Ran(exit_code=exit_code, output=shown, notice=notice)


def _changed_observation(changed: Changed) -> Template:
    """The path is a data hole because a model chooses it and `tree_path` admits a newline."""
    return t"Updated:\n{changed.path:data}"


def _viewed_observation(viewed: Viewed) -> Template:
    return t"{viewed.text:data}{viewed.notice}"


def _ran_observation(ran: Ran) -> Template:
    if not ran.output:
        return t"exit code {ran.exit_code}\n(no output)"
    return t"exit code {ran.exit_code}\n{ran.notice}{ran.output:data}"


TOOLS: dict[str, Tool[Any, Any]] = {
    "read": Tool("read", ReadArgs, Viewed, observe=_viewed_observation),
    "edit": Tool("edit", EditArgs, Changed, observe=_changed_observation),
    "write": Tool("write", WriteArgs, Changed, observe=_changed_observation),
    "bash": Tool("bash", BashArgs, Ran, observe=_ran_observation),
}

RUNNERS: dict[str, Callable[[Tree, Any], Any]] = {
    "read": read,
    "edit": edit,
    "write": write,
    "bash": bash,
}

REFUSALS = (Refused, EditRefused, UnsafeTreePath, TierUnavailable)
"""What a tool raises when it will not carry out a call: each becomes a `ToolRefused` the model
reads. Anything else is a defect and fails the op."""


def serve(op: CallTool[Any]) -> Any:
    """Carry out one tool call from its arguments alone."""
    match op:
        case CallTool(name=name, args={"tree": tree}) if name == SUITE_TOOL:
            try:
                return run_suite(tree, tier=Tier.CONTAINER)
            except TierUnavailable as refused:
                return ToolRefused(refused=True, tool=name, diagnostic=str(refused))
        case CallTool(name=name, args={"tree": tree, **given}) if name in RUNNERS:
            args = TOOLS[name].args.model_validate(given)
            try:
                return RUNNERS[name](tree, args)
            except REFUSALS as refused:
                return ToolRefused(refused=True, tool=name, diagnostic=str(refused))
        case CallTool(name=name):
            # An observation rather than an exception, for the reason every other refusal here is
            # one: a model that invents a name should read what there is and choose again, where a
            # raised error ends the run. The first live run named `functions.edit`.
            return ToolRefused(
                refused=True,
                tool=name,
                diagnostic=text(
                    t"There is no tool named {name}. The tools are: {', '.join(sorted(RUNNERS))}."
                ),
            )
        case unreachable:
            assert_never(unreachable)


def serving(skills: SkillRegistry) -> Callable[[CallTool[Any]], Any]:
    """`serve`, answering a skill's disclosure from `skills` as well."""

    def served(op: CallTool[Any]) -> Any:
        match op:
            case CallTool(name=name, args={"skill": named}) if name == DISCLOSE_TOOL:
                return skills.disclose(named)
            case _:
                return serve(op)

    return served
