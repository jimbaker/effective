"""SkillsBench harness that drives the task containers itself: a three-arm skills run.

BenchFlow's only custom-agent plane is an ACP server running *inside* the task container (the
in-process ``import_path`` schema belongs to Harbor, which the SkillsBench paper, Li et al. 2026,
arXiv:2602.12670, used and the repo does not ship). An in-container agent cannot host the 3.14
substrate or the host-side pin/span machinery, so this harness **drives the containers**:
``run_agent`` runs here, under effective's handlers, with the telemetry and pin machinery intact;
the agent's one tool is a shell exec into the task container (podman); the task's own
``verifier/test.sh`` runs in-container and writes ``/logs/verifier/reward.txt``, so scoring is
the benchmark's contract.

| arm       | skills delivery                                                                     |
|-----------|-------------------------------------------------------------------------------------|
| `absent`  | none: the floor                                                                     |
| `inline`  | every SKILL.md body in the system prompt (the paper's inline-all "with-skill"       |
|           | delivery), plus the skill files copied to ``/skills`` in-container so               |
|           | ``references/`` and ``scripts/`` are usable, as their harness adapters provide      |
| `catalog` | the registry *catalog* in the system prompt plus an ``activate_skill`` tool; a      |
|           | disclosure is an explicit, recorded act returning a ``Pin``, and the span carries   |
|           | ``skill.*`` fields, which separate skill-present from skill-consulted               |

Bench tooling: podman/pyyaml are invoked lazily so the substrate's lean-dep
policy holds (pyyaml is a dev-group dependency; the module imports it only when
a task is loaded).
"""

import json
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_core import to_jsonable_python

from agent.runtime import LocalCtx, make_tool_runner
from agent.tasks import task_scope
from effective.api import scoped
from effective.channels import Prompt, Repair, TypedField
from effective.channels import render as channel_render
from effective.cost import CostBudget, MeteredInterpreter, Usage
from effective.envelope import Envelope, JsonEnvelope
from effective.handlers.absurd import DurableHandler
from effective.interpreters.openai import OpenAITurnCaller, to_openai_messages, usage_from_openai
from effective.react import AssistantTurn, ToolRequest, ToolResult, Trajectory, run_agent
from effective.skills import Pin, SkillRegistry
from effective.telemetry import Span, measurements, multi, otlp_jsonl_sink, traced

ARMS = ("absent", "inline", "catalog")


@dataclass(frozen=True)
class TaskSpec:
    """One SkillsBench task, parsed from its ``task.md`` (schema 1.3)."""

    task_id: str
    path: Path
    prompt: str
    agent_timeout_sec: float
    verifier_timeout_sec: float
    cpus: float
    memory_mb: int
    workdir: str

    @property
    def environment(self) -> Path:
        return self.path / "environment"

    @property
    def skills_dir(self) -> Path:
        return self.environment / "skills"


def load_task(task_dir: str | Path) -> TaskSpec:
    """Parse ``task.md`` frontmatter + prompt body. Honors their per-task
    timeouts — the protocol's comparability requirement."""
    import yaml  # dev-group dep; bench tooling only

    task_dir = Path(task_dir)
    text = (task_dir / "task.md").read_text()
    if not text.startswith("---"):
        raise ValueError(f"{task_dir}: task.md has no frontmatter")
    _, fm, body = text.split("---", 2)
    meta = yaml.safe_load(fm)
    env = meta.get("environment", {})
    return TaskSpec(
        task_id=task_dir.name,
        path=task_dir,
        prompt=body.strip(),
        agent_timeout_sec=float(meta.get("agent", {}).get("timeout_sec", 900.0)),
        verifier_timeout_sec=float(meta.get("verifier", {}).get("timeout_sec", 900.0)),
        cpus=float(env.get("cpus", 1)),
        memory_mb=int(env.get("memory_mb", 4096)),
        workdir=str(env.get("workdir", "/app")),
    )


def _sh(args: list[str], *, timeout: float | None = None, check: bool = False) -> Any:
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=check)


def _qualified_context(environment: Path, tmp: Path) -> Path:
    """Copy the build context and qualify bare ``FROM image:tag`` lines to
    ``docker.io/library/`` — podman short-name resolution has no search list on
    this host, and the tasks all use official single-name images."""
    ctx = tmp / "context"
    shutil.copytree(environment, ctx)
    dockerfile = ctx / "Dockerfile"
    lines = []
    for line in dockerfile.read_text().splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            image = stripped.split()[1]
            if "/" not in image and not image.startswith("docker.io"):
                line = line.replace(image, f"docker.io/library/{image}", 1)
        lines.append(line)
    dockerfile.write_text("\n".join(lines) + "\n")
    return ctx


class TaskContainer:
    """One task's container under podman: build, run (their cpu/mem limits),
    exec (the agent's shell), verify (their contract), destroy."""

    def __init__(self, task: TaskSpec) -> None:
        self.task = task
        self.tag = f"sb-{task.task_id}"
        self.cid: str | None = None

    def build(self, *, timeout: float = 600.0) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ctx = _qualified_context(self.task.environment, Path(tmp))
            r = _sh(["podman", "build", "-q", "-t", self.tag, str(ctx)], timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"build failed for {self.task.task_id}: {r.stderr[-800:]}")

    def start(self) -> None:
        r = _sh(
            [
                "podman",
                "run",
                "-d",
                "--rm",
                f"--cpus={self.task.cpus}",
                f"--memory={self.task.memory_mb}m",
                self.tag,
                "sleep",
                "infinity",
            ],
            timeout=60,
        )
        if r.returncode != 0:
            raise RuntimeError(f"run failed for {self.task.task_id}: {r.stderr[-800:]}")
        self.cid = r.stdout.strip()

    def exec(self, command: str, *, timeout: float = 120.0, user: str | None = None) -> str:
        """Run a shell command in the container; the agent's observation is the
        combined output + exit code (truncated — LLM context, not a log)."""
        assert self.cid, "container not started"
        args = ["podman", "exec"]
        if user:
            args += ["--user", user]
        args += ["--workdir", self.task.workdir, self.cid, "bash", "-c", command]
        try:
            r = _sh(args, timeout=timeout)
        except subprocess.TimeoutExpired:
            return f"[command timed out after {timeout:.0f}s]"
        out = (r.stdout or "") + (("\n[stderr]\n" + r.stderr) if r.stderr.strip() else "")
        if len(out) > 8000:
            out = out[:4000] + f"\n...[{len(out) - 8000} chars elided]...\n" + out[-4000:]
        return f"{out.strip() or '(no output)'}\n[exit code: {r.returncode}]"

    def copy_in(self, host_path: Path, container_path: str) -> None:
        assert self.cid, "container not started"
        _sh(["podman", "cp", str(host_path), f"{self.cid}:{container_path}"], check=True)

    def run_verifier(self) -> float:
        """Their contract: ``/verifier/test.sh`` runs in-container and writes a
        scalar to ``/logs/verifier/reward.txt``. Copy the verifier in, run it,
        read the reward out. A broken verifier run scores 0.0 (their semantics:
        no reward file means no reward)."""
        assert self.cid, "container not started"
        self.copy_in(self.task.path / "verifier", "/verifier")
        _sh(
            ["podman", "exec", self.cid, "bash", "/verifier/test.sh"],
            timeout=self.task.verifier_timeout_sec,
        )
        r = _sh(["podman", "exec", self.cid, "cat", "/logs/verifier/reward.txt"], timeout=30)
        try:
            return float(r.stdout.strip())
        except ValueError:
            return 0.0

    def stop(self) -> None:
        if self.cid:
            _sh(["podman", "rm", "-f", self.cid], timeout=60)
            self.cid = None


# --- the bench wire turn ------------------------------------------------------
# The wire alphabet is an explicit, swappable seam (effective.envelope — the
# heterogeneous-alphabet principle). History, measured: JSON-in-JSON args
# corrupted multiline shell commands (the model authors the inner escaping);
# flat strict-schema fields still lost 27/120 trials to two-objects-per-
# completion; the envelope seam makes the surviving convention a VALUE the
# harness can ablate — JsonEnvelope (lenient flat JSON) vs SectionEnvelope
# (sentinel sections, raw bodies: a heredoc needs no escaping at all).


class _BenchTurn(BaseModel):
    """One bench turn: act with bash / activate_skill, or answer to finish."""

    thought: str | None = None
    tool: str | None = None
    command: str | None = None
    skill: str | None = None
    answer: str | None = None


def _turn_prompt() -> Prompt[_BenchTurn]:
    """The bench turn as a t-string channel template — the PEP 750 seam this
    harness exists to showcase. The `Template` is the homoiconic structure;
    the wire alphabet is late-bound (an `Envelope` at the caller), and both
    alphabets resolve through this one typed `Prompt`. This is the capability
    eager string-assembly adapters can't express: their format logic is welded
    into the prompt-building; here the format is a value applied at render."""
    thought = TypedField(str | None)
    tool = TypedField(str | None)
    command = TypedField(str | None)
    skill = TypedField(str | None)
    answer = TypedField(str | None)
    return channel_render(
        t"{thought}{tool}{command}{skill}{answer}",
        output=_BenchTurn,
    )


TURN_PROMPT = _turn_prompt()

_NULLISH = frozenset({"", "null", "none", "n/a", "-"})


def _resolve_turn(raw: Mapping[str, Any]) -> _BenchTurn:
    """Envelope output -> the typed turn, through ``Prompt.resolve``. Section
    bodies arrive as raw text, so nullish spellings normalize to None; absent
    sections fill as None (a bench loses turns, not trials)."""
    filled: dict[str, Any] = {}
    for name in TURN_PROMPT.channels:
        value = raw.get(name)
        if isinstance(value, str) and value.strip().lower() in _NULLISH:
            value = None
        filled[name] = value
    turn = TURN_PROMPT.resolve(filled)
    if isinstance(turn, Repair):
        raise ValueError(f"unresolvable turn: {turn.reason}")
    if turn.tool is not None:
        turn = turn.model_copy(update={"tool": turn.tool.strip().lower()})
    return turn


class BenchTurnCaller(OpenAITurnCaller):
    """The OpenAI caller parameterized by a response ``Envelope``.

    The envelope owns the answer-format contract (the harness appends its
    ``instructions`` to the system prompt) and the completion parse. One retry
    per turn; ``wire_failures`` counts first-parse failures so the wire
    ablation sees turn-level quality, not just trial survival."""

    def __init__(self, *args: Any, envelope: Envelope | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.envelope: Envelope = envelope if envelope is not None else JsonEnvelope()
        self.wire_failures = 0

    def __call__(self, op: Any) -> tuple[AssistantTurn, Usage]:
        try:
            return self._call_once(op)
        except Exception:
            self.wire_failures += 1
            return self._call_once(op)

    def _call_once(self, op: Any) -> tuple[AssistantTurn, Usage]:
        messages = [
            {"role": "system", "content": self.system_prompt},
            *to_openai_messages(op.messages),
        ]
        kwargs: dict[str, Any] = dict(self.extra)
        if isinstance(self.envelope, JsonEnvelope):
            kwargs["response_format"] = {"type": "json_object"}
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_completion_tokens=self.max_completion_tokens,
            **kwargs,
        )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("empty completion (refusal or truncation)")
        turn = _resolve_turn(self.envelope.parse(content))
        usage = replace(
            usage_from_openai(response, self.price), latency_s=time.perf_counter() - start
        )
        tool = None
        if turn.tool == "bash":
            tool = ToolRequest(name="bash", args={"command": turn.command or ""})
        elif turn.tool == "activate_skill":
            tool = ToolRequest(name="activate_skill", args={"skill": turn.skill or ""})
        return AssistantTurn(thought=turn.thought or "", tool=tool, answer=turn.answer), usage


# --- the three arms ----------------------------------------------------------

_PROTOCOL = """You are an autonomous engineer working inside a Linux container.
Work in {workdir}. Complete the task exactly as specified; create every file the
task names, then verify them before finishing. Each turn you give: `thought`
(brief reasoning) and EITHER an action (`tool` plus its argument) OR a final
`answer`. Actions:
  tool: bash, command: <shell command>  -> run it; you get output + exit code
Prefer non-interactive commands. Write files with heredocs (cat <<'EOF' > file).
FINISH (answer instead of tool) only after the task's files exist and you have
verified them with bash — if work remains, you MUST act this turn.

{format_contract}"""

_CATALOG_TOOL = """
You also have:
  tool: activate_skill, skill: <name>  -> disclose the full instructions of a named skill

Available skills (name: description):
{catalog}

Skill files (references/, scripts/) are under /skills/<name>/ in the container.
Activate a skill BEFORE relying on it; activate only skills relevant to the task."""

_INLINE_SKILLS = """
## Skills

The following skill instructions apply to this task. Their files (references/,
scripts/) are under /skills/<name>/ in the container.

{bodies}"""


def system_prompt(
    arm: str, task: TaskSpec, registry: SkillRegistry | None, envelope: Envelope | None = None
) -> str:
    envelope = envelope if envelope is not None else JsonEnvelope()
    contract = envelope.instructions(TURN_PROMPT.channels)
    base = _PROTOCOL.format(workdir=task.workdir, format_contract=contract)
    if arm == "absent" or registry is None:
        return base
    if arm == "inline":
        bodies = "\n\n".join(
            f"### Skill: {name}\n\n{registry.disclose(name).body}" for name in registry.names()
        )
        return base + _INLINE_SKILLS.format(bodies=bodies)
    if arm == "catalog":
        return base + _CATALOG_TOOL.format(catalog=registry.index())
    raise ValueError(f"unknown arm {arm!r} (use {ARMS})")


class PinToolResult(ToolResult):
    """A tool observation that *is* a disclosure: the body rides ``content``
    (what the model reads); ``name``/``content_hash`` make the result pin-shaped
    so the traced span carries ``skill.*`` attribution (verified use)."""

    name: str
    content_hash: str


def make_tools(
    container: TaskContainer,
    registry: SkillRegistry | None,
    pins: list[Pin],
    *,
    exec_timeout: float = 120.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(tools, agents) for ``make_tool_runner``: the bash exec, and — when a
    registry is present (catalog arm) — the activation tool, which records the
    pin and returns the disclosed body."""

    def bash(args: dict[str, Any]) -> str:
        return container.exec(str(args.get("command", "")), timeout=exec_timeout)

    tools: dict[str, Any] = {"bash": bash}
    agents: dict[str, Any] = {}
    if registry is not None:

        def activate(op: Any) -> ToolResult:
            name = str(op.args.get("skill", ""))
            try:
                pin = registry.disclose(name)
            except LookupError:
                known = ", ".join(registry.names())
                return ToolResult(content=f"unknown skill {name!r}; available: {known}")
            pins.append(pin)
            return PinToolResult(content=pin.body, name=pin.name, content_hash=pin.content_hash)

        agents["activate_skill"] = activate
    return tools, agents


# --- one trial ----------------------------------------------------------------


@dataclass
class TrialResult:
    task_id: str
    arm: str
    attempt: int
    reward: float
    stop_reason: str
    steps: int
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cost_usd: float
    elapsed_sec: float
    pins: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    wire: str = "json"
    wire_failures: int = 0


def run_trial(
    task: TaskSpec,
    arm: str,
    attempt: int,
    *,
    make_caller: Callable[[str], Any],
    out_dir: Path,
    max_iters: int = 24,
    budget_usd: float = 0.25,
    exec_timeout: float = 120.0,
    envelope: Envelope | None = None,
) -> TrialResult:
    """One (task, arm, attempt) trial: fresh container, the arm's prompt/tools,
    the agent loop under a wall-clock cap (their ``agent.timeout_sec``), then
    their verifier. ``make_caller(system_prompt)`` builds the LLM caller — the
    model column AND the wire alphabet (``envelope``) are injected; the harness
    is model- and format-agnostic."""
    trial_dir = out_dir / task.task_id / arm / f"attempt-{attempt}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    registry = None
    if arm in ("inline", "catalog") and task.skills_dir.is_dir():
        registry = SkillRegistry.load(task.skills_dir)

    container = TaskContainer(task)
    pins: list[Pin] = []
    spans: list[Span] = []
    sid = f"{arm}:{task.task_id}:{attempt}"
    sink = multi(otlp_jsonl_sink(trial_dir / "spans.jsonl"), spans.append)
    started = time.monotonic()
    reward, error, traj = 0.0, None, None
    interp: MeteredInterpreter | None = None
    caller: Any = None
    try:
        container.build()
        container.start()
        if registry is not None:
            container.copy_in(task.skills_dir, "/skills")
        tools, agents = make_tools(container, registry, pins, exec_timeout=exec_timeout)
        caller = make_caller(system_prompt(arm, task, registry, envelope))
        interp = MeteredInterpreter(
            llm=caller,
            tools=make_tool_runner(tools, agents),
            budget=CostBudget(budget_usd),
            domain_layers=[traced(sink, session_id=sid, agent_name=f"effective:{arm}")],
        )
        handler = DurableHandler(LocalCtx(), interp)

        result: list[Any] = [None]
        errors: list[BaseException] = []

        def drive() -> None:
            try:
                result[0] = handler.run(
                    lambda: scoped(
                        # `sid` is the TELEMETRY session id and stays one; it is a composite
                        # (`{arm}:{task_id}:{attempt}`) built by an f-string, so splicing it here
                        # flattened three fields into one hole — and `compose_key` refuses a bare
                        # `str` outright, so this line could not execute at all. The ctx is a
                        # fresh `LocalCtx()` per trial, so the task id alone names the scope.
                        task_scope(task.task_id),
                        lambda: run_agent(task.prompt, max_iters=max_iters),
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        worker = threading.Thread(target=drive, daemon=True)
        worker.start()
        worker.join(task.agent_timeout_sec)
        if worker.is_alive():
            error = f"agent timeout ({task.agent_timeout_sec:.0f}s, their protocol cap)"
        elif errors:
            error = f"{type(errors[0]).__name__}: {errors[0]}"
        else:
            traj = Trajectory.model_validate(result[0])
        reward = container.run_verifier()
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"
    finally:
        container.stop()

    meter = interp.meter if interp is not None else Usage()
    trial = TrialResult(
        task_id=task.task_id,
        arm=arm,
        attempt=attempt,
        reward=reward,
        stop_reason=traj.stop_reason if traj else "error",
        steps=len(traj.steps) if traj else 0,
        prompt_tokens=meter.prompt_tokens,
        completion_tokens=meter.completion_tokens,
        cache_read_tokens=meter.cache_read_input_tokens,
        cost_usd=meter.cost,
        elapsed_sec=time.monotonic() - started,
        pins=[{"name": p.name, "content_hash": p.content_hash} for p in pins],
        error=error,
        wire=type(envelope).__name__ if envelope is not None else "JsonEnvelope",
        wire_failures=getattr(caller, "wire_failures", 0),
    )
    (trial_dir / "trial.json").write_text(json.dumps(asdict(trial), indent=1))
    if traj is not None:
        (trial_dir / "trajectory.json").write_text(json.dumps(to_jsonable_python(traj), indent=1))
    # The spans were being collected and never read — two references, both writes. `measurements`
    # is the consumer they lacked: it folds them into `{placed key: (cost, duration_ns)}`, which is
    # what `TrialResult.cost_usd` structurally cannot say. The meter is a SCALAR total, so a trial
    # that cost twice as much could not tell you WHERE; this attributes every dollar and nanosecond
    # to the op address that spent it, which is the join the telemetry arc built.
    #
    # In-process `Span` objects rather than the sidecar beside them, deliberately: `measurements`
    # takes the dataclasses and needs no dialect, where `sidecar_measurements` would re-parse what
    # we already hold and would depend on the wire format by name.
    (trial_dir / "measurements.json").write_text(json.dumps(measurements_record(spans), indent=1))
    return trial


def measurements_record(spans: Iterable[Span]) -> dict[str, dict[str, float | int | None]]:
    """A trial's spans folded to `{placed key: {cost_usd, duration_ns}}`, ready for `json.dumps`.

    Named rather than inlined so it can be tested without a container — `run_trial` needs a live
    image, so anything spelled inside it ships unpinned.

    **`cost_usd` may be `None`, and that is the point of the shape.** A TOOL span carries no cost
    attribute at all, and writing `0.0` there would read as "this node was free" — the lie
    `Node.cost` stopped telling when it stopped being a `float`. `measurements` keeps the two
    apart and this record does not collapse them.
    """
    return {
        key: {"cost_usd": usd, "duration_ns": ns}
        for key, (usd, ns) in sorted(measurements(spans).items())
    }


def summarize(trials: list[TrialResult]) -> dict[str, Any]:
    """Their pass-rate shape per (task, arm) plus the cost columns our protocol
    adds; ``pass`` = reward 1.0 (their pass@1 convention on binary verifiers)."""
    by: dict[tuple[str, str], list[TrialResult]] = {}
    for t in trials:
        by.setdefault((t.task_id, t.arm), []).append(t)
    rows = []
    for (task_id, arm), ts in sorted(by.items()):
        rows.append(
            {
                "task": task_id,
                "arm": arm,
                "n": len(ts),
                "pass_rate": sum(1 for t in ts if t.reward >= 1.0) / len(ts),
                "mean_reward": sum(t.reward for t in ts) / len(ts),
                "mean_prompt_tokens": sum(t.prompt_tokens for t in ts) / len(ts),
                "mean_completion_tokens": sum(t.completion_tokens for t in ts) / len(ts),
                "mean_cost_usd": sum(t.cost_usd for t in ts) / len(ts),
                "mean_elapsed_sec": sum(t.elapsed_sec for t in ts) / len(ts),
                "activations": sum(len(t.pins) for t in ts),
                "errors": sum(1 for t in ts if t.error),
            }
        )
    return {"rows": rows}
