"""Agent-Skills registry: standard-layout skill packs as data-axis fragments.

The Agent Skills standard (agentskills.io) is a filesystem convention:
``<pack>/<skill-name>/SKILL.md`` with YAML frontmatter (``name`` +
``description`` required), optional ``references/``, ``scripts/``, ``assets/``.
``SkillRegistry.load`` walks that layout and exposes each skill as the channel
processor expects:

- ``index()``: the catalog (every name + description), interpolated into the
  *cached* prefix: the spec's metadata disclosure level.
- ``body(name)``: the SKILL.md body as a **degenerate Template** (zero
  interpolations), spliced by a ``Skill`` node as the volatile disclosed tail:
  the activation level. Value-independent by construction, so it passes the
  independence lint trivially.
- ``info(name).content_hash``: a blake2b tree hash over the whole skill
  directory (body + references + scripts + assets). This is the pin identity:
  activation records ``(name, hash)`` so the ledger can say which skill version
  informed which decision.

Frontmatter is hand-parsed: the spec's required fields are flat strings, and
``effective/`` stays lean (no YAML runtime dep). Full-YAML validation
(``skills-ref``) is dev tooling.

Loading does filesystem I/O, so it belongs at handler/setup time, never
between yields. ``effective.channels`` stays I/O-free and depends on nothing
here: the walk sees a registry only as the structural ``SkillResolver``
protocol (``body(name)``), which ``SkillRegistry`` satisfies.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from hashlib import blake2b
from pathlib import Path
from string.templatelib import Template

from pydantic import BaseModel, ConfigDict

from effective.api import Effect, step
from effective.domain import CallTool
from effective.keys import Index, Key, Name, Tag, compose_key

_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_FLAT_FIELD_RE = re.compile(r"^(?P<key>[A-Za-z0-9_-]+):(?P<rest>.*)$")
_HASH_SIZE = 16  # digest_size — same family/size as the deterministic span ids


class SkillPackError(Exception):
    """A skill pack breaches the standard layout or the frontmatter subset —
    raised at ``load`` (the dev gate: early, located, never a silent skip)."""


@dataclass(frozen=True)
class Script:
    """A pinned executable script from a skill pack: the source plus the KEY of the
    activation that pinned it. Pass it straight to ``effective.code.run_code`` (it
    satisfies the ``PinnedScript`` protocol structurally), which runs it inside
    ``scoped(key)``, so every segment checkpoint underneath carries the pack as a
    FRAME and the ledger separates pinned from improvised code structurally, with no
    recorded payload at all.

    ``key`` is the activation's key and deliberately not the pack's content hash: a
    hash is a run-varying coordinate no projection can drop, so putting it here would
    make two runs of one program fold to different nodes. The hash is one join away,
    on the activation checkpoint."""

    source: str
    key: Key


class Pin(BaseModel):
    """A recorded skill-content pin: the run's only interface to the changing
    skill tree. Activation is pin-at-time-T; ``refresh`` is re-pin-at-time-T'.
    Both return a ``Pin`` the workflow threads explicitly into renders
    (``{skill("x", pin=p)}``; no handler pin-state, per no-call/cc).

    The pin **carries the disclosed body and the pack's scripts**, so it rides
    the activation step's checkpoint and replay resolves content from *recorded*
    state: the working tree is never consulted for a pinned render or a pinned
    script run. The durable artifact store is a keyed placeholder, so the content
    rides ``ctx.step``; a pack that ships *large* scripts needs content-addressed
    ``store_artifact`` resolution first. JSON-round-trips through the Absurd
    checkpoint serde (``to_jsonable_python`` + ``TypeAdapter``)."""

    model_config = ConfigDict(frozen=True)

    name: str
    content_hash: str
    body: str
    scripts: dict[str, str] = {}
    key: Key | None = None
    """The activation step's key, attached by ``activate_skill``/``refresh_skill``.

    DERIVED, not recorded: the domain interpreter builds the pin and knows nothing about
    keys, so this is re-attached deterministically on both the live and the replay path
    from the same composed key. A recorded copy would denormalize the identity into the
    payload, and the identity belongs in the key."""

    def script(self, filename: str) -> Script:
        """The pinned script ``filename`` (posix path relative to the pack's
        ``scripts/``) with provenance — replay-exact by construction: the source
        came off the activation checkpoint, never the working tree."""
        source = self.scripts.get(filename)
        if source is None:
            raise LookupError(
                f"pin {self.name!r} has no script {filename!r} (has {sorted(self.scripts)})"
            )
        if self.key is None:
            raise ValueError(
                f"pin {self.name!r} has no activation key, so a script from it cannot be scoped "
                "to the pack that pinned it. Obtain the pin from `activate_skill`/`refresh_skill` "
                "rather than constructing one."
            )
        return Script(source=source, key=self.key)


SKILL = Tag("skill")
"""The skill-pin namespace — two variants, `activate` and `refresh`, separated by the literal
at index 1."""

DISCLOSE_TOOL = "skill-disclose"
"""The reserved CallTool name for a skill disclosure — the handler's domain
interpreter implements it by delegating to ``SkillRegistry.disclose`` (the
registry is deployment infrastructure, exactly like the model caller)."""


def activate_skill(name: str) -> Effect[Pin]:
    """Activation as a recorded, namespaced step: ``skill:{name},activate``.

    **Each call is its own pin event** — activate once and thread the pin. On
    the durable engine a duplicate same-named step gets an occurrence-suffixed
    checkpoint key (the SDK's ``name#2``), so a second activation performs a
    fresh disclose — an unnamed refresh, deterministic on replay but a new pin.
    (The in-memory ``RecordingHandler`` cans responses by name and would hand
    back the same value — don't lean on that divergence.) A deliberate,
    *named* re-read is ``refresh_skill``."""
    # Two VARIANTS of one namespace, separated by the literal at index 1 (`activate` here,
    # `refresh` below). `name` is INTERIOR — a static follows it — so it is `Segment`-typed and
    # sits at a counted offset.
    key = compose_key(t"{SKILL}:{Name(name)},activate")
    pin = yield from step(
        key.stored(),
        CallTool(name=DISCLOSE_TOOL, args={"skill": name, "event": "activate"}, result_schema=Pin),
    )
    return pin.model_copy(update={"key": key})


def refresh_skill(name: str, n: int) -> Effect[Pin]:
    """Refresh = re-pin: same op shape, explicitly numbered as
    ``skill:{name},refresh,{n}``, so its checkpoint key can never collide with
    the activation's or an earlier refresh's (injectivity). ``n`` is the
    workflow's own ordinal (explicitly threaded state, not handler state)."""
    key = compose_key(t"{SKILL}:{Name(name)},refresh,{Index(n)}")
    pin = yield from step(
        key.stored(),
        CallTool(
            name=DISCLOSE_TOOL, args={"skill": name, "event": "refresh", "n": n}, result_schema=Pin
        ),
    )
    return pin.model_copy(update={"key": key})


@dataclass(frozen=True)
class SkillInfo:
    """One loaded skill's identity: the spec's required metadata, the pin hash,
    and the remaining flat frontmatter fields (``license``, ``compatibility``,
    …) verbatim."""

    name: str
    description: str
    content_hash: str
    path: Path | None  # None for an in-memory (test / native-extension) skill
    fields: Mapping[str, str] = field(default_factory=dict)


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split ``SKILL.md`` text into (flat frontmatter fields, body).

    Hand-parsed YAML *subset* — exactly what the spec's string fields need:
    ``key: value`` pairs at top level, optional matched quotes, ``|`` / ``>``
    block scalars (joined by newline / space; chomping indicators tolerated,
    not honored), and **block sequences of plain scalars** (``- item`` lines
    under a bare ``key:``, joined by newline, e.g. ``allowed-tools``, kept
    faithfully for its consumers). A flow list that does
    not close on its own line is rejected loudly: a truncated value is the
    wrong input to hand a future permission compiler. Nested *maps* under a
    bare ``key:`` (e.g. ``metadata``) are skipped verbatim — the substrate does
    not consume them. Anything richer is ``skills-ref``'s job.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillPackError("SKILL.md must open with a `---` frontmatter fence")
    try:
        end = next(i for i, ln in enumerate(lines[1:], start=1) if ln.strip() == "---")
    except StopIteration:
        raise SkillPackError("unterminated frontmatter: no closing `---`") from None
    fields: dict[str, str] = {}
    i = 1
    while i < end:
        line = lines[i]
        i += 1
        if not line.strip() or line.startswith("#") or line[0] in " \t":
            continue  # blank, comment, or the continuation of a skipped nested map
        m = _FLAT_FIELD_RE.match(line)
        if m is None:
            raise SkillPackError(f"unparseable frontmatter line: {line!r}")
        key, rest = m["key"], m["rest"].strip()
        if rest.startswith("[") and not rest.endswith("]"):
            raise SkillPackError(
                f"frontmatter field {key!r}: a flow list must close on its own line "
                "(a truncated value would silently mis-parse)"
            )
        if rest.startswith(("|", ">")):
            fields[key], i = _block_scalar(lines, i, end, joiner="\n" if rest[0] == "|" else " ")
        elif not rest and (items := _block_sequence(lines, i, end)) is not None:
            fields[key], i = items
        elif value := _unquote(rest):  # a bare `key:` heads a nested map — skipped above
            fields[key] = value
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return fields, body


def _block_scalar(lines: list[str], i: int, end: int, *, joiner: str) -> tuple[str, int]:
    """Gather a ``|`` / ``>`` block's indented lines; returns (value, next line)."""
    block: list[str] = []
    while i < end and (not lines[i].strip() or lines[i][0] in " \t"):
        block.append(lines[i].strip())
        i += 1
    return joiner.join(block).strip(), i


def _block_sequence(lines: list[str], i: int, end: int) -> tuple[str, int] | None:
    """Parse an indented ``- item`` block sequence of plain scalars under a bare
    ``key:`` into a newline-joined string; ``None`` if the block is not a
    sequence (a nested map — skipped by the caller as documented)."""
    items: list[str] = []
    j = i
    while j < end and (not lines[j].strip() or lines[j][0] in " \t"):
        stripped = lines[j].strip()
        if stripped:
            if not stripped.startswith("- "):
                return None  # a nested map (or something richer) — not ours
            items.append(_unquote(stripped.removeprefix("- ").strip()))
        j += 1
    if not items:
        return None
    return "\n".join(items), j


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def _tree_hash(root: Path) -> str:
    """Deterministic blake2b tree hash of every file under ``root`` (SKILL.md,
    references, scripts, assets), keyed by sorted posix-relative path: the pin
    identity. Content-only: mtimes and walk order cannot move it.

    Symlinks are rejected loudly: ``rglob`` does not descend a symlinked
    directory (content behind it would be invisible to the pin) and a symlinked
    file hashes state *outside* the skill dir — either way the hash would lie
    about what the pin covers. The fail-loud dev gate, same as the layout
    checks."""
    h = blake2b(digest_size=_HASH_SIZE)
    for p in sorted(root.rglob("*")):
        if p.is_symlink():
            raise SkillPackError(
                f"{p}: symlinks are not allowed in a skill pack (the content hash "
                "could not cover what the pin claims)"
            )
        if not p.is_file():
            continue
        try:
            h.update(p.relative_to(root).as_posix().encode())
        except UnicodeEncodeError as exc:
            raise SkillPackError(f"{root}: undecodable filename in skill pack: {exc}") from None
        h.update(b"\0")
        h.update(blake2b(p.read_bytes(), digest_size=_HASH_SIZE).digest())
    return h.hexdigest()


def _template_hash(template: Template) -> str:
    """Pin identity for an in-memory body: hash the template's static strings
    (its value-independent shape — all a degenerate body has)."""
    h = blake2b(digest_size=_HASH_SIZE)
    for s in template.strings:
        h.update(s.encode())
        h.update(b"\0")
    return h.hexdigest()


@dataclass(frozen=True)
class _Entry:
    info: SkillInfo
    body: Template
    scripts: Mapping[str, str] = field(default_factory=dict)


def _load_scripts(skill_dir: Path) -> dict[str, str]:
    """The pack's ``scripts/*.py`` sources, keyed by posix path relative to
    ``scripts/`` — what a ``Pin`` carries for ``pin.script(...)``. Python only:
    the executable tier runs in the sandboxed code engine; other
    script kinds stay hash-covered but undisclosed. Non-UTF-8 is a loud dev
    error (a script the engine cannot read is a broken pack, not a skip)."""
    scripts_dir = skill_dir / "scripts"
    if not scripts_dir.is_dir():
        return {}
    out: dict[str, str] = {}
    for p in sorted(scripts_dir.rglob("*.py")):
        if not p.is_file():
            continue
        try:
            out[p.relative_to(scripts_dir).as_posix()] = p.read_text()
        except UnicodeDecodeError as exc:
            raise SkillPackError(f"{p}: script is not valid UTF-8: {exc}") from None
    return out


class SkillRegistry:
    """The loaded skill set behind one run's renders.

    Satisfies ``effective.channels.SkillResolver`` structurally. Immutable
    after construction: a changed working tree is a *new* registry, never a
    mutation the handler sees ambiently.
    """

    def __init__(self, entries: Mapping[str, _Entry]):
        self._entries = dict(entries)

    @classmethod
    def load(cls, root: str | Path) -> SkillRegistry:
        """Scan ``root`` for the standard layout: every immediate subdirectory
        holding a ``SKILL.md`` is a skill. Validates the frontmatter subset,
        the spec's name grammar, and name == directory (the dev gate)."""
        root = Path(root)
        if not root.is_dir():
            raise SkillPackError(f"skill pack root is not a directory: {root}")
        entries: dict[str, _Entry] = {}
        for skill_md in sorted(root.glob("*/SKILL.md")):
            skill_dir = skill_md.parent
            fields, body = parse_frontmatter(skill_md.read_text())
            name = fields.pop("name", "")
            description = fields.pop("description", "")
            if not name or not description:
                raise SkillPackError(f"{skill_md}: frontmatter requires name and description")
            if not _NAME_RE.match(name):
                raise SkillPackError(
                    f"{skill_md}: name {name!r} breaches the spec grammar "
                    "(lowercase alphanumerics and inner hyphens)"
                )
            if name != skill_dir.name:
                raise SkillPackError(f"{skill_md}: name {name!r} != directory {skill_dir.name!r}")
            info = SkillInfo(name, description, _tree_hash(skill_dir), skill_dir, fields)
            entries[name] = _Entry(info, Template(body), _load_scripts(skill_dir))
        if not entries:
            raise SkillPackError(f"no */SKILL.md under {root} — not a skill pack")
        return cls(entries)

    @classmethod
    def in_memory(cls, skills: Mapping[str, tuple[str, Template]]) -> SkillRegistry:
        """A registry from ``{name: (description, body)}``, for tests and for
        native effective-skills whose bodies carry channels. The same grammar and
        pin discipline apply."""
        entries: dict[str, _Entry] = {}
        for name, (description, body) in skills.items():
            if not _NAME_RE.match(name):
                raise SkillPackError(f"skill name {name!r} breaches the spec grammar")
            info = SkillInfo(name, description, _template_hash(body), None)
            entries[name] = _Entry(info, body)
        return cls(entries)

    def index(self) -> str:
        """The catalog — one ``- name: description`` line per skill, sorted.
        This is the spec's always-loaded metadata level; interpolate it into
        the *cached* prefix (disclosure-is-the-cache-boundary)."""
        return "\n".join(
            f"- {e.info.name}: {e.info.description}"
            for e in sorted(self._entries.values(), key=lambda e: e.info.name)
        )

    def _get(self, name: str) -> _Entry:
        """Look up an entry or raise ``LookupError`` — the located render error the
        walk converts an unknown-skill reference into."""
        entry = self._entries.get(name)
        if entry is None:
            raise LookupError(f"unknown skill {name!r} (registry has {sorted(self._entries)})")
        return entry

    def body(self, name: str) -> Template:
        """The disclosed fragment for ``name`` — raises ``LookupError`` for an
        unknown skill (the walk converts it to a located render error)."""
        return self._get(name).body

    def info(self, name: str) -> SkillInfo:
        return self._get(name).info

    def disclose(self, name: str) -> Pin:
        """The pin for ``name`` at this registry's snapshot — what a handler's
        domain interpreter returns for the ``skill-disclose`` tool. Covers
        standard (markdown) bodies; a native channel-bearing body cannot ride a
        checkpoint as text, so it cannot be pinned."""
        entry = self._get(name)
        if list(entry.body.interpolations):
            raise SkillPackError(
                f"skill {name!r} has a native (channel-bearing) body; pinning covers "
                "standard markdown bodies, since a channel-bearing body cannot ride a checkpoint "
                "as text"
            )
        return Pin(
            name=name,
            content_hash=entry.info.content_hash,
            body="".join(entry.body.strings),
            scripts=dict(entry.scripts),
        )

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))
