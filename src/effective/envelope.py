"""Response envelopes: the wire alphabet as an explicit, swappable seam.

The channel processor's ``Prompt.resolve`` consumes a ``Mapping`` and has never
cared where it came from; callers have implicitly assumed "the provider's JSON
mode" and hardcoded that convention. This module makes the envelope a value:

- ``instructions(channels)`` renders the answer-format contract for the prompt
  tail (how the model should shape its reply), and
- ``parse(text)`` turns a raw completion into the ``Mapping`` that
  ``resolve()`` already takes.

Two alphabets, chosen by the heterogeneous-alphabet principle, as measured on
the SkillsBench M1 wire ladder:

- ``JsonEnvelope``: one JSON object, parsed LENIENTLY (first object wins,
  control characters tolerated). Right when every value is a scalar; wrong the
  moment a value carries code or shell text, because the model must then author
  inner escaping by hand (the JSON-in-JSON failure: 27/120 trials).
- ``SectionEnvelope``: sentinel-delimited sections (``[[ ## name ## ]]``, the
  convention DSPy proved in the wild): structure lives in a marker alphabet
  that cannot collide with JSON/code/shell metacharacters, and section BODIES
  are raw, so a heredoc rides through with zero escaping. Depth is perceived,
  not counted.

Values are coerced by the channels themselves at ``resolve()`` (TypeAdapter
handles ``"0.9"`` -> float), so an envelope deals only in text: it is the
data-axis seam between provider bytes and typed channels. Like every channel
seam it is tunable, and the M1 harness is its measuring instrument.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol


class Envelope(Protocol):
    """The wire-alphabet seam: prompt-side contract + completion-side parse."""

    def instructions(self, channels: Mapping[str, Any]) -> str: ...
    def parse(self, text: str) -> dict[str, Any]: ...


class EnvelopeParseError(ValueError):
    """The completion carried no parseable payload for this envelope."""


@dataclass(frozen=True)
class JsonEnvelope:
    """One flat JSON object; leniently decoded.

    ``parse`` takes the FIRST valid object (models sometimes emit a second —
    the "trailing characters" failure) and tolerates control characters in
    strings (models write literal newlines). A bench loses turns, not trials.
    """

    def instructions(self, channels: Mapping[str, Any]) -> str:
        keys = ", ".join(f'"{name}"' for name in channels)
        return (
            "Respond with a single JSON object containing exactly these keys: "
            f"{keys}. Use null for keys that do not apply this turn."
        )

    def parse(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        start = stripped.find("{")
        if start < 0:
            raise EnvelopeParseError("no JSON object in completion")
        try:
            obj, _ = json.JSONDecoder(strict=False).raw_decode(stripped[start:])
        except json.JSONDecodeError as exc:
            raise EnvelopeParseError(f"unparseable JSON object: {exc}") from None
        if not isinstance(obj, dict):
            raise EnvelopeParseError(f"expected an object, got {type(obj).__name__}")
        return obj


_MARKER = re.compile(r"^\[\[ *## *([A-Za-z0-9_-]+) *## *\]\] *$", re.MULTILINE)
_DONE = "completed"


@dataclass(frozen=True)
class SectionEnvelope:
    """Sentinel-delimited sections; bodies are raw text.

    The marker line ``[[ ## name ## ]]`` opens a section that runs to the next
    marker (or end of text); ``[[ ## completed ## ]]`` closes the reply. The
    marker alphabet shares no metacharacters with JSON, shell, or code, so a
    section body needs NO escaping at any depth — the heterogeneous-alphabet
    principle as a wire format. Sections may arrive in any order; a repeated
    name last-wins; text before the first marker is ignored (models preface).
    A value that is itself structured can carry single-level JSON *inside* its
    section — sections outer, JSON inner: adjacent levels, different alphabets.
    """

    def instructions(self, channels: Mapping[str, Any]) -> str:
        sections = "\n".join(f"[[ ## {name} ## ]]\n<{name}>" for name in channels)
        return (
            "Respond in labeled sections, each opened by its marker on its own "
            "line, exactly in this shape:\n\n"
            f"{sections}\n[[ ## {_DONE} ## ]]\n\n"
            "Write each value as plain text on the lines after its marker — no "
            "quoting or escaping, multi-line values are fine. Omit a section "
            "that does not apply this turn. Always end with the "
            f"[[ ## {_DONE} ## ]] marker."
        )

    def parse(self, text: str) -> dict[str, Any]:
        matches = list(_MARKER.finditer(text))
        if not matches:
            raise EnvelopeParseError("no section markers in completion")
        out: dict[str, Any] = {}
        for i, m in enumerate(matches):
            name = m.group(1)
            if name == _DONE:
                continue
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            out[name] = text[m.end() : end].strip()
        return out
