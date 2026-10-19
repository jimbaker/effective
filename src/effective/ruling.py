"""A principal's answer to an authority park, and the shape the engines store it in.

An authority park waits on a name in a reserved namespace (`RESERVED_AUTHORITY_TAGS`). A policy's
verdict can park an op; only a principal settles the park, and the answer it delivers is an
`AuthorityRuling` that records who ruled. The engines refuse an authority name whose payload is
not one. An await receives the answer alone, unless its schema is `ruling_schema(S)`, when it
receives the whole ruling with the answer read as `S`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from types import GenericAlias
from typing import Any, assert_never, get_args, get_origin

from pydantic_core import to_jsonable_python

from effective.keys import QUALIFYING_TAGS, RESERVED_AUTHORITY_TAGS, Key, Segment
from effective.keys.grammar import TAG_SEPARATOR, TERM_SEPARATOR, KeySyntaxError, parse
from effective.ops import Unretryable

RULING = "__ruling__"
"""The key a stored ruling sits under, the only key of its payload."""


class UnruledAnswer(ValueError, Unretryable):
    """An answer to an authority park that is not a principal's ruling. Nothing commits on it."""

    def __init__(self, name: str | Key) -> None:
        super().__init__("an authority park's answer is not a ruling", _text(name))
        self.name = name


@dataclass(frozen=True)
class AuthorityRuling[T]:
    """A principal's answer to one authority park. `by` names who ruled."""

    by: Segment
    answer: T


def _text(name: str | Key) -> str:
    match name:
        case Key():
            return name.stored()
        case str():
            return name
        case unreachable:
            assert_never(unreachable)


def names_authority(name: str | Key) -> bool:
    """Does `name` address an authority park?

    A substrate park carries its authority tag in some term: first when unframed, after the frame
    terms when a gather branch or a fork rescopes it (`gather:0,1;budget-grant:r1,0`,
    `fork:r-fork;approve:...`). A qualifying tag (`QUALIFYING_TAGS`) frames the name after it and
    is no park, so `fork:r-fork;review:m1` answers no. An author's await name is one term whose
    tag `ops.event_name` has already refused to reserve, so no author name answers yes."""
    return any(
        tag in RESERVED_AUTHORITY_TAGS and tag not in QUALIFYING_TAGS for tag in _tags(name)
    )


def _tags(name: str | Key) -> list[str]:
    """Each term's tag: a `Key`'s own terms, or a wire name's, read as the grammar reads them."""
    match name:
        case Key():
            return [term.tag for term in name.terms()]
        case str():
            try:
                return [term.tag for term in parse(name).terms]
            except KeySyntaxError:
                return [term.partition(TAG_SEPARATOR)[0] for term in name.split(TERM_SEPARATOR)]
        case unreachable:
            assert_never(unreachable)


HYPOTHETICAL = Segment("hypothetical")
"""Who rules a counterfactual's fork point: the delta a fork pre-delivers, which reaches only its
own `fork:`/`hyp:` lineage."""


def ruled_answer(name: str | Key, payload: Any, *, by: Segment) -> Any:
    """What answers the park `name`: `by`'s ruling where `name` is an authority park, and the
    payload as it stands elsewhere."""
    return AuthorityRuling(by, payload) if names_authority(name) else payload


def dump_ruling(ruling: AuthorityRuling) -> dict[str, Any]:
    """The JSON an engine stores for `ruling`."""
    return {RULING: {"by": ruling.by, "answer": to_jsonable_python(ruling.answer)}}


def wire_payload(name: str | Key, payload: Any) -> Any:
    """The payload an engine stores for an event named `name`: a ruling in its stored shape, and
    any other payload as it stands, except under an authority name, which only a ruling answers."""
    match payload:
        case AuthorityRuling():
            return dump_ruling(payload)
        case _ if names_authority(name):
            raise UnruledAnswer(name)
        case _:
            return payload


def answer_of(name: str | Key, raw: Any) -> Any:
    """What the payload stored under `name` answers: a ruling's answer where `name` is an authority
    park, and the payload as it stands elsewhere, whatever its shape."""
    if not names_authority(name):
        return raw
    match _load_ruling(raw):
        case AuthorityRuling(answer=answer):
            return answer
        case None:
            raise UnruledAnswer(name)


def ruling_schema(schema: Any) -> Any:
    """The schema of an await that receives the whole ruling, its answer read as `schema`."""
    return GenericAlias(AuthorityRuling, (schema,))


def split_ruling(delivered: Any) -> tuple[Segment | None, Any]:
    """Who ruled `delivered` and its answer, or `None` and `delivered` itself when no ruling came
    with it, which only an in-process recorder's canned payload is."""
    match delivered:
        case AuthorityRuling(by=by, answer=answer):
            return by, answer
        case _:
            return None, delivered


def load_answer(name: str | Key, schema: Any, raw: Any, load: Callable[[Any, Any], Any]) -> Any:
    """What an await on `name` receives from the stored payload `raw`, read by `load(schema, _)`.

    | schema             | the await receives                                                 |
    |--------------------|--------------------------------------------------------------------|
    | `ruling_schema(S)` | the ruling, its answer read as `S`; any other payload ends the run |
    | any other          | what `answer_of` gives, read as `schema`                           |

    A ruling schema on a name that is no authority park is refused, since no principal answers
    that park.
    """
    match get_origin(schema), get_args(schema):
        case (origin, (_,)) if origin is AuthorityRuling and not names_authority(name):
            raise ValueError(f"{_text(name)} is no authority park, so no ruling answers it")
        case (origin, (inner,)) if origin is AuthorityRuling:
            match _load_ruling(raw):
                case AuthorityRuling(by=by, answer=answer):
                    return AuthorityRuling(by, load(inner, answer))
                case None:
                    raise UnruledAnswer(name)
        case _:
            return load(schema, answer_of(name, raw))


def _load_ruling(raw: Any) -> AuthorityRuling | None:
    """The ruling a stored payload holds, or `None` for any other payload, a ruler that is not a
    segment among them."""
    match raw:
        case {"__ruling__": {"by": str(by), "answer": answer}} if len(raw) == 1:
            try:
                return AuthorityRuling(Segment(by), answer)
            except ValueError:
                return None
        case _:
            return None
