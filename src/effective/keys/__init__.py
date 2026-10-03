"""The identity grammar: the marker types a call site hands to `compose_key`, and the composer.

| marker | means | legal position | is a `str`? |
|---|---|---|---|
| `Tag` | a namespace terminal | leading | yes |
| `AuthorityTag` | a namespace whose names ARE the authorization | leading | yes |
| `Segment` | delimiter-free, so unambiguous mid-string | interior | yes |
| `Name` | a coordinate naming a distinct position | interior | yes |
| `Run` | a coordinate identifying an execution | interior | yes |
| `Index` | a coordinate counting repetitions | interior | an `int` |
| `Subject` | a coordinate carrying the domain's value | interior | yes |
| `Key` | a finished composition (may contain delimiters) | terminal | **NO — opaque** |

The four roles declare what the coordinate MEANS, which is what a projection reads: a fold keeps a
`Name` and drops an `Index`, a cross-run comparison drops a `Run`, and every projection keeps a
`Subject`. `Segment` stays the building block three of them are made of; `Index` is an `int`,
delimiter-free by type. `Role` is the protocol a drop set is typed over.

The first three are inputs. A call site builds one to promise the composer something that the
marker checks at construction, and the composer reads that promise off the marker's `str`-ness and
is done with it. `Key` is the output: it travels, it is stored, and replay binds to it, so it is
opaque and reached through named exits instead of a `str`-ness anyone can pick up.

Where things live::

    grammar     the key itself — terms, atoms, `parse`, `render`, `Key`, `Scope`
    marker      the promises a call site makes: `Tag`, `Segment`, `AuthorityTag`
    processor   `compose_key`, the t-string interpreter, and its refusals
    frame       the leading terms a `scoped(...)` contributes, and a thread of control's ordinals
    registry    the source map: read a composed key back into named fields and a line

This module re-exports the surface those five present. New code should name the submodule that
owns what it wants; the names here are what the tree already imports.
"""

from effective.keys.frame import (
    FramePosition,
    frame_path,
    scope_prefix,
    unframed,
)
from effective.keys.grammar import (
    Key,
    Scope,
)
from effective.keys.marker import (
    RESERVED_AUTHORITY_TAGS,
    AuthorityTag,
    Index,
    Name,
    Ordinal,
    Role,
    Run,
    Segment,
    Subject,
    Tag,
    authored_key,
    carries_structure,
)
from effective.keys.processor import (
    DEFAULT_DIRECTIVE,
    DOMAIN_DIRECTIVE,
    compose_key,
    gather_prefix,
    race_choice,
    race_endings,
    race_prefix,
)

__all__ = [
    "DEFAULT_DIRECTIVE",
    "DOMAIN_DIRECTIVE",
    "RESERVED_AUTHORITY_TAGS",
    "AuthorityTag",
    "FramePosition",
    "Index",
    "Key",
    "Name",
    "Ordinal",
    "Role",
    "Run",
    "Scope",
    "Segment",
    "Subject",
    "Tag",
    "authored_key",
    "carries_structure",
    "compose_key",
    "frame_path",
    "gather_prefix",
    "race_choice",
    "race_endings",
    "race_prefix",
    "scope_prefix",
    "unframed",
]
