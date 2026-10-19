# Parse a template once per shape

`effective/keys/processor.py` (`_skeleton`), `infra/tdom` (`CachedTemplateParserProxy`).

A t-string processor does two jobs: it reads a template's structure, and it fills that structure
with values. The structure comes from the call site, so every call from one site has the same
structure, and a processor can read it once and fill it on every call. `compose_key` does this, so
a sweep that composes thousands of keys from a handful of call sites parses a handful of
skeletons.

## What a template's shape is

PEP 750 evaluates a t-string into a `Template` at the call site, and the parts split by when they
are fixed:

| part | fixed | so the shape |
|---|---|---|
| `strings`, the statics | at compile time | includes them |
| each hole's `expression` | at compile time | includes it, if the processor reads it (a refusal names it) |
| each hole's `conversion` | at compile time | includes it, if the processor reads it |
| each hole's `format_spec` | **at each call**: a nested field (`{x:{width}}`) is evaluated into the spec eagerly, like the value | includes the spec string as evaluated |
| each hole's `value` | at each call | excludes it |

So the shape is the statics plus each hole's expression, conversion and evaluated spec. A spec
looks static and is not: `t"{x:{w}}"` carries a different `format_spec` whenever `w` changes, and a
cache keyed on the statics alone serves one call's spec to the next.

## The rules

| rule | why |
|---|---|
| key on everything the structural read takes from the template, the spec included | the read is a pure function of exactly those parts, so the memo returns what the read would |
| key by equality | the statics are a tuple of strings; equal tuples are the same shape whichever object holds them |
| leave every value check outside the memo, in the order it ran | a value varies per call, and moving a check into the memo changes which defect a two-defect template reports |
| let a refusal raise every time | `functools.lru_cache` stores results, never exceptions, so a refused shape is refused at each call |
| bound the cache | the shapes are a program's call sites, finite and small, but a bound costs nothing |

A memo keyed on statics alone is sound when the structural read takes nothing else.
`tdom` keys `CachableTemplate` on `strings` alone, and its parser reads each hole's expression only
for an error message, leaving the spec and conversion to render time, where it reads them from the
live template. The `CONSIDER` comment beside that key asks exactly this page's question.
`compose_key`'s read takes all four parts: `domain=` and `default=` in the spec change the skeleton,
and the expression names the culprit in a refusal. So `_skeleton` keys on all four,
and `tests/test_key_structure.py` pins two templates alike in their statics, then alike but for an
expression, composed apart.

## Why this is the t-string's to give

An f-string has no shape to remember: by the time a function holds one it is a flat string, and
the structure, if any, has to be parsed out of the bytes again on every call. A `Template` hands
the processor its structure apart from its values, which is what [concepts/flatten](flatten.md)
argues for on safety grounds; the same separation is what makes the structural read cacheable.
