"""Tests for `scripts/key_sweep.py`, each recording the MUTATION that reddens it.

The sweep's counts look plausible whether or not it is right, so running it shows nothing. Two
defects hide that way: a compose rule that reads the LEADING tag where its docstring claims every
term, and a `refused()` that normalizes a doubled separator before parsing, repairing the one
signature it exists to find. An assertion whose mutation leaves it green measures nothing.
"""

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.key_sweep import (  # noqa: E402
    EXEMPT_DIRS,
    EXEMPT_FILES,
    composing_fstrings,
    is_exempt,
    mangled,
    markdown_prose,
    refused,
    spellings,
)

TAGS = {
    "ledger": ["ledger;{}"],
    "gather": ["gather:{},{};{}"],
    "artifact": ["artifact:{}/{},{}"],
}
"""A hand-built stand-in for `build/key-registry.json`, deliberately tiny.

The registry is the oracle for a SWEEP; it is the wrong oracle for a test OF the sweep, which
wants a fixed alphabet so a case cannot pass because production happened to mint something.
"""


def compose_findings(source: str) -> list[tuple[int, str, str]]:
    return list(composing_fstrings(source, TAGS))


def test_an_fstring_composing_behind_a_consumer_scope_is_a_finding():
    """The T1 shape. `q` is nobody's namespace, and `ledger` sits behind it.

    MUTATION: put `leading_tag(body) in tags` back in `composing_fstrings` — this reddens and the
    bare `ledger:{ev}` case below stays green, which is how the defect survived a green run.
    """
    findings = compose_findings('x = f"q;ledger:{ev}"\n')
    assert [(token, tag) for _line, token, tag in findings] == [("q;ledger:{ev}", "ledger")]


def test_an_fstring_composing_at_the_head_is_still_a_finding():
    """The case the head-only rule DID catch — kept so the widening is shown to be a widening
    rather than a swap.

    MUTATION: make `claimed_tag` return `None` for a head match; this reddens.
    """
    findings = compose_findings('y = f"ledger:{ev}"\n')
    assert [token for _line, token, _tag in findings] == ["ledger:{ev}"]


def test_a_t_string_is_never_a_compose_finding():
    """`t"…"` is the sanctioned composer — the whole point of the rule is the flatten, not the
    interpolation, so the template form must stay silent.

    MUTATION: drop the `"f" not in prefix` guard; this reddens.
    """
    assert compose_findings('z = t"ledger:{ev}"\n') == []


def test_prose_and_error_messages_are_not_composes():
    """Two structural exclusions, both required and both easy to lose.

    A SPACE means prose; a TRAILING SEPARATOR means a frame prefix under construction, which is
    the render-backend position the layering rule sanctions.

    MUTATION: delete either arm of `_is_not_a_key`; the matching line reddens.
    """
    source = 'a = f"ledger: the append for {mid} failed"\nb = f"gather:{g},{i};"\n'
    assert compose_findings(source) == []


def test_a_doubled_separator_is_refused_and_reported_as_mangled():
    """The T2 shape — the fingerprint of a rewrite that ran and did not finish.

    MUTATION: restore `re.sub(r"[:;,/]{2,}", ":", probe)` in `refused()`; the `ledger;;m1`
    assertion reddens, and the sweep goes back to reporting green on its own damage.

    **The token is `ledger;;m1` for a measured reason, and the first draft got it wrong.** The
    obvious choice, `gather:0,3;;ledger`, is refused under BOTH versions — the normalizer turns it
    into `gather:0,3:ledger`, which the grammar refuses anyway because `3:ledger` is not an atom —
    so it cannot tell the two apart, and a test written around it passed under its own named
    mutation. `ledger;;m1` normalizes to `ledger:m1`, which PARSES; that is the only shape where
    the suppression is observable. Both are kept below: one detects the defect, the other is the
    near-miss that does not, which is worth leaving visible.
    """
    assert refused("ledger;;m1"), "the normalizer would make this parse as `ledger:m1`"
    assert mangled("ledger;;m1")
    assert refused("gather:0,3;;ledger")  # refused either way — cannot detect the mutation
    assert mangled("gather:0,3;;ledger")
    assert not mangled("gather:0,3;ledger")


@pytest.mark.parametrize(
    "token",
    [
        "$awaitEvent:approve;x",  # another runtime's identity, in a shape ours happens to admit
        "$awaitEvent:gather:0:0:ev",  # …and one it does not — this is what FOREIGN protects
        "artifact:message/ledger,sha256-9f2c",  # `/` is path structure; `ledger` here is an atom
        "gather:0,1;ledger;m1",  # simply in the language
    ],
)
def test_tokens_the_sweep_must_stay_silent_about(token):
    """The cases the docstring calls deliberate. Two were listed as *refused-but-fine* until this
    test was written, and both actually parse or short-circuit — a docstring claim a reproduction
    refuted.

    MUTATION: delete the `FOREIGN` guard in `refused()`; **only the second case reddens**, and the
    first is kept to show why. `$awaitEvent:approve;x` happens to be well-formed under OUR grammar
    too, so it is silent for a reason that has nothing to do with being foreign — a payload we
    cannot parse (`gather:0:0:ev` is Absurd's, in Absurd's spelling) is the only shape that
    distinguishes "we have no opinion" from "we agree". Guarding on the first alone would have
    left the guard untested.
    """
    assert not refused(token)


@pytest.mark.parametrize("token", ["a:b:c", "gather:0:0:tool:x", "artifact:message/rfc822:9f2c"])
def test_retired_spellings_are_refused(token):
    """The flat form, in the three shapes it actually appears in. The third matters most: it reads
    like the legal MIME case one test up and differs only in the separator before the digest.

    MUTATION: make `refused()` return `False` on `KeySyntaxError`; all three redden.
    """
    assert refused(token)


def test_the_history_layer_is_exempt():
    """The `reports` directory is dated by contract and is where history is relocated TO, so
    sweeping it would penalize the correct move.

    `docs/CHANGELOG.md` is asserted NON-exempt: the directory rule subsumes any file rule, and
    the retirement of the file rule is pinned rather than merely absent.

    MUTATION: return `False` from `is_exempt`; the first assertion reddens.
    """
    dated = Path("reports")
    assert is_exempt(dated / "a-dated-note.md")
    assert is_exempt(dated / "archived" / "an-archived-note.md")
    assert not is_exempt(Path("docs/CHANGELOG.md")), "the directory rule subsumes it"
    assert not is_exempt(Path("docs/wiki.md"))
    assert not is_exempt(Path("docs/reports.md")), "a FILE named reports is not the directory"


def test_the_exempt_layers_are_the_ones_the_report_names():
    """The constants drive both the skip and the printed domain line, so a layer cannot be exempt
    in the walk and unmentioned in the report.

    MUTATION: hard-code either list inside `is_exempt`; this stays green, which is why the
    assertion is on the CONSTANTS rather than on the printed line — see the comment below.
    """
    # A weaker test than it looks, and deliberately kept: it pins the pair, not the wiring. The
    # wiring is pinned by `test_the_history_layer_is_exempt` above, which calls the
    # predicate. Together they cover what neither does alone.
    assert EXEMPT_DIRS == ("reports",)
    assert EXEMPT_FILES == (), "the file rule is kept as a mechanism with no current member"


def test_a_template_is_matched_whole_rather_than_truncated_at_a_paren():
    r"""The tail charset excludes `(`, `)` and `=`, so a flat scan cut a real template short and
    reported a token nobody wrote — `artifact:{Segment` for
    `t"artifact:{Segment(kind)}/{Segment(subtype)},{digest}"`. Sixteen findings were this.

    Matching a `{…}` group WHOLE fixes it without loosening the match in prose, and it upgrades
    the class rather than hiding it: the full template reaches `refused()`, its holes normalize
    to atoms, and a genuinely wrong template is still a finding — which the second case pins.

    MUTATION: drop the `\{[^{}\n]*\}|` alternation from `BARE`; the first assertion reddens.
    """
    line = 'composes t"artifact:{Segment(kind)}/{Segment(subtype)},{digest}" for the key'
    assert list(spellings(line)) == ["artifact:{Segment(kind)}/{Segment(subtype)},{digest}"]
    # ...and having been matched whole, it is correctly NOT a finding
    assert not refused("artifact:{Segment(kind)}/{Segment(subtype)},{digest}")
    # while a template that is wrong still is
    assert refused("artifact:{Segment(kind)}:{digest}")


def test_a_foreign_identity_keeps_its_marker_through_the_tokenizer():
    r"""`refused()` short-circuits on a `$`-prefixed token, but only if the token still HAS its
    `$`. A capture that starts at a letter turns `$awaitEvent:…` in prose into `awaitEvent:…`,
    the guard sees nothing foreign, and Absurd's identity is reported as our drift. A guard the
    tokenizer feeding it can defeat is not a guard.

    MUTATION: drop the `\$?` from `BARE`; the first assertion reddens.
    """
    line = "a park on $awaitEvent:gather:0,0;rec:0;review:m1 presents it inside the head term"
    assert list(spellings(line)) == ["$awaitEvent:gather:0,0;rec:0;review:m1"]
    assert not refused("$awaitEvent:gather:0,0;rec:0;review:m1")
    # …and the same payload WITHOUT the marker is ours, and is refused
    assert refused("awaitEvent:gather:0,0;rec:0;review:m1")


def test_a_key_inside_a_diagram_is_read_as_the_diagram_shows_it():
    """Mermaid escapes a message's `;` as `#59;`, and `#` is also the occurrence sigil, so the
    raw text reads as occurrence 59. Inside a fence the sweep reads the rendered key."""
    source = "```mermaid\nsequenceDiagram\n  K-->>S: gather:0,1#59;rec:0/step#59;leaf\n```\n"
    lines = [text for _, text in markdown_prose(source)]
    assert "  K-->>S: gather:0,1;rec:0/step;leaf" in lines
    assert not any("#59" in text for text in lines)
    outside = [text for _, text in markdown_prose("step#59;leaf\n")]
    assert outside == ["step#59;leaf"], "outside a fence the text is the text"
