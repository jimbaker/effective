"""The vendored tdom's acceptance gate: what it renders and what it refuses.

`infra/tdom` is 0.1.17 plus one patch, and an upgrade is judged by this file. A well-formed
template renders the same bytes, and a malformed one raises the same diagnostic:

| case               | template              | outcome                                         |
|--------------------|-----------------------|-------------------------------------------------|
| attribute          | `<g class="{k}"></g>` | renders; what the patch fixes on 3.14.7         |
| text               | `<p>{k}</p>`          | renders                                         |
| empty              | `<br>`                | renders self-closed                             |
| less               | `<`                   | refused; a flushing parser renders `&lt;`       |
| unclosed           | `<div>`               | refused                                         |
| mismatch           | `<div></span>`        | refused                                         |
| attribute-unclosed | `<g class="{k}`       | refused, naming the open quote                  |
| comment            | `<!--`                | refused; a flushing parser renders a comment    |
"""

import re

import pytest
from tdom import html

k = "x"


@pytest.mark.parametrize(
    ("template", "rendered"),
    [
        (t'<g class="{k}"></g>', '<g class="x"></g>'),
        (t"<p>{k}</p>", "<p>x</p>"),
        (t"<br>", "<br />"),
    ],
    ids=["attribute", "text", "empty"],
)
def test_a_well_formed_template_renders(template, rendered):
    assert str(html(template)) == rendered


@pytest.mark.parametrize(
    ("template", "diagnostic"),
    [
        (t"<", "Parser expects more data, is the template valid html?"),
        (t"<div>", "Invalid HTML structure: unclosed tags remain."),
        (t"<div></span>", "Mismatched closing tag </span> for element <div>."),
        (
            t'<g class="{k}',
            "Parser expects more data, maybe you left an attribute quote unclosed?",
        ),
        (t"<!--", "Parser expects more data, is the template valid html?"),
    ],
    ids=["less", "unclosed", "mismatch", "attribute-unclosed", "comment"],
)
def test_a_malformed_template_is_refused_with_its_diagnostic(template, diagnostic):
    with pytest.raises(ValueError, match=re.escape(diagnostic)) as refused:
        html(template)
    assert str(refused.value) == diagnostic
