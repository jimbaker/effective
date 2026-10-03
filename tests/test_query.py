"""`search_query`: a value spliced into a search is text, and never an operator."""

import pytest

from effective.query import search_query

FORGED = "2025 -site:discuss.python.org"


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        ("PEP 750 template strings", "PEP 750 template strings"),
        (FORGED, "2025 site discuss.python.org"),
        ('"exact phrase" +must', "exact phrase must"),
        ("cats OR dogs AND birds", "cats or dogs and birds"),
        ("intitle:secret", "intitle secret"),
        (3.14, "3.14"),
    ],
    ids=[
        "plain words",
        "a forged exclusion",
        "quotes and a plus",
        "operators",
        "a field",
        "number",
    ],
)
def test_a_plain_hole_renders_its_words_with_every_operator_disarmed(value, rendered):
    assert search_query(t"{value}") == rendered


def test_static_text_keeps_the_operators_its_author_wrote():
    year = "2025"
    assert (
        search_query(t"site:peps.python.org pep 750 {year}") == "site:peps.python.org pep 750 2025"
    )


@pytest.mark.parametrize(
    ("value", "rendered"),
    [("3.14", '"3.14"'), (FORGED, '"2025 -site:discuss.python.org"'), ('say "hi"', '"say hi"')],
)
def test_a_phrase_hole_quotes_its_value_as_one_phrase(value, rendered):
    assert search_query(t"{value:phrase}") == rendered


def test_an_either_hole_joins_phrases_with_or():
    values = ["2029", "2031 OR 2030"]
    assert search_query(t"launched {values:either}") == 'launched "2029" OR "2031 OR 2030"'


def test_an_exclude_hole_puts_site_before_each_host():
    hosts = ["peps.python.org", "news.example"]
    assert search_query(t"pep 750 {hosts:exclude}") == (
        "pep 750 -site:peps.python.org -site:news.example"
    )


def test_whitespace_collapses_to_one_space():
    empty = ""
    assert search_query(t"  pep   {empty}  750 ") == "pep 750"


@pytest.mark.parametrize(
    "render",
    [
        lambda: search_query(t"{['bad host!']:exclude}"),
        lambda: search_query(t"{['Upper.Example']:exclude}"),
        lambda: search_query(t"{'peps.python.org':exclude}"),
        lambda: search_query(t"{'x':upper}"),
        lambda: search_query(t"{'x'!r}"),
        lambda: search_query(t"{True}"),
        lambda: search_query(t"{['a']}"),
    ],
    ids=[
        "a host with a space",
        "an uppercase host",
        "a string to exclude",
        "unknown spec",
        "a conversion",
        "a bool",
        "a list as text",
    ],
)
def test_a_hole_the_grammar_has_no_rendering_for_is_refused(render):
    with pytest.raises(ValueError, match="search_query"):
        render()
