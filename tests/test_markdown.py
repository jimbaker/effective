"""`markdown` and `table`: a value is literal text, and a table reads aligned rendered or not."""

import pytest

from effective.markdown import markdown, table


@pytest.mark.parametrize(
    ("value", "rendered"),
    [
        ("plain", "plain"),
        ("a | b", "a \\| b"),
        ("*bold* _it_ `code`", "\\*bold\\* \\_it\\_ \\`code\\`"),
        ("<script>", "\\<script\\>"),
        ("[link](x)", "\\[link\\](x)"),
        ("x\n# heading", "x \\# heading"),
        ("back\\slash", "back\\\\slash"),
    ],
    ids=[
        "plain",
        "pipe",
        "emphasis and code",
        "markup",
        "link",
        "newline then heading",
        "backslash",
    ],
)
def test_a_hole_renders_as_literal_inline_text(value, rendered):
    assert markdown(t"{value}") == rendered


def test_static_text_keeps_its_markup_and_a_nested_template_composes():
    name = "a*b"
    inner = t"**{name}**"
    assert markdown(t"# Report\n{inner} done") == "# Report\n**a\\*b** done"


def test_a_hole_takes_its_format_spec_and_conversion():
    ratio, label = 0.5, "x"
    assert markdown(t"{ratio:.0%} {label!r}") == "50% 'x'"


def test_a_table_pads_every_column_to_its_widest_cell():
    rendered = table(["policy", "ops"], [["worth | corroborating", 60], ["breadth", 103]])
    assert rendered.splitlines() == [
        "| policy                 | ops |",
        "| ---------------------- | --- |",
        "| worth \\| corroborating | 60  |",
        "| breadth                | 103 |",
    ]


def test_a_wide_character_takes_two_columns():
    assert table(["name"], [["日本"], ["abcd"]]).splitlines() == [
        "| name |",
        "| ---- |",
        "| 日本 |",
        "| abcd |",
    ]


def test_a_cell_template_keeps_its_markup_and_escapes_a_static_pipe():
    n = 2
    assert table(["correct"], [[t"**{n}** of {n} | all"]]).splitlines()[2] == (
        "| **2** of 2 \\| all |"
    )


def test_a_row_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="rows \\[1\\] are not 2 cells wide"):
        table(["a", "b"], [["1", "2"], ["3"]])
