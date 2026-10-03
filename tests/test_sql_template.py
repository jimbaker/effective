"""The SQL boundary as a `Template` processor (`effective.sql`) — the sqlite3 half of the seam.

Two things are pinned here. The **safety** property: a value hole is data at the driver, never
text in the statement, so the Bobby Tables payload round-trips as a stored string and the table it
names survives. And the **refusals** the structure makes possible: a hole whose *position* is
wrong (inside a literal, an identifier, a comment), a hole whose *spec* the DSL does not define, a
hole whose *value* cannot be one parameter. Each refusal exists because the processor is handed a
`Template` — position, arity, and the author's own `expression` — rather than a finished string.
"""

import sqlite3

import pytest

from effective.sql import PLACEHOLDER, Query, SqlTemplateError, bind, quote_identifier

# --- composition: structure out, data aside -----------------------------------------------


def test_a_bare_hole_becomes_a_placeholder_and_a_parameter():
    task_id, name = "t-1", "extract"
    query = bind(t"SELECT state FROM checkpoints WHERE task_id={task_id} AND name={name}")
    assert query == Query(
        "SELECT state FROM checkpoints WHERE task_id=? AND name=?", ("t-1", "extract")
    )


def test_parameters_keep_template_order_across_adjacent_and_repeated_holes():
    a, b = 1, 2
    # Adjacent holes are unambiguous here (unlike a key composition): parameters are positional,
    # so `??` splits by arity, not by a delimiter.
    assert bind(t"SELECT {a}{b}, {a}").parameters == (1, 2, 1)


def test_a_template_with_no_holes_composes_to_itself_with_no_parameters():
    assert bind(t"SELECT 1") == Query("SELECT 1", ())


def test_the_query_splats_into_a_driver_call():
    sql, parameters = bind(t"SELECT {1}")
    assert (sql, parameters) == ("SELECT ?", (1,))


def test_statics_pass_through_verbatim_including_multiline_concatenation():
    name = "n"
    query = bind(
        t"SELECT state FROM checkpoints "
        t"WHERE name={name} "
        t"ORDER BY rowid"  # adjacent t-strings concatenate at compile time (PEP 750)
    )
    assert query.sql == "SELECT state FROM checkpoints WHERE name=? ORDER BY rowid"


# --- identifiers: the one place a value legitimately becomes structure ----------------------


def test_an_identifier_hole_is_quoted_into_the_text_not_bound():
    table = "c_default"
    assert bind(t"SELECT 1 FROM {table:i}") == Query('SELECT 1 FROM "c_default"', ())


def test_an_identifier_carrying_a_quote_is_escaped_by_doubling():
    table = 'evil"; DROP TABLE checkpoints; --'
    assert bind(t"SELECT 1 FROM {table:i}").sql == (
        'SELECT 1 FROM "evil""; DROP TABLE checkpoints; --"'
    )


@pytest.mark.parametrize("bad", [42, None, b"c_default"])
def test_a_non_string_identifier_is_refused(bad):
    with pytest.raises(SqlTemplateError, match="not a str"):
        bind(t"SELECT 1 FROM {bad:i}")


def test_an_empty_identifier_is_refused():
    table = ""
    with pytest.raises(SqlTemplateError, match="empty"):
        bind(t"SELECT 1 FROM {table:i}")


def test_a_nul_bearing_identifier_is_refused():
    with pytest.raises(SqlTemplateError, match="NUL"):
        quote_identifier("c_\x00default", "table")


# --- the refusals a Template makes possible and an f-string cannot --------------------------


def test_a_hole_inside_a_string_literal_is_refused_and_names_the_fix():
    prefix = "budget-grant:r1:"
    with pytest.raises(SqlTemplateError) as caught:
        bind(t"SELECT 1 FROM events WHERE name LIKE '{prefix}%'")
    assert "quoted SQL string literal" in str(caught.value)
    assert "prefix" in str(caught.value)  # the hole's own source expression


def test_a_hole_inside_a_quoted_identifier_is_refused():
    queue = "default"
    with pytest.raises(SqlTemplateError, match="quoted identifier"):
        bind(t'SELECT 1 FROM "c_{queue}"')


def test_a_hole_inside_a_line_comment_is_refused():
    note = "why"
    with pytest.raises(SqlTemplateError, match="`--` comment"):
        bind(t"SELECT 1 -- {note}\n")


def test_a_hole_inside_a_block_comment_is_refused():
    note = "why"
    with pytest.raises(SqlTemplateError, match=r"`/\* \*/` comment"):
        bind(t"SELECT /* {note} */ 1")


def test_a_closed_literal_or_comment_leaves_the_next_hole_bindable():
    # The scanner tracks regions, so an *earlier* literal/comment must not poison the rest.
    name = "n"
    assert bind(t"SELECT 'a?b' /* c */ -- d\n, {name}").parameters == ("n",)


def test_a_doubled_quote_inside_a_literal_does_not_leave_the_literal_open():
    name = "n"
    assert bind(t"SELECT 'it''s', {name}").parameters == ("n",)


def test_a_manual_placeholder_in_the_static_sql_is_refused():
    name = "n"
    with pytest.raises(SqlTemplateError, match="manual"):
        bind(t"SELECT 1 FROM t WHERE a=? AND b={name}")


def test_a_placeholder_inside_a_string_literal_is_data_not_a_placeholder():
    assert bind(t"SELECT 'a?b'") == Query("SELECT 'a?b'", ())


def test_an_undefined_format_spec_is_refused_rather_than_silently_dropped():
    n = 7
    with pytest.raises(SqlTemplateError, match="format spec this DSL does not define"):
        bind(t"SELECT {n:03d}")


def test_a_conversion_is_refused_rather_than_binding_a_different_value():
    value = "x"
    with pytest.raises(SqlTemplateError, match="conversion"):
        bind(t"SELECT {value!r}")


@pytest.mark.parametrize("ids", [[1, 2], (1, 2), {1, 2}, {"a": 1}])
def test_a_container_value_is_refused_with_the_fix_named(ids):
    with pytest.raises(SqlTemplateError, match="one scalar per placeholder"):
        bind(t"SELECT 1 FROM t WHERE id IN {ids}")


# --- against the real driver: the safety property, executed -------------------------------


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE checkpoints (task_id TEXT, name TEXT, state TEXT)")
    yield connection
    connection.close()


def test_a_bobby_tables_value_is_stored_as_data_and_the_table_survives(conn):
    payload = "'); DROP TABLE checkpoints; --"
    task_id = "t-1"
    conn.execute(
        *bind(t"INSERT INTO checkpoints (task_id, name, state) VALUES ({task_id}, {payload}, '1')")
    )
    read = bind(t"SELECT name FROM checkpoints WHERE task_id={task_id}")
    stored = conn.execute(*read).fetchone()
    assert stored == (payload,)  # the payload is a value, not statement text


def test_a_like_pattern_binds_whole_which_is_what_the_literal_refusal_pushes_you_to(conn):
    conn.execute("INSERT INTO checkpoints VALUES ('t-1', 'budget-grant:r1,0', '1')")
    conn.execute("INSERT INTO checkpoints VALUES ('t-1', 'extract', '1')")
    pattern = "budget-grant:r1,%"
    read = bind(t"SELECT name FROM checkpoints WHERE name LIKE {pattern}")
    rows = conn.execute(*read).fetchall()
    assert rows == [("budget-grant:r1,0",)]


def test_an_identifier_hole_reaches_the_driver_as_a_real_table_name(conn):
    table = "checkpoints"
    conn.execute("INSERT INTO checkpoints VALUES ('t-1', 'extract', '1')")
    assert conn.execute(*bind(t"SELECT count(*) FROM {table:i}")).fetchone() == (1,)


def test_the_placeholder_constant_is_the_paramstyle_the_driver_actually_speaks(conn):
    assert conn.execute(f"SELECT {PLACEHOLDER}", (5,)).fetchone() == (5,)
