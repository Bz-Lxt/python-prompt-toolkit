"""
Tests for the Unicode Bidirectional Algorithm (UAX #9) support.

The algorithm itself is validated against the full Unicode 16
``BidiCharacterTest.txt`` conformance suite by an external runner; the tests
in this file cover the public API, the golden vectors of representative
cases, the stateful visual caret navigation, the Buffer integration and the
BiDiProcessor. The logical text must never be mutated: reordering is always
display-only through index permutations.
"""

from __future__ import annotations

import random

import pytest

from prompt_toolkit.bidi import (
    BidiNavigator,
    _carets_at_visual_x,
    analyze_line,
    get_base_direction,
    has_rtl,
    logical_caret_to_visual,
    visual_caret_for_column,
)
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.processors import BiDiProcessor, TransformationInput

# -- Golden UBA vectors (levels and visual order taken from the Unicode 16
# BidiCharacterTest.txt, so they are authoritative rather than derived). ---


def test_pure_rtl_line_is_reversed():
    result = analyze_line("אבג")

    assert result.base_direction == "R"
    assert result.levels == (1, 1, 1)
    assert result.text == "גבא"
    assert result.visual_to_logical == (2, 1, 0)
    assert result.logical_to_visual == (2, 1, 0)


def test_mixed_ltr_rtl_line():
    result = analyze_line("שלום world")

    assert result.text == "world םולש"
    # Hebrew run on level 1, embedded Latin run on level 2.
    assert result.levels == (1, 1, 1, 1, 1, 2, 2, 2, 2, 2)


def test_arabic_letters_with_arabic_numbers():
    # EN digits inside an RTL paragraph resolve to AN (level 2).
    result = analyze_line("123 بيت")

    assert result.levels == (2, 2, 2, 1, 1, 1, 1)
    assert result.text == "تيب 123"


def test_bracket_pair_n0_golden_vector():
    # Official vector: inner brackets around Hebrew are retyped to the RTL
    # level, the outer brackets around the whole LTR span stay at level 0.
    result = analyze_line("a(א[ב]!)b", base_direction="L")

    assert result.levels == (0, 0, 1, 1, 1, 1, 0, 0, 0)
    assert result.visual_to_logical == (0, 1, 5, 4, 3, 2, 6, 7, 8)


def test_empty_and_filled_bracket_pairs_golden_vector():
    result = analyze_line("a()א(ב)", base_direction="L")

    assert result.levels == (0, 0, 0, 1, 1, 1, 1)
    assert result.visual_to_logical == (0, 1, 2, 6, 5, 4, 3)


def test_canonically_equivalent_bracket_pairs():
    # U+2329 and U+3008 are canonical equivalents (BD7): pairing U+2329 with
    # U+3009 must give exactly the same result as the regular U+3008/U+3009
    # pair. Golden vector from BidiCharacterTest.txt.
    angled = analyze_line("א \u2329ב.1\u232a", base_direction="L")
    canonical = analyze_line("א \u3008ב.1\u3009", base_direction="L")

    assert angled.levels == canonical.levels == (1, 1, 1, 1, 1, 2, 1)
    assert angled.visual_to_logical == canonical.visual_to_logical == (
        6,
        5,
        4,
        3,
        2,
        1,
        0,
    )


def test_bracket_stack_overflow_suppresses_n0():
    # The bracket stack holds at most 63 openers. At 64 nested pairs the
    # whole N0 rule is aborted: no bracket gets retyped and the brackets
    # resolve exactly like unpaired neutral characters (here '#').
    overflow = "L" + "(" * 64 + "א" + ")" * 64
    neutral_only = "L" + "#" * 64 + "א" + "#" * 64

    result = analyze_line(overflow, base_direction="L")
    assert result.levels == analyze_line(
        neutral_only, base_direction="L"
    ).levels
    # Output stays a valid permutation of the logical characters.
    assert sorted(result.visual_to_logical) == list(range(len(result.text)))
    assert len(result.levels) == len(result.text)

    # One level less still fits the stack and therefore differs.
    within_limit = "L" + "(" * 63 + "א" + ")" * 63
    assert analyze_line(within_limit, base_direction="L").levels != result.levels


def test_l4_mirroring_only_on_odd_level_brackets():
    logical = "א(ב)"

    mirrored = analyze_line(logical, mirror=True)
    assert mirrored.text == "(ב)א"

    raw = analyze_line(logical, mirror=False)
    assert raw.text == ")ב(א"
    # Mirroring changes glyphs only, never the index permutation.
    assert mirrored.visual_to_logical == raw.visual_to_logical


def test_permutation_invariants_and_caching():
    rng = random.Random(1234)
    alphabet = ["a", "b", "א", "ב", " ", "1", "2", "م"]
    for _ in range(200):
        line = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 30)))
        result = analyze_line(line)
        n = len(line)

        assert sorted(result.visual_to_logical) == list(range(n))
        assert sorted(result.logical_to_visual) == list(range(n))
        assert all(
            result.logical_to_visual[result.visual_to_logical[i]] == i
            for i in range(n)
        )
        assert len(result.levels) == n
        # The reordered display text is a permutation of the logical text.
        assert "".join(line[i] for i in result.visual_to_logical) == result.text
        # Caching returns an equal result.
        assert analyze_line(line) is result or analyze_line(line) == result


def test_has_rtl_and_base_direction():
    assert has_rtl("abc") is False
    assert has_rtl("aא") is True
    assert has_rtl("م") is True
    # Arabic-Indic digits are weak RTL (AN).
    assert has_rtl("١٢٣") is True

    assert get_base_direction("abc") == "L"
    assert get_base_direction("א") == "R"
    # No strong character: default to LTR.
    assert get_base_direction("123") == "L"


def test_pure_ltr_is_identity():
    result = analyze_line("plain latin 123")

    assert result.text == "plain latin 123"
    assert all(level == 0 for level in result.levels)
    assert result.visual_to_logical == tuple(range(len(result.text)))


# -- Caret mapping ----------------------------------------------------------


def test_visual_caret_for_column_roundtrip():
    result = analyze_line("שלום world")
    for x in range(len(result.text) + 1):
        caret = visual_caret_for_column(result, x)
        assert 0 <= caret <= len(result.text)
        # The caret must be a boundary that can be rendered at this column.
        assert caret in _carets_at_visual_x(result, x)


def test_logical_caret_edges_use_real_rendered_cells():
    # In a mixed line the last logical character is not rendered at a visual
    # line edge; the boundary caret must point at its actual cell edge rather
    # than blindly at column 0 or len(line).
    result = analyze_line("b2גב")

    start_x = logical_caret_to_visual(result, 0)
    end_x = logical_caret_to_visual(result, len(result.text))
    assert 0 in _carets_at_visual_x(result, start_x)
    assert len(result.text) in _carets_at_visual_x(result, end_x)


# -- BidiNavigator -----------------------------------------------------------


def test_navigator_pure_rtl_walks():
    result = analyze_line("אבג")

    # Logical start is the visual right edge: right arrow is a no-op there.
    navigator = BidiNavigator.for_logical_caret(result, 0)
    assert navigator.visual_x == 3
    walk = [navigator.logical_caret]
    for _ in range(4):
        walk.append(navigator.right())
    assert walk == [0, 0, 0, 0, 0]

    # Left arrow walks through every logical caret and clamps at the edge.
    navigator = BidiNavigator.for_logical_caret(result, 0)
    walk = [navigator.logical_caret]
    for _ in range(4):
        walk.append(navigator.left())
    assert walk == [0, 1, 2, 3, 3]


def test_navigator_states_are_always_valid():
    rng = random.Random(42)
    alphabet = ["a", "b", "c", "א", "ב", "ג", " ", "1", "2", "3"]
    for _ in range(500):
        line = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 25)))
        result = analyze_line(line)
        n = len(line)

        # Every straight walk keeps a valid (visual_x, caret) state, moves
        # the visual x monotonically and clamps at the line edges.
        navigator = BidiNavigator.for_logical_caret(result, 0)
        previous_x = -1
        for _ in range(n + 2):
            assert navigator.visual_x >= previous_x
            assert (
                navigator.logical_caret
                in _carets_at_visual_x(result, navigator.visual_x)
            )
            previous_x = navigator.visual_x
            navigator.right()
        assert navigator.visual_x == n

        navigator = BidiNavigator.for_logical_caret(result, n)
        previous_x = n + 1
        for _ in range(n + 2):
            assert navigator.visual_x <= previous_x
            assert (
                navigator.logical_caret
                in _carets_at_visual_x(result, navigator.visual_x)
            )
            previous_x = navigator.visual_x
            navigator.left()
        assert navigator.visual_x == 0


def test_navigator_bfs_reaches_every_caret():
    rng = random.Random(7)
    alphabet = ["a", "b", "א", "ב", " ", "1", "م", "ب"]
    for _ in range(300):
        line = "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 20)))
        result = analyze_line(line)
        n = len(line)

        start = BidiNavigator.for_logical_caret(result, 0)
        seen: set[tuple[int, int]] = set()
        queue = [(start.visual_x, start.logical_caret)]
        while queue:
            x, caret = queue.pop()
            if (x, caret) in seen:
                continue
            seen.add((x, caret))
            left = BidiNavigator(result, x, caret)
            left.left()
            queue.append((left.visual_x, left.logical_caret))
            right = BidiNavigator(result, x, caret)
            right.right()
            queue.append((right.visual_x, right.logical_caret))

        assert {caret for _, caret in seen} == set(range(n + 1))


def test_navigator_count_matches_repeated_single_steps():
    result = analyze_line("HELLO שלום 123")
    for count in (1, 2, 3, 5):
        stepped = BidiNavigator.for_logical_caret(result, 0)
        for _ in range(count):
            stepped.right()
        counted = BidiNavigator.for_logical_caret(result, 0)
        counted.right(count)
        assert (
            counted.visual_x == stepped.visual_x
            and counted.logical_caret == stepped.logical_caret
        )


# -- Buffer integration ------------------------------------------------------


@pytest.fixture
def bidi_buffer():
    return Buffer(bidi=True)


def test_buffer_bidi_disabled_keeps_logical_movement():
    buffer = Buffer(bidi=False)
    buffer.text = "שלום world"
    buffer.cursor_right()
    assert buffer.cursor_position == 1
    buffer.cursor_left()
    assert buffer.cursor_position == 0


def test_buffer_horizontal_walks_are_valid_and_complete(bidi_buffer):
    rng = random.Random(99)
    alphabet = ["a", "b", "א", "ב", " ", "1", "م"]
    lines = ["HELLO שלום 123", "مكة 123 بيت", "b2גב", "אבג"]
    lines += [
        "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 18)))
        for _ in range(200)
    ]

    for line in lines:
        result = analyze_line(line)
        n = len(line)
        bidi_buffer.text = line

        # BFS through the real Buffer movement methods reaches every caret.
        bidi_buffer.cursor_position = 0
        bidi_buffer._bidi_visual_x = None
        seen: set[tuple[int, int]] = set()
        queue = [
            (bidi_buffer.cursor_position, bidi_buffer._bidi_navigator().visual_x)
        ]
        while queue:
            column, visual_x = queue.pop()
            if (column, visual_x) in seen:
                continue
            seen.add((column, visual_x))

            bidi_buffer.cursor_position = column
            bidi_buffer._bidi_visual_x = visual_x
            bidi_buffer.cursor_left()
            queue.append(
                (bidi_buffer.cursor_position, bidi_buffer._bidi_visual_x)
            )

            bidi_buffer.cursor_position = column
            bidi_buffer._bidi_visual_x = visual_x
            bidi_buffer.cursor_right()
            queue.append(
                (bidi_buffer.cursor_position, bidi_buffer._bidi_visual_x)
            )

        assert {column for column, _ in seen} == set(range(n + 1))
        for column, visual_x in seen:
            assert column in _carets_at_visual_x(result, visual_x)

        # The logical text is never touched by navigation.
        assert bidi_buffer.text == line


def test_buffer_start_and_end_of_line(bidi_buffer):
    bidi_buffer.text = "שלום world"
    result = analyze_line(bidi_buffer.text)

    bidi_buffer.cursor_position = 3
    bidi_buffer.cursor_to_start_of_line()
    assert bidi_buffer._bidi_visual_x == 0
    assert bidi_buffer.cursor_position in _carets_at_visual_x(result, 0)

    bidi_buffer.cursor_to_end_of_line()
    assert bidi_buffer._bidi_visual_x == len(result.text)
    assert bidi_buffer.cursor_position == len(result.text)


def test_buffer_vertical_movement_preserves_visual_x(bidi_buffer):
    bidi_buffer.text = "abc\nשלום\n123"

    bidi_buffer.cursor_position = 0
    bidi_buffer.cursor_down()
    assert bidi_buffer.document.cursor_position_row == 1
    assert bidi_buffer.preferred_column == 0
    bidi_buffer.cursor_down()
    assert bidi_buffer.document.cursor_position_row == 2
    bidi_buffer.cursor_up()
    assert bidi_buffer.document.cursor_position_row == 1


def test_buffer_visual_state_invalidated_on_external_changes(bidi_buffer):
    bidi_buffer.text = "שלום world"
    bidi_buffer.cursor_right()
    bidi_buffer.cursor_right()
    assert bidi_buffer._bidi_visual_x is not None

    # An arbitrary cursor jump invalidates the remembered side of the
    # boundary; the next navigation re-derives a consistent state.
    bidi_buffer.cursor_position = 0
    navigator = bidi_buffer._bidi_navigator()
    assert (
        navigator.logical_caret
        in _carets_at_visual_x(analyze_line(bidi_buffer.text), navigator.visual_x)
    )

    bidi_buffer.cursor_right()
    bidi_buffer.insert_text("X")
    assert bidi_buffer._bidi_visual_x is None
    assert bidi_buffer.preferred_column is None


def test_buffer_text_never_mutates_on_navigation(bidi_buffer):
    bidi_buffer.text = "שלום world 123 بيت"
    original = bidi_buffer.text
    for _ in range(100):
        bidi_buffer.cursor_left()
        bidi_buffer.cursor_right()
    bidi_buffer.cursor_to_start_of_line()
    bidi_buffer.cursor_to_end_of_line()
    assert bidi_buffer.text == original


# -- BiDiProcessor -----------------------------------------------------------


def _transformation_input(line: str, fragments):
    document = Document(line, 0)
    return TransformationInput(
        buffer_control=None,
        document=document,
        lineno=0,
        source_to_display=lambda index: index,
        fragments=fragments,
        width=80,
        height=1,
    )


def test_processor_reorders_and_keeps_styles():
    line = "שלום world 123"
    fragments = [("", "שלום"), ("class:red", " world "), ("bold", "123")]

    transformation = BiDiProcessor().apply_transformation(
        _transformation_input(line, fragments)
    )
    result = analyze_line(line)

    assert "".join(text for _, text in transformation.fragments) == result.text

    # Each rendered character keeps the style of its logical source fragment.
    expected_styles = [
        "class:red" if 4 <= index <= 10 else ("bold" if index >= 11 else "")
        for index in range(len(line))
    ]
    rendered_styles = "".join(style for style, _ in transformation.fragments)
    assert rendered_styles == "".join(
        expected_styles[index] for index in result.visual_to_logical
    )


def test_processor_mapping_closures_are_bounded():
    line = "שלום world"
    transformation = BiDiProcessor().apply_transformation(
        _transformation_input(line, [("", line)])
    )
    n = len(line)

    for caret in range(n + 1):
        visual_x = transformation.source_to_display(caret)
        assert 0 <= visual_x <= n
        assert 0 <= transformation.display_to_source(visual_x) <= n


def test_processor_is_noop_without_rtl():
    transformation = BiDiProcessor().apply_transformation(
        _transformation_input("hello world", [("", "hello world")])
    )
    assert "".join(text for _, text in transformation.fragments) == "hello world"
