"""
Bidirectional text support (Unicode Bidirectional Algorithm, UAX #9).

This module implements a self contained, dependency free implementation of
the *display* part of the Unicode Bidirectional Algorithm: it turns a logical
order string (the order in which text is stored and typed) into the visual
order in which it has to be rendered on a terminal, while keeping the original
logical string untouched.

The implementation is deliberately practical rather than exhaustive:

* Paragraph base direction is computed according to rules P2/P3.
* Explicit embedding/override/isolate controls (rules X1-X8) are tracked, but
  formatting characters themselves are never removed from the visual output.
  They are assigned the current embedding level instead. This keeps the
  logical/visual mapping a true permutation, which is required for cursor and
  mouse positioning. (The characters are zero width and invisible anyway.)
* The weak (W1-W7), neutral (N1-N2) and implicit (I1-I2) type rules are
  implemented, together with the full paired bracket rules (BD2-BD18),
  including canonical equivalent bracket matching (BD7) and the bracket
  stack overflow handling mandated by rule BD16.
* Reordering follows rule L2 and mirrored glyphs are substituted according to
  rule L4 for characters that resolve to a right-to-left level.
* Arabic letter shaping (joining) is intentionally not handled here. Joining
  is a rendering concern (it belongs to the terminal/font stack), while this
  module is only concerned with ordering.

The main entry point is :func:`analyze_line`, which returns a
:class:`BidiLine` result. The result contains the visual text as well as the
permutation between logical and visual indices. Everything else in
``prompt_toolkit`` (cursor positioning, mouse handling, selection and line
wrapping) is built on top of that permutation.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .cache import SimpleCache

__all__ = [
    "BidiLine",
    "BidiNavigator",
    "analyze_line",
    "get_base_direction",
    "has_rtl",
    "logical_caret_to_visual",
    "visual_caret_for_column",
]

# Character types, as returned by ``unicodedata.bidirectional``.
L = "L"
R = "R"
AL = "AL"
EN = "EN"
ES = "ES"
ET = "ET"
AN = "AN"
CS = "CS"
B = "B"
S = "S"
WS = "WS"
ON = "ON"
NSM = "NSM"
BN = "BN"

# Explicit directional controls.
LRE = "LRE"
LRO = "LRO"
RLE = "RLE"
RLO = "RLO"
PDF = "PDF"
LRI = "LRI"
RLI = "RLI"
FSI = "FSI"
PDI = "PDI"

_STRONG_TYPES = frozenset({L, R, AL})

# Paired brackets. ``unicodedata`` does not expose the Bidi_Paired_Bracket
# property, so the full list from the Unicode BidiBrackets data is embedded
# here. (Open bracket -> closing bracket.)
_BRACKET_PAIRS: dict[str, str] = {
    "(": ")",  # U+0028
    "[": "]",  # U+005B
    "{": "}",  # U+007B
    "༺": "༻",  # U+0F3A
    "༼": "༽",  # U+0F3C
    "᚛": "᚜",  # U+169B
    "⁅": "⁆",  # U+2045
    "⁽": "⁾",  # U+207D
    "₍": "₎",  # U+208D
    "⌈": "⌉",  # U+2308
    "⌊": "⌋",  # U+230A
    "\u2329": "\u232a",  # U+2329 (canonically equivalent to U+3008)
    "❨": "❩",  # U+2768
    "❪": "❫",  # U+276A
    "❬": "❭",  # U+276C
    "❮": "❯",  # U+276E
    "❰": "❱",  # U+2770
    "❲": "❳",  # U+2772
    "❴": "❵",  # U+2774
    "⟅": "⟆",  # U+27C5
    "⟦": "⟧",  # U+27E6
    "⟨": "⟩",  # U+27E8
    "⟪": "⟫",  # U+27EA
    "⟬": "⟭",  # U+27EC
    "⟮": "⟯",  # U+27EE
    "⦃": "⦄",  # U+2983
    "⦅": "⦆",  # U+2985
    "⦇": "⦈",  # U+2987
    "⦉": "⦊",  # U+2989
    "⦋": "⦌",  # U+298B
    "⦍": "\u2990",  # U+298D -> U+2990
    "⦏": "⦎",  # U+298F
    "⦑": "⦒",  # U+2991
    "⦓": "⦔",  # U+2993
    "⦕": "⦖",  # U+2995
    "⦗": "⦘",  # U+2997
    "⧘": "⧙",  # U+29D8
    "⧚": "⧛",  # U+29DA
    "⧼": "⧽",  # U+29FC
    "⸢": "⸣",  # U+2E22
    "⸤": "⸥",  # U+2E24
    "⸦": "⸧",  # U+2E26
    "⸨": "⸩",  # U+2E28
    "⹕": "⹖",  # U+2E55
    "⹗": "⹘",  # U+2E57
    "⹙": "⹚",  # U+2E59
    "⹛": "⹜",  # U+2E5B
    "〈": "〉",  # U+3008
    "《": "》",  # U+300A
    "「": "」",  # U+300C
    "『": "』",  # U+300E
    "【": "】",  # U+3010
    "〔": "〕",  # U+3014
    "〖": "〗",  # U+3016
    "〘": "〙",  # U+3018
    "〚": "〛",  # U+301A
    "﹙": "﹚",  # U+FE59
    "﹛": "﹜",  # U+FE5B
    "﹝": "﹞",  # U+FE5D
    "（": "）",  # U+FF08
    "［": "］",  # U+FF3B
    "｛": "｝",  # U+FF5B
    "｟": "｠",  # U+FF5F
    "｢": "｣",  # U+FF62
}
_OPEN_BRACKETS = frozenset(_BRACKET_PAIRS)
_CLOSE_BRACKETS = frozenset(_BRACKET_PAIRS.values())
_BRACKET_CLOSING_TO_OPEN = {close: open_ for open_, close in _BRACKET_PAIRS.items()}


def _canonical_bracket(char: str) -> str:
    """
    Return the canonical identity of a bracket character.

    Rule BD7 matches brackets on canonical equivalents as well as on exact
    code points. In practice this only concerns U+2329/U+232A, which are
    canonical singletons of U+3008/U+3009. Every other bracket normalizes to
    itself. Working on these identities makes a closing U+3009 match an
    opening U+2329 (and the reverse) exactly like the reference algorithm.
    """
    decomposed = unicodedata.normalize("NFD", char)
    return decomposed if len(decomposed) == 1 else char


_CANONICAL_BRACKET_PAIRS = {
    _canonical_bracket(open_): _canonical_bracket(close)
    for open_, close in _BRACKET_PAIRS.items()
}
_CANONICAL_CLOSING_TO_OPEN = {
    close: open_ for open_, close in _CANONICAL_BRACKET_PAIRS.items()
}

# Bidi mirroring glyphs (rule L4). This is the common subset of the
# Bidi_Mirroring_Glyph Unicode property.
_MIRRORED: dict[str, str] = {}
for _open, _close in _BRACKET_PAIRS.items():
    _MIRRORED[_open] = _close
    _MIRRORED[_close] = _open

# Maximum amount of bracket pairs, as mandated by the Unicode algorithm.
_BRACKET_PAIR_LIMIT = 63


@dataclass(frozen=True)
class BidiLine:
    """
    The result of analyzing one line of text.

    :param text: The text in visual (display) order.
    :param logical_to_visual: For every logical index, its visual position.
        Length equals the length of the input line; the mapping is a
        permutation, so every visual position is reached exactly once.
    :param visual_to_logical: For every visual position, the logical index.
    :param levels: The resolved embedding level for every logical character.
        Odd levels represent right-to-left characters.
    :param base_direction: The paragraph base direction: ``"L"`` or ``"R"``.
    """

    text: str
    logical_to_visual: tuple[int, ...]
    visual_to_logical: tuple[int, ...]
    levels: tuple[int, ...]
    base_direction: str

    def __post_init__(self) -> None:
        length = len(self.text)
        assert len(self.logical_to_visual) == length
        assert len(self.visual_to_logical) == length
        assert len(self.levels) == length


# Cache of analyzed lines. Rendering the same line multiple times (once for
# measuring, once for drawing, ...) is very common.
_bidi_cache: SimpleCache[tuple[str, str, bool], BidiLine] = SimpleCache(maxsize=64)


def has_rtl(text: str) -> bool:
    """
    Return ``True`` when the text contains any right-to-left character
    (Hebrew, Arabic, ...). Pure left-to-right text can be rendered without
    running the full algorithm.
    """
    for char in text:
        bidi_type = unicodedata.bidirectional(char)
        # Arabic numbers are weak right-to-left characters: their presence
        # also requires running the full algorithm.
        if bidi_type in (R, AL, AN):
            return True
    return False


def get_base_direction(text: str) -> str:
    """
    Determine the paragraph base direction according to rules P2/P3.

    The direction is determined by the first strong character (L, R or AL).
    When no strong character is found, the paragraph defaults to ``"L"``.
    """
    for char in text:
        bidi_type = unicodedata.bidirectional(char)
        if bidi_type in (R, AL):
            return R
        if bidi_type == L:
            return L
    return L


def _char_types(text: str) -> list[str]:
    return [unicodedata.bidirectional(char) or ON for char in text]


def _fsi_direction(text: str, start: int) -> str:
    """
    Resolve the direction of an FSI control at `start`: scan until the matching
    PDI (accounting for nested isolates) and apply P3 to the enclosed text.
    """
    depth = 1
    index = start + 1
    length = len(text)
    while index < length:
        bidi_type = unicodedata.bidirectional(text[index])
        if bidi_type in (LRI, RLI, FSI):
            depth += 1
        elif bidi_type == PDI:
            depth -= 1
            if depth == 0:
                break
        index += 1
    return get_base_direction(text[start + 1 : index])


def _resolve_explicit_levels(
    text: str, types: list[str], base_direction: str
) -> tuple[list[int], list[str]]:
    """
    Apply rules X1-X8. Return the initial embedding level for every character
    and the override status per character (``L``, ``R`` or ``ON`` meaning "no
    override"). Explicit formatting characters are kept and assigned the
    current embedding level.
    """
    base_level = 0 if base_direction == L else 1

    # Stack entries: (embedding level, override status, isolate entry).
    stack: list[tuple[int, str, bool]] = [(base_level, ON, False)]
    levels = [base_level] * len(text)
    overrides: list[str] = [ON] * len(text)

    # Counters used by the X-rules to cap nesting depth.
    embedding_count = 0
    isolate_count = 0
    valid_isolate_count = 0

    for index, bidi_type in enumerate(types):
        level, override, _isolate = stack[-1]
        levels[index] = level
        overrides[index] = override

        if bidi_type in (RLE, LRE, RLO, LRO):
            # R* controls produce an odd level, L* controls an even level.
            new_level = level + (1 if bidi_type in (RLE, RLO) else 2)
            if new_level <= 62 and embedding_count < 60:
                if bidi_type == RLO:
                    new_override: str = R
                elif bidi_type == LRO:
                    new_override = L
                else:
                    new_override = ON
                stack.append((new_level, new_override, False))
                embedding_count += 1
                levels[index] = level
            continue

        if bidi_type in (RLI, LRI, FSI):
            if bidi_type == FSI:
                resolved = _fsi_direction(text, index)
                new_level = level + (1 if resolved == R else 2)
            else:
                new_level = level + (1 if bidi_type == RLI else 2)
            if new_level <= 62 and isolate_count < 63:
                stack.append((new_level, ON, True))
                isolate_count += 1
                valid_isolate_count += 1
                levels[index] = new_level
                overrides[index] = ON
            continue

        if bidi_type == PDI:
            if valid_isolate_count > 0:
                # Pop back to (and including) the last isolate entry.
                while stack and not stack[-1][2]:
                    stack.pop()
                    embedding_count = max(0, embedding_count - 1)
                if stack:
                    stack.pop()
                valid_isolate_count -= 1
                isolate_count = max(0, isolate_count - 1)
            new_level, new_override, _ = stack[-1]
            levels[index] = new_level
            overrides[index] = new_override
            continue

        if bidi_type == PDF:
            if embedding_count > 0 and not stack[-1][2]:
                stack.pop()
                embedding_count -= 1
            continue

        # B (paragraph separator) resets to the paragraph level.
        if bidi_type == B:
            levels[index] = base_level

    # Apply overrides to the character types.
    resolved_types = list(types)
    for index, override in enumerate(overrides):
        if override in (L, R):
            resolved_types[index] = override
        elif types[index] == BN:
            # Keep formatting characters in the sequence; they inherit the
            # current level and behave like neutral characters.
            resolved_types[index] = ON
    return levels, resolved_types


# Types that still count as neutral when N1/N2 run.
_NEUTRAL_TYPES = frozenset({B, S, WS, ON})

# Strong types as seen by rules N0/N1. Numeric types (EN and AN) count as
# right-to-left strong types, mirroring the reference implementation.
_RTL_STRONG = frozenset({R, EN, AN})
_LTR_STRONG = frozenset({L})


def _level_runs(
    levels: list[int], base_level: int
) -> list[tuple[int, int, int, str, str]]:
    """
    Split the paragraph into level runs (rule X10) and compute the start-of-run
    (SOR) and end-of-run (EOR) direction for every run.

    The direction at a run boundary is derived from the parity of the highest
    adjacent embedding level. Returns ``(start, end, run_level, sor, eor)``.
    """
    if not levels:
        return []

    boundaries = [0]
    for index in range(1, len(levels)):
        if levels[index] != levels[index - 1]:
            boundaries.append(index)
    boundaries.append(len(levels))

    runs: list[tuple[int, int, int, str, str]] = []
    base_direction = R if base_level % 2 else L

    for run_index, start in enumerate(boundaries[:-1]):
        end = boundaries[run_index + 1]
        run_level = levels[start]

        if run_index == 0:
            sor = base_direction
        else:
            left_level = levels[start - 1]
            sor = R if max(left_level, run_level) % 2 else L

        if end == len(levels):
            eor = base_direction
        else:
            right_level = levels[end]
            eor = R if max(run_level, right_level) % 2 else L

        runs.append((start, end, run_level, sor, eor))
    return runs


def _apply_weak_rules(types: list[str], levels: list[int], base_level: int) -> None:
    """Apply rules W1-W7 in place, scoped to the level runs."""
    runs = _level_runs(levels, base_level)

    # W1: NSM takes the type of the previous character, or the SOR type when
    # it is the first character of a level run.
    # W2: EN becomes AN when the previous strong character is AL.
    for start, end, _run_level, sor, _eor in runs:
        previous_type = sor
        previous_strong = sor
        for index in range(start, end):
            bidi_type = types[index]
            if bidi_type == NSM:
                types[index] = previous_type
                continue
            if bidi_type == EN and previous_strong == AL:
                types[index] = AN
            elif bidi_type in (R, L, AL):
                previous_strong = bidi_type
            previous_type = types[index]

    # W3: AL becomes R.
    for index in range(len(types)):
        if types[index] == AL:
            types[index] = R

    # W4-W7, again scoped to the level runs.
    for start, end, _run_level, sor, _eor in runs:
        # W4.
        for index in range(start + 1, end - 1):
            bidi_type = types[index]
            if bidi_type not in (ES, CS):
                continue
            if types[index - 1] == EN and types[index + 1] == EN:
                types[index] = EN
            elif (
                bidi_type == CS
                and types[index - 1] == AN
                and types[index + 1] == AN
            ):
                types[index] = AN
            else:
                types[index] = ON

        # W5: an ET sequence adjacent to EN becomes EN.
        index = start
        while index < end:
            if types[index] == ET:
                seq_start = index
                while index < end and types[index] == ET:
                    index += 1
                seq_end = index
                before = types[seq_start - 1] if seq_start > start else sor
                after = types[seq_end] if seq_end < end else ON
                if before == EN or after == EN:
                    for inner in range(seq_start, seq_end):
                        types[inner] = EN
            else:
                index += 1

        # W6.
        for index in range(start, end):
            if types[index] in (ES, ET, CS):
                types[index] = ON

        # W7.
        previous_strong = sor
        for index in range(start, end):
            bidi_type = types[index]
            if bidi_type in (R, L):
                previous_strong = bidi_type
            elif bidi_type == EN and previous_strong == L:
                types[index] = L


def _strong_direction(bidi_type: str) -> str | None:
    """Map a resolved type to an N0/N1 strong direction, if it has one."""
    if bidi_type in (R, EN, AN):
        return R
    if bidi_type == L:
        return L
    return None


def _prior_strong(
    types: list[str], position: int, run_start: int, sor: str
) -> str:
    """First strong direction before ``position`` inside a level run."""
    for index in range(position, run_start - 1, -1):
        direction = _strong_direction(types[index])
        if direction is not None:
            return direction
    return sor


def _apply_bracket_rules(
    text: str,
    types: list[str],
    levels: list[int],
    base_level: int,
) -> None:
    """
    Apply rule N0 (the paired bracket rules BD2-BD18).

    Brackets are paired with a stack. For every matched pair, the direction is
    decided from the strong characters enclosed by the brackets, falling back
    on the prior context and finally on the embedding direction of the level
    run. Only the two bracket characters are retyped here; the neutral text
    between them is resolved afterwards by N1/N2, which is where rule BD18
    takes effect.
    """
    runs = _level_runs(levels, base_level)
    run_of_index = [0] * len(text)
    for run_number, (start, end, _level, _sor, _eor) in enumerate(runs):
        for index in range(start, end):
            run_of_index[index] = run_number

    # Pair the brackets with a single stack for the whole paragraph. The
    # stack stores the canonical identity of each opening bracket, so that
    # canonically equivalent brackets (U+2329 vs U+3008) pair as well.
    stack: list[tuple[int, str]] = []  # (logical index, canonical opener)
    pairs: list[tuple[int, int]] = []  # (open index, close index)
    overflow = False

    for index, char in enumerate(text):
        if char in _OPEN_BRACKETS:
            if len(stack) >= _BRACKET_PAIR_LIMIT:
                # Rule BD16: when the bracket stack overflows, paired
                # bracket processing for the whole run sequence ceases
                # immediately. No bracket is retyped by N0; every bracket
                # is left as a neutral and resolved later by N1/N2.
                overflow = True
                break
            stack.append((index, _canonical_bracket(char)))
        elif char in _CLOSE_BRACKETS:
            expected_open = _CANONICAL_CLOSING_TO_OPEN[_canonical_bracket(char)]
            for stack_index in range(len(stack) - 1, -1, -1):
                open_index, open_char = stack[stack_index]
                if open_char == expected_open:
                    # Any opening brackets above the match can no longer be
                    # paired and are discarded (rules BD8/BD9).
                    del stack[stack_index:]
                    pairs.append((open_index, index))
                    break

    if overflow:
        return

    # Rule N0 resolves outer pairs before inner pairs: sort by the position
    # of the opening bracket.
    pairs.sort()

    for open_index, close_index in pairs:
        run_number = run_of_index[open_index]
        run_start, _run_end, run_level, sor, _eor = runs[run_number]
        embedding_direction = R if run_level % 2 else L
        opposite_direction = L if embedding_direction == R else R

        # Inspect the strong characters enclosed by the brackets. Numeric
        # types count as right-to-left strong types.
        embedding_found = False
        opposite_found = False
        for inner in range(open_index + 1, close_index):
            direction = _strong_direction(types[inner])
            if direction == embedding_direction:
                embedding_found = True
                break
            if direction == opposite_direction:
                opposite_found = True

        if embedding_found:
            direction = embedding_direction
        elif opposite_found:
            # N0 step c1: resolve via the strong type before the pair.
            direction = _prior_strong(types, open_index - 1, run_start, sor)
            if direction != opposite_direction:
                direction = embedding_direction
        else:
            # No strong type between the brackets: leave the pair to N1/N2.
            continue

        types[open_index] = direction
        types[close_index] = direction

        # UBA80 fix-up: combining marks attached to the brackets follow the
        # direction of the bracket they belong to.
        for inner in range(open_index + 1, close_index):
            if unicodedata.bidirectional(text[inner]) == NSM:
                types[inner] = direction
            elif types[inner] not in _NEUTRAL_TYPES:
                break
        run_end = runs[run_of_index[close_index]][1]
        for inner in range(close_index + 1, run_end):
            if unicodedata.bidirectional(text[inner]) == NSM:
                types[inner] = direction
            elif types[inner] not in _NEUTRAL_TYPES:
                break


def _apply_neutral_rules(
    types: list[str], levels: list[int], base_level: int
) -> None:
    """Apply rules N1 and N2 in place, scoped to the level runs."""
    # B and S characters count as neutrals while scanning the context, but
    # their own type is never rewritten: the implicit rules keep them at the
    # embedding level and rule L1 resets them afterwards.
    runs = _level_runs(levels, base_level)
    for start, end, run_level, sor, eor in runs:
        index = start
        while index < end:
            if types[index] not in _NEUTRAL_TYPES:
                index += 1
                continue

            neutral_start = index
            while index < end and types[index] in _NEUTRAL_TYPES:
                index += 1
            neutral_end = index

            # Resolve the strong direction on both sides, using the SOR/EOR
            # at the run boundaries.
            before = sor
            for previous_index in range(neutral_start - 1, start - 1, -1):
                if types[previous_index] not in _NEUTRAL_TYPES:
                    before = types[previous_index]
                    break
            after = eor
            for next_index in range(neutral_end, end):
                if types[next_index] not in _NEUTRAL_TYPES:
                    after = types[next_index]
                    break

            if before in _RTL_STRONG and after in _RTL_STRONG:
                resolved = R
            elif before in _LTR_STRONG and after in _LTR_STRONG:
                resolved = L
            else:
                # N2: take the embedding direction of this level run.
                resolved = R if run_level % 2 else L

            for inner in range(neutral_start, neutral_end):
                # Segment and paragraph separators are neutral boundaries but
                # keep their own type; they are handled by rule L1.
                if types[inner] not in (B, S):
                    types[inner] = resolved


def _apply_implicit_levels(types: list[str], levels: list[int]) -> None:
    """Apply rules I1/I2 in place."""
    for index, bidi_type in enumerate(types):
        level = levels[index]
        if level % 2 == 0:
            # I1
            if bidi_type == R:
                levels[index] = level + 1
            elif bidi_type in (AN, EN):
                levels[index] = level + 2
            # L and everything else stay at the current level.
        else:
            # I2: at an odd level, R characters keep their level while L, EN
            # and AN go up one level.
            if bidi_type in (L, EN, AN):
                levels[index] = level + 1


def _apply_line_reset_rule(
    line: str, types: list[str], levels: list[int], base_level: int
) -> None:
    """
    Rule L1: reset the level of segment/paragraph separators and of trailing
    whitespace to the paragraph level. The *original* character types are
    used, as required by the specification.
    """
    for index, char in enumerate(line):
        if unicodedata.bidirectional(char) in (B, S):
            levels[index] = base_level

    for index in range(len(line) - 1, -1, -1):
        if unicodedata.bidirectional(line[index]) in (WS, B, S, BN):
            levels[index] = base_level
        else:
            break


def _reorder(levels: tuple[int, ...]) -> tuple[int, ...]:
    """
    Rule L2: turn resolved levels into the visual order.

    Returns a tuple mapping visual position to logical index.
    """
    order = list(range(len(levels)))
    if not order:
        return ()

    max_level = max(levels)
    for depth in range(max_level, 0, -1):
        index = 0
        length = len(order)
        while index < length:
            if levels[order[index]] >= depth:
                start = index
                while index < length and levels[order[index]] >= depth:
                    index += 1
                order[start:index] = reversed(order[start:index])
            else:
                index += 1
    return tuple(order)


def analyze_line(
    line: str,
    base_direction: str | None = None,
    *,
    mirror: bool = True,
) -> BidiLine:
    """
    Run the bidirectional algorithm on a single line of text.

    :param line: Logical order text (a single line, without line breaks).
    :param base_direction: ``"L"``, ``"R"`` or ``None`` to auto detect the
        paragraph base direction.
    :param mirror: When ``True``, mirror brackets on right-to-left levels
        according to rule L4.
    """
    if base_direction is None:
        base_direction = get_base_direction(line)

    cache_key = (line, base_direction, mirror)
    return _bidi_cache.get(cache_key, lambda: _analyze_line(line, base_direction, mirror))


def _analyze_line(line: str, base_direction: str, mirror: bool) -> BidiLine:
    base_level = 0 if base_direction == L else 1

    # Fast path: a left-to-right paragraph without any right-to-left character
    # does not need any reordering.
    if base_direction == L and not has_rtl(line):
        length = len(line)
        return BidiLine(
            text=line,
            logical_to_visual=tuple(range(length)),
            visual_to_logical=tuple(range(length)),
            levels=(0,) * length,
            base_direction=base_direction,
        )

    types = _char_types(line)
    levels, types = _resolve_explicit_levels(line, types, base_direction)
    _apply_weak_rules(types, levels, base_level)
    _apply_bracket_rules(line, types, levels, base_level)
    _apply_neutral_rules(types, levels, base_level)
    _apply_implicit_levels(types, levels)
    _apply_line_reset_rule(line, types, levels, base_level)

    visual_to_logical = _reorder(tuple(levels))
    logical_to_visual = [0] * len(line)
    for visual_index, logical_index in enumerate(visual_to_logical):
        logical_to_visual[logical_index] = visual_index

    if mirror:
        chars = list(line)
        for logical_index, level in enumerate(levels):
            if level % 2 and chars[logical_index] in _MIRRORED:
                chars[logical_index] = _MIRRORED[chars[logical_index]]
        line = "".join(chars)

    visual_chars = "".join(line[logical_index] for logical_index in visual_to_logical)

    return BidiLine(
        text=visual_chars,
        logical_to_visual=tuple(logical_to_visual),
        visual_to_logical=visual_to_logical,
        levels=tuple(levels),
        base_direction=base_direction,
    )


# -- Caret movement helpers -------------------------------------------------


def logical_caret_to_visual(result: BidiLine, caret: int) -> int:
    """
    Map a logical caret position (0..len(line), pointing between characters)
    to a visual x position.

    A logical caret sits on the shared edge of two character cells. When the
    two neighbouring characters have opposite directions (a run boundary),
    that single logical boundary corresponds to two different visual
    positions. We canonicalize to the leftmost one so that:

    * the mapping is a function (one visual position per logical caret),
    * arrow walks terminate at the visual line edges instead of cycling
      between the two sides of a run boundary,
    * a click on the visual left edge (``visual_caret_for_column``) maps
      back to the same column.
    """
    length = len(result.text)
    if caret <= 0:
        # Caret at the logical start: it sits on the sole edge contributed
        # by the first logical character at its actual rendered cell.
        if not length:
            return 0
        cell = result.logical_to_visual[0]
        return cell + 1 if result.levels[0] % 2 else cell
    if caret >= length:
        last = length - 1
        cell = result.logical_to_visual[last]
        return cell if result.levels[last] % 2 else cell + 1

    # Edge contributed by the preceding character: RTL characters end on
    # the visual left edge of their cell, LTR characters on the right edge.
    preceding = caret - 1
    prev_cell = result.logical_to_visual[preceding]
    prev_edge = prev_cell if result.levels[preceding] % 2 else prev_cell + 1

    # Edge contributed by the following character: RTL characters start on
    # the visual right edge, LTR characters on the left edge.
    next_cell = result.logical_to_visual[caret]
    next_edge = next_cell + 1 if result.levels[caret] % 2 else next_cell

    return min(prev_edge, next_edge)


def _carets_at_visual_x(result: BidiLine, visual_x: int) -> set[int]:
    """
    All logical carets that can be rendered at visual column ``visual_x``.

    A column sits on the shared edge of two rendered cells. When reordering
    joins two direction runs, that edge can belong to two different logical
    boundaries (a "dual caret" position). The left edge of the cell on the
    right and the right edge of the cell on the left are both collected; at
    the line edges the logical start/end carets are added as well.
    """
    length = len(result.text)
    visual_x = max(0, min(visual_x, length))
    carets: set[int] = set()

    # Left edge of the cell rendered at this column.
    if visual_x < length:
        index = result.visual_to_logical[visual_x]
        carets.add(index + 1 if result.levels[index] % 2 else index)
    else:
        carets.add(length)

    # Right edge of the cell rendered just to the left.
    if visual_x > 0:
        index = result.visual_to_logical[visual_x - 1]
        carets.add(index if result.levels[index] % 2 else index + 1)
    else:
        carets.add(0)

    return carets


class BidiNavigator:
    """
    Stateful visual caret navigation for one analyzed line.

    At a direction run boundary a single logical caret can correspond to two
    visual positions, and a single visual column can host two logical carets.
    Stateless caret math therefore has to pick one side and unavoidably skips
    cells or cycles between the two sides of an embedded run. The navigator
    keeps an explicit ``(visual_x, logical_caret)`` pair, moves the visual x
    one rendered cell per arrow press, and resolves dual-caret columns by
    preferring the logical boundary adjacent to the current caret, falling
    back to the primary (cell-based) mapping. This makes every character
    reachable and guarantees that walks terminate at the visual line edges
    without cycling.
    """

    def __init__(
        self, result: BidiLine, visual_x: int, logical_caret: int | None = None
    ) -> None:
        self.result = result
        self.visual_x = max(0, min(visual_x, len(result.text)))
        if logical_caret is None:
            logical_caret = visual_caret_for_column(result, self.visual_x)
        self._logical_caret = logical_caret

    @classmethod
    def for_logical_caret(
        cls, result: BidiLine, caret: int
    ) -> BidiNavigator:
        "Create a navigator positioned at the canonical visual position."
        visual_x = logical_caret_to_visual(result, caret)
        # The canonical column is one of the caret's own edges, so the caret
        # is always a valid occupant of that column (possibly the secondary
        # one at a dual-caret boundary).
        return cls(result, visual_x, caret)

    @property
    def logical_caret(self) -> int:
        "Logical caret position corresponding to the current state."
        return self._logical_caret

    def _move(self, visual_x: int, going_right: bool) -> int:
        candidates = _carets_at_visual_x(self.result, visual_x)
        if len(candidates) == 1:
            caret = next(iter(candidates))
        else:
            # Prefer crossing into the logically adjacent boundary so that
            # consecutive arrow presses visit each character once.
            adjacent = (
                self._logical_caret + 1 if going_right
                else self._logical_caret - 1
            )
            if adjacent in candidates:
                caret = adjacent
            else:
                caret = visual_caret_for_column(self.result, visual_x)
        self.visual_x = visual_x
        self._logical_caret = caret
        return caret

    def left(self, count: int = 1) -> int:
        "Move ``count`` cells to the visual left; return the new logical caret."
        target_x = max(0, self.visual_x - count)
        if target_x == self.visual_x:
            # Already at the visual edge: keep the current side of the
            # boundary instead of jumping to the other logical caret that
            # happens to share this column.
            return self._logical_caret
        return self._move(target_x, going_right=False)

    def right(self, count: int = 1) -> int:
        "Move ``count`` cells to the visual right; return the new logical caret."
        target_x = min(len(self.result.text), self.visual_x + count)
        if target_x == self.visual_x:
            return self._logical_caret
        return self._move(target_x, going_right=True)


def visual_caret_for_column(result: BidiLine, visual_x: int) -> int:
    """
    Given a target visual column (for instance preserved while moving up or
    down), return the logical caret position closest to that column.
    """
    length = len(result.text)
    visual_x = max(0, min(visual_x, length))
    if visual_x == length:
        return length

    # Prefer the logical boundary on the visual left of the target cell.
    logical_index = result.visual_to_logical[visual_x]
    if result.levels[logical_index] % 2:
        return logical_index + 1
    return logical_index
