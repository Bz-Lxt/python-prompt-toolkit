from __future__ import annotations

from contextlib import contextmanager

import pytest

from prompt_toolkit.application import Application
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.filters import Condition
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.layout import FocusDirection, InvalidLayoutError, Layout
from prompt_toolkit.layout.containers import (
    ConditionalContainer,
    DynamicContainer,
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
)
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.output import DummyOutput


def test_layout_class():
    c1 = BufferControl()
    c2 = BufferControl()
    c3 = BufferControl()
    win1 = Window(content=c1)
    win2 = Window(content=c2)
    win3 = Window(content=c3)

    layout = Layout(container=VSplit([HSplit([win1, win2]), win3]))

    # Listing of windows/controls.
    assert list(layout.find_all_windows()) == [win1, win2, win3]
    assert list(layout.find_all_controls()) == [c1, c2, c3]

    # Focusing something.
    layout.focus(c1)
    assert layout.has_focus(c1)
    assert layout.has_focus(win1)
    assert layout.current_control == c1
    assert layout.previous_control == c1

    layout.focus(c2)
    assert layout.has_focus(c2)
    assert layout.has_focus(win2)
    assert layout.current_control == c2
    assert layout.previous_control == c1

    layout.focus(win3)
    assert layout.has_focus(c3)
    assert layout.has_focus(win3)
    assert layout.current_control == c3
    assert layout.previous_control == c2

    # Pop focus. This should focus the previous control again.
    layout.focus_last()
    assert layout.has_focus(c2)
    assert layout.has_focus(win2)
    assert layout.current_control == c2
    assert layout.previous_control == c1


def test_create_invalid_layout():
    with pytest.raises(InvalidLayoutError):
        Layout(HSplit([]))


#
# Geometric (spatial) focus movement.
#


class _FixedSizeDummyOutput(DummyOutput):
    "Dummy output that reports a configurable screen size."

    def __init__(self, columns: int, rows: int) -> None:
        self._size = Size(columns=columns, rows=rows)

    def get_size(self) -> Size:
        return self._size


@contextmanager
def rendered_layout(layout: Layout, columns: int = 80, rows: int = 40):
    """
    Render the given layout once (without starting the event loop) and make the
    application current, so that the layout geometry is up to date.
    """
    with create_pipe_input() as pipe_input:
        app = Application(
            layout=layout,
            output=_FixedSizeDummyOutput(columns, rows),
            input=pipe_input,
            full_screen=True,
        )

        # We render outside of a running event loop, so don't actually start
        # any background tasks. (A minimal task stub is enough; buffer history
        # loading expects an object with `add_done_callback`.)
        class _FakeTask:
            def __init__(self, coroutine) -> None:
                coroutine.close()

            def add_done_callback(self, callback) -> None:
                pass

        def create_background_task(coroutine, **kw):
            return _FakeTask(coroutine)

        app.create_background_task = create_background_task

        with set_app(app):
            app.renderer.render(app, app.layout)
            app.layout.update_parents_relations()
            yield app


def make_window() -> Window:
    return Window(BufferControl())


def test_directional_focus_splits():
    # Layout:
    #   +-------+---+
    #   |   a   |   |
    #   +-------+ b |
    #   |   c   |   |
    #   +-------+---+
    a, b, c = make_window(), make_window(), make_window()
    layout = Layout(VSplit([HSplit([a, c]), b]))

    with rendered_layout(layout):
        positions = layout.visible_windows_to_write_positions

        assert positions[a] != positions[b]

        layout.focus(a)
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == b

        layout.focus_neighboring_window(FocusDirection.LEFT)
        assert layout.current_window == a

        layout.focus_neighboring_window(FocusDirection.DOWN)
        assert layout.current_window == c

        layout.focus_neighboring_window(FocusDirection.UP)
        assert layout.current_window == a

        # String directions are accepted as well.
        layout.focus_neighboring_window("right")
        assert layout.current_window == b


def test_directional_focus_no_target_retains_focus():
    a, b, c = make_window(), make_window(), make_window()
    layout = Layout(VSplit([HSplit([a, c]), b]))

    with rendered_layout(layout):
        # Nothing above the top-left window.
        layout.focus(a)
        stack_size = len(layout._stack)
        layout.focus_neighboring_window(FocusDirection.UP)
        assert layout.current_window == a
        # The focus stack must not have been touched.
        assert len(layout._stack) == stack_size

        # Nothing to the right of the right-hand window (the window stacked
        # vertically in the left column does not count).
        layout.focus(b)
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == b

        # Nothing below the bottom-left window.
        layout.focus(c)
        layout.focus_neighboring_window(FocusDirection.DOWN)
        assert layout.current_window == c


def test_directional_focus_without_render():
    # Before the first render, there is no geometry at all: keep the focus.
    a, b = make_window(), make_window()
    layout = Layout(VSplit([a, b]))

    assert layout.visible_windows_to_write_positions == {}
    layout.focus_neighboring_window(FocusDirection.RIGHT)
    assert layout.current_window == a


def test_directional_focus_picks_geometrically_nearest():
    # Two floats on the right hand side, at an equal (symmetric) distance from
    # the vertical midpoint of the background window.
    background = make_window()
    top_right = make_window()
    bottom_right = make_window()
    root = FloatContainer(
        content=background,
        floats=[
            Float(top=14, left=45, width=20, height=4, content=top_right),
            Float(top=21, left=45, width=20, height=4, content=bottom_right),
        ],
    )
    layout = Layout(root)

    with rendered_layout(layout):
        # Both floats are symmetric around the vertical midpoint of the
        # background, hence at exactly the same distance. The tie is resolved
        # in reading order: the upper window wins, and the choice is stable.
        for _ in range(3):
            layout.focus(background)
            layout.focus_neighboring_window(FocusDirection.RIGHT)
            assert layout.current_window == top_right

        # From the upper float, moving left returns into the overlapping
        # background window (non-modal floats do not trap the focus).
        layout.focus_neighboring_window(FocusDirection.LEFT)
        assert layout.current_window == background

        # Moving down from the background reaches the lower float.
        layout.focus(background)
        layout.focus_neighboring_window(FocusDirection.DOWN)
        assert layout.current_window == bottom_right


def test_directional_focus_stays_inside_modal_float():
    background = make_window()
    modal_left = make_window()
    modal_right = make_window()
    root = FloatContainer(
        content=background,
        floats=[
            Float(
                top=5,
                left=5,
                width=70,
                height=10,
                content=VSplit([modal_left, modal_right], modal=True),
            )
        ],
    )
    layout = Layout(root)

    with rendered_layout(layout):
        layout.focus(modal_left)

        # Only windows of the modal area are focusable now.
        assert set(layout.get_visible_focusable_windows()) == {
            modal_left,
            modal_right,
        }

        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == modal_right

        # The background is geometrically to the right / below, but must not
        # be reached while the modal float has the focus.
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == modal_right
        layout.focus_neighboring_window(FocusDirection.DOWN)
        assert layout.current_window == modal_right

        layout.focus_neighboring_window(FocusDirection.LEFT)
        assert layout.current_window == modal_left
        layout.focus_neighboring_window(FocusDirection.UP)
        assert layout.current_window == modal_left


def test_directional_focus_after_resize():
    a, b = make_window(), make_window()
    layout = Layout(VSplit([a, b]))

    with rendered_layout(layout, columns=40, rows=20):
        assert layout.visible_windows_to_write_positions[a].width == 20
        layout.focus(a)
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == b

    # Re-render the same layout at a different size. The geometry has to be
    # taken from the fresh render.
    with rendered_layout(layout, columns=30, rows=20):
        assert layout.visible_windows_to_write_positions[a].width == 15
        assert layout.visible_windows_to_write_positions[b].xpos == 15
        layout.focus(a)
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == b


def test_directional_focus_dynamic_containers():
    state = {"show": True}
    a = make_window()
    b = make_window()
    holder = {
        "container": VSplit(
            [a, ConditionalContainer(b, filter=Condition(lambda: state["show"]))]
        )
    }
    layout = Layout(DynamicContainer(lambda: holder["container"]))

    with rendered_layout(layout):
        assert b in layout.visible_windows_to_write_positions
        layout.focus(a)
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == b

    # Hide the right window and render again: there is no target anymore, the
    # focus is retained.
    state["show"] = False
    with rendered_layout(layout):
        assert b not in layout.visible_windows_to_write_positions
        layout.focus(a)
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == a

    # A newly added (different) window is reachable immediately.
    c = make_window()
    holder["container"] = VSplit([a, c])
    with rendered_layout(layout):
        assert c in layout.visible_windows_to_write_positions
        layout.focus(a)
        layout.focus_neighboring_window(FocusDirection.RIGHT)
        assert layout.current_window == c
