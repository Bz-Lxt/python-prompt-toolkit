from __future__ import annotations

from prompt_toolkit.key_binding.key_processor import KeyPressEvent
from prompt_toolkit.layout.layout import FocusDirection

__all__ = [
    "focus_next",
    "focus_previous",
    "focus_up",
    "focus_down",
    "focus_left",
    "focus_right",
]

E = KeyPressEvent


def focus_next(event: E) -> None:
    """
    Focus the next visible Window.
    (Often bound to the `Tab` key.)
    """
    event.app.layout.focus_next()


def focus_previous(event: E) -> None:
    """
    Focus the previous visible Window.
    (Often bound to the `BackTab` key.)
    """
    event.app.layout.focus_previous()


def focus_up(event: E) -> None:
    """
    Focus the nearest visible Window above the currently focused Window, using
    the actual on-screen geometry. The focus stays within the current modal
    area and is left unchanged when no Window lies above.
    """
    event.app.layout.focus_neighboring_window(FocusDirection.UP)


def focus_down(event: E) -> None:
    """
    Focus the nearest visible Window below the currently focused Window, using
    the actual on-screen geometry. The focus stays within the current modal
    area and is left unchanged when no Window lies below.
    """
    event.app.layout.focus_neighboring_window(FocusDirection.DOWN)


def focus_left(event: E) -> None:
    """
    Focus the nearest visible Window to the left of the currently focused
    Window, using the actual on-screen geometry. The focus stays within the
    current modal area and is left unchanged when no Window lies to the left.
    """
    event.app.layout.focus_neighboring_window(FocusDirection.LEFT)


def focus_right(event: E) -> None:
    """
    Focus the nearest visible Window to the right of the currently focused
    Window, using the actual on-screen geometry. The focus stays within the
    current modal area and is left unchanged when no Window lies to the right.
    """
    event.app.layout.focus_neighboring_window(FocusDirection.RIGHT)
