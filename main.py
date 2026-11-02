"""Точка запуска локального интерфейса GARoute."""

from nicegui import ui

from interface.controls import UIState
from interface.layout import build_layout


@ui.page("/")
def index() -> None:
    """Создать отдельное состояние для открытой страницы."""

    build_layout(UIState())


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="GARoute", host="127.0.0.1", port=8080, reload=False, show=False)
