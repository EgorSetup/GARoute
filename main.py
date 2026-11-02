"""Точка запуска локального интерфейса GARoute."""

import os
from dataclasses import replace

from nicegui import ui

from interface.controls import UIState
from interface.layout import build_layout


@ui.page("/")
def index() -> None:
    """Создать отдельное состояние для открытой страницы."""

    state = UIState(demo_mode=os.environ.get("GAROUTE_MAP_EVENT_TEST") == "1")
    if os.environ.get("GAROUTE_CHANGE_TEST") == "1":
        state.change_demo_mode = True
        state.config = replace(
            state.config,
            fitness=replace(state.config.fitness, normalization_mode="manual",
                            time_reference_s=600, length_reference_m=3000,
                            complexity_reference=20),
        )
        state.run_results.append(object())  # Только тестовая история; не результат ГА.
    build_layout(state)


if __name__ in {"__main__", "__mp_main__"}:
    ui.run(title="GARoute", host="127.0.0.1", port=8080, reload=False, show=False)
