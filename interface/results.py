"""Пустой каркас результатов и воспроизведения без вымышленных запусков."""

from nicegui import ui

from interface.controls import UIState
from visualization.charts import render_empty_charts


def _disabled_playback_button(label: str, icon: str) -> None:
    button = ui.button(label, icon=icon).props("outline dense").disable()
    button.tooltip("История поколений пока отсутствует")


def build_playback_controls(state: UIState) -> None:
    """Разместить все команды перемотки, заблокировав их без истории."""

    with ui.element("section").classes("results-section"):
        ui.label("Воспроизведение поколений").classes("panel-heading")
        if not state.playback.has_history:
            ui.label("Истории поколений нет. Перемотка станет доступна после расчёта.").classes(
                "panel-copy"
            )
        with ui.element("div").classes("playback-actions"):
            _disabled_playback_button("В начало", "first_page")
            _disabled_playback_button("Шаг назад", "skip_previous")
            _disabled_playback_button("Воспроизвести", "play_arrow")
            _disabled_playback_button("Пауза", "pause")
            _disabled_playback_button("Шаг вперёд", "skip_next")
            _disabled_playback_button("В конец", "last_page")
        ui.label("Поколение: нет данных").classes("panel-copy")
        ui.slider(min=0, max=max(1, state.playback.generation_count - 1), value=0).disable().classes(
            "w-full"
        )
        ui.select(
            {0.5: "0,5×", 1.0: "1×", 2.0: "2×"}, label="Скорость воспроизведения",
            value=None,
        ).props("outlined dense disable").classes("setting-field")


def build_results_panel(state: UIState) -> None:
    """Разместить будущие карточки, историю и четыре пустых графика."""

    ui.label("Результаты").classes("panel-heading")
    with ui.element("section").classes("empty-block"):
        ui.label("Расчёты ещё не выполнены").classes("panel-heading")
        ui.label("После запуска здесь появятся лучшие маршруты и их метрики.").classes(
            "panel-copy"
        )
    with ui.element("section").classes("results-section"):
        ui.label("Карточки лучших маршрутов").classes("panel-heading")
        ui.label("Пока нет запусков для сравнения: L, T, C и J не вычислены.").classes(
            "panel-copy"
        )
    build_playback_controls(state)
    with ui.element("section").classes("results-section"):
        ui.label("Графики поколений").classes("panel-heading")
        render_empty_charts()
