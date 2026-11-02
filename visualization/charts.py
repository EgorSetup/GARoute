"""Четыре места графиков до появления расчётных чисел."""

from nicegui import ui


CHART_TITLES = (
    "Сходимость стоимости J",
    "Разнообразие популяции, %",
    "Распределение качества J",
    "История операций",
)


def render_empty_charts() -> None:
    """Показать названия графиков без нарисованных данных."""

    with ui.column().classes("w-full chart-list"):
        for title in CHART_TITLES:
            with ui.card().classes("w-full chart-placeholder"):
                ui.label(title).classes("penalty-title")
                ui.label("Нет данных поколений. График появится после расчёта.").classes(
                    "panel-copy"
                )
