"""Четыре графика только из сохранённых чисел поколений."""

from dataclasses import dataclass
from statistics import fmean, median
from typing import Callable, Sequence

import plotly.graph_objects as go
from nicegui import ui

from algorithm.genetic.evolution import GenerationSnapshot
from algorithm.models import Individual


CHART_TITLES = (
    "Сходимость стоимости J",
    "Разнообразие популяции, %",
    "Распределение качества J",
    "История операций",
)


@dataclass(frozen=True, slots=True)
class CostStatistics:
    generation: int
    best: float
    mean: float
    median: float
    worst: float


def cost_statistics(history: Sequence[GenerationSnapshot]) -> tuple[CostStatistics, ...]:
    """Min/mean/median/max J полной сохранённой популяции."""

    return tuple(CostStatistics(
        snapshot.generation,
        min(person.j for person in snapshot.population),
        fmean(person.j for person in snapshot.population),
        median(person.j for person in snapshot.population),
        max(person.j for person in snapshot.population),
    ) for snapshot in history)


def diversity_percentage(population: Sequence[Individual]) -> float:
    """Среднее попарное расстояние Жаккара направленных EdgeId в процентах."""

    count = len(population)
    if count < 2:
        return 0.0
    edges = tuple(set(person.route.edges) for person in population)
    distance_sum = 0.0
    for first in range(count):
        for second in range(first + 1, count):
            union = edges[first] | edges[second]
            if union:
                distance_sum += 1 - len(edges[first] & edges[second]) / len(union)
    return 100 * distance_sum * 2 / (count * (count - 1))


def _figure(title: str, y_title: str) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        title=title, xaxis_title="Поколение", yaxis_title=y_title,
        margin=dict(l=50, r=20, t=45, b=45), height=260,
        template="plotly_white", legend=dict(orientation="h", y=-0.28),
    )
    return figure


def build_convergence_chart(history: Sequence[GenerationSnapshot]) -> go.Figure:
    figure = _figure(CHART_TITLES[0], "J, безразмерная стоимость")
    stats = cost_statistics(history)
    figure.add_scatter(x=[s.generation for s in stats], y=[s.best for s in stats],
                       mode="lines+markers", name="Лучшая J")
    figure.add_scatter(x=[s.generation for s in stats], y=[s.mean for s in stats],
                       mode="lines+markers", name="Средняя J")
    return figure


def build_diversity_chart(history: Sequence[GenerationSnapshot]) -> go.Figure:
    figure = _figure(CHART_TITLES[1], "Среднее расстояние Жаккара, %")
    figure.update_yaxes(range=[0, 100])
    figure.add_scatter(x=[s.generation for s in history],
                       y=[diversity_percentage(s.population) for s in history],
                       mode="lines+markers", name="Разнообразие")
    return figure


def build_quality_chart(history: Sequence[GenerationSnapshot]) -> go.Figure:
    figure = _figure(CHART_TITLES[2], "J, безразмерная стоимость")
    stats = cost_statistics(history)
    for name, values in (("Лучшая J", [s.best for s in stats]),
                         ("Медианная J", [s.median for s in stats]),
                         ("Худшая J", [s.worst for s in stats])):
        figure.add_scatter(x=[s.generation for s in stats], y=values,
                           mode="lines+markers", name=name)
    return figure


def build_operations_chart(history: Sequence[GenerationSnapshot]) -> go.Figure:
    figure = _figure(CHART_TITLES[3], "Число событий")
    for name, values in (
        ("Успешные скрещивания", [s.stats.crossover_changes for s in history]),
        ("Успешные мутации", [s.stats.mutation_changes for s in history]),
        ("Отбраковки", [s.stats.rejected_candidates for s in history]),
    ):
        figure.add_bar(x=[s.generation for s in history], y=values, name=name)
    figure.update_layout(barmode="group")
    return figure


BUILDERS = (build_convergence_chart, build_diversity_chart,
            build_quality_chart, build_operations_chart)


def _plot(figure: go.Figure) -> None:
    """Не вызывать Plotly resize у скрытой вкладки NiceGUI."""

    options = figure.to_plotly_json()
    options["config"] = {"responsive": False}
    ui.plotly(options).classes("w-full")


def render_charts(history_supplier: Callable[[], Sequence[GenerationSnapshot]]) -> Callable[[], None]:
    """Обновлять четыре места и увеличение только при смене истории."""

    figures: list[go.Figure | None] = [None] * 4
    with ui.dialog() as enlarged, ui.card().classes("chart-enlarged"):
        enlarged_body = ui.column().classes("w-full")
        ui.button("Закрыть", on_click=enlarged.close).props("flat")

    def show(index: int) -> None:
        enlarged_body.clear()
        with enlarged_body:
            if figures[index] is None:
                ui.label(CHART_TITLES[index]).classes("panel-heading")
                ui.label("Нет данных поколений.").classes("panel-copy")
            else:
                _plot(figures[index])
        enlarged.open()

    containers = []
    with ui.column().classes("w-full chart-list"):
        for index, title in enumerate(CHART_TITLES):
            with ui.card().classes("w-full chart-placeholder"):
                ui.label(title).classes("penalty-title")
                body = ui.column().classes("w-full")
                containers.append(body)
                ui.button("Увеличить", icon="open_in_full",
                          on_click=lambda i=index: show(i)).props("flat dense")

    def refresh() -> None:
        history = tuple(history_supplier())
        for index, container in enumerate(containers):
            container.clear()
            figures[index] = BUILDERS[index](history) if history else None
            with container:
                if figures[index] is None:
                    ui.label("Нет данных поколений. График появится после расчёта.").classes(
                        "panel-copy"
                    )
                else:
                    _plot(figures[index])

    refresh()
    return refresh
