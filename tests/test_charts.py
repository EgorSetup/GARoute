"""Числа и оси четырёх графиков сверяются с ручными поколениями."""

import pytest

from algorithm.genetic.evolution import GenerationSnapshot, OperationStats
from algorithm.models import Individual, Route, RouteMetrics
from visualization.charts import (
    build_convergence_chart, build_diversity_chart, build_operations_chart,
    build_quality_chart, cost_statistics, diversity_percentage,
)


def person(cost: float, *edges) -> Individual:
    return Individual(Route(tuple(edges)), cost, RouteMetrics(100, 10, 0))


def snapshot(generation: int, population: tuple[Individual, ...],
             stats: OperationStats | None = None) -> GenerationSnapshot:
    if stats is None:
        stats = OperationStats(0, 0, 0, 0, 0, len(population), 0, 0, ())
    return GenerationSnapshot(population, 0, stats, (), generation)


def test_cost_statistics() -> None:
    history = (
        snapshot(0, (person(1), person(2), person(3))),
        snapshot(1, (person(1), person(1), person(4), person(10))),
    )
    stats = cost_statistics(history)
    assert (stats[0].best, stats[0].mean, stats[0].median, stats[0].worst) == (1, 2, 2, 3)
    assert (stats[1].best, stats[1].mean, stats[1].median, stats[1].worst) == (1, 4, 2.5, 10)
    convergence = build_convergence_chart(history)
    assert tuple(convergence.data[0].x) == (0, 1)
    assert tuple(convergence.data[0].y) == (1, 1)
    assert tuple(convergence.data[1].y) == (2, 4)
    quality = build_quality_chart(history)
    assert tuple(quality.data[1].y) == (2, 2.5)
    assert tuple(quality.data[2].y) == (3, 10)
    assert "безразмерная" in convergence.layout.yaxis.title.text


def test_diversity_definition() -> None:
    a, b, c = (1, 2, 0), (2, 3, 0), (3, 4, 0)
    identical = (person(1, a, b), person(2, a, b))
    disjoint = (person(1, a), person(2, c))
    partial = (person(1, a, b), person(2, b, c))
    assert diversity_percentage(identical) == 0
    assert diversity_percentage(disjoint) == 100
    assert diversity_percentage(partial) == pytest.approx(100 * 2 / 3)
    assert diversity_percentage((person(1, a),)) == 0
    assert diversity_percentage((person(1), person(2))) == 0
    chart = build_diversity_chart((snapshot(0, identical), snapshot(1, partial)))
    assert tuple(chart.data[0].y) == pytest.approx((0, 100 * 2 / 3))
    assert tuple(chart.layout.yaxis.range) == (0, 100)


def test_operator_counts() -> None:
    stats = OperationStats(5, 2, 4, 3, 1, 2, 7, 1, (("invalid", 7),))
    chart = build_operations_chart((snapshot(3, (person(1), person(2)), stats),))
    assert [trace.name for trace in chart.data] == [
        "Успешные скрещивания", "Успешные мутации", "Отбраковки",
    ]
    assert [tuple(trace.y) for trace in chart.data] == [(2,), (3,), (7,)]
    assert all(tuple(trace.x) == (3,) for trace in chart.data)


def test_empty_history() -> None:
    assert cost_statistics(()) == ()
    for builder in (build_convergence_chart, build_diversity_chart,
                    build_quality_chart, build_operations_chart):
        chart = builder(())
        assert all(not trace.x and not trace.y for trace in chart.data)


def test_no_repair_series() -> None:
    chart = build_operations_chart((snapshot(0, (person(1),)),))
    assert len(chart.data) == 3
    assert not any("ремонт" in trace.name.lower() for trace in chart.data)
