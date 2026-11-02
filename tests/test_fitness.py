"""Единая стоимость J и веса четырёх пресетов."""

from math import inf, nan

import pytest

from algorithm.evaluation.fitness import evaluate_cost, resolve_preset
from algorithm.evaluation.normalization import NormalizationScales
from algorithm.models import RouteMetrics


SCALES = NormalizationScales(time_s=100, length_m=1000, complexity=10)


def test_preset_weights_and_manual_cost():
    assert resolve_preset("fast") == (1, 0, 0)
    assert resolve_preset("short") == (0, 1, 0)
    assert resolve_preset("easy") == (0.15, 0.10, 0.75)
    assert resolve_preset("custom", (0.2, 0.3, 0.5)) == (0.2, 0.3, 0.5)
    metrics = RouteMetrics(1500, 200, 5)
    assert evaluate_cost(metrics, resolve_preset("fast"), SCALES) == 2
    assert evaluate_cost(metrics, resolve_preset("short"), SCALES) == 1.5
    assert evaluate_cost(metrics, resolve_preset("custom", (0.2, 0.3, 0.5)), SCALES) == 1.1
    assert evaluate_cost(metrics, resolve_preset("easy"), SCALES) == pytest.approx(0.825)


def test_minimization_order_differs_by_preset():
    quick_long = RouteMetrics(2000, 100, 4)
    slow_short = RouteMetrics(1000, 200, 4)
    assert evaluate_cost(quick_long, resolve_preset("fast"), SCALES) < evaluate_cost(
        slow_short, resolve_preset("fast"), SCALES)
    assert evaluate_cost(slow_short, resolve_preset("short"), SCALES) < evaluate_cost(
        quick_long, resolve_preset("short"), SCALES)


@pytest.mark.parametrize("name,weights", [
    ("custom", None), ("custom", (0.2, 0.3, 0.4)),
    ("custom", (-0.1, 0.5, 0.6)), ("custom", (nan, 0.5, 0.5)),
    ("missing", None),
])
def test_invalid_weights_or_preset(name, weights):
    with pytest.raises(ValueError):
        resolve_preset(name, weights)


@pytest.mark.parametrize("metrics,scales", [
    (RouteMetrics(1, 0, 0), SCALES),
    (RouteMetrics(1, 1, -1), SCALES),
    (RouteMetrics(1, inf, 0), SCALES),
    (RouteMetrics(1, 1, 0), NormalizationScales(0, 1, 1)),
])
def test_invalid_metrics_or_scales(metrics, scales):
    with pytest.raises(ValueError):
        evaluate_cost(metrics, (1, 0, 0), scales)
