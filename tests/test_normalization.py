"""Эталоны по полной P0 и валидация ручных значений."""

from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from algorithm.evaluation.normalization import (
    NormalizationScales, compute_scales, validate_manual_scales,
)
from algorithm.models import RouteMetrics


def test_initial_means_and_zero_complexity_scale():
    metrics = (RouteMetrics(1000, 100, 0), RouteMetrics(3000, 300, 0))
    scales = compute_scales(metrics)
    assert scales == NormalizationScales(200, 2000, 1)
    assert compute_scales((RouteMetrics(1000, 100, 2), RouteMetrics(3000, 300, 4))) == (
        NormalizationScales(200, 2000, 3)
    )


def test_scales_are_immutable_and_inputs_not_mutated():
    source = [RouteMetrics(100, 10, 1), RouteMetrics(200, 20, 2)]
    scales = compute_scales(source)
    source[0] = RouteMetrics(900, 90, 9)
    assert scales == NormalizationScales(15, 150, 1.5)
    with pytest.raises(FrozenInstanceError):
        scales.time_s = 100


@pytest.mark.parametrize("scales", [
    NormalizationScales(0, 10, 1), NormalizationScales(10, 0, 1),
    NormalizationScales(10, 10, 0), NormalizationScales(nan, 10, 1),
    NormalizationScales(10, inf, 1), NormalizationScales("10", 10, 1),
])
def test_manual_invalid(scales):
    assert validate_manual_scales(scales).errors


@pytest.mark.parametrize("metrics", [
    (), (RouteMetrics(0, 10, 1),), (RouteMetrics(10, 0, 1),),
    (RouteMetrics(10, 10, -1),), (RouteMetrics(10, nan, 1),),
])
def test_zero_time_length_and_other_invalid_metrics(metrics):
    with pytest.raises(ValueError):
        compute_scales(metrics)
