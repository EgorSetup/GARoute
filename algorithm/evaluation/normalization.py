"""Неизменяемые эталоны нормализации для одного расчёта."""

from dataclasses import dataclass
from math import fsum, isfinite
from typing import Sequence

from algorithm.models import RouteMetrics, ValidationReport


@dataclass(frozen=True, slots=True)
class NormalizationScales:
    """Tref в секундах, Lref в метрах, Cref в баллах."""

    time_s: float
    length_m: float
    complexity: float


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def validate_manual_scales(scales: NormalizationScales) -> ValidationReport:
    """Отвергнуть нули, NaN, бесконечность и неподходящие типы."""

    errors = tuple(
        f"{name}: требуется положительное конечное число"
        for name, value in (
            ("Tref (с)", scales.time_s),
            ("Lref (м)", scales.length_m),
            ("Cref (баллы)", scales.complexity),
        )
        if not _finite_number(value) or value <= 0
    )
    return ValidationReport(errors=errors)


def compute_scales(metrics: Sequence[RouteMetrics]) -> NormalizationScales:
    """Вычислить средние по полной P0 без скрытой замены нулевых T/L."""

    if not metrics:
        raise ValueError("empty P0 metrics")
    for item in metrics:
        if any(not _finite_number(value)
               for value in (item.time_s, item.length_m, item.complexity)):
            raise ValueError("nonfinite P0 metrics")
        if item.time_s <= 0 or item.length_m <= 0 or item.complexity < 0:
            raise ValueError("invalid P0 metrics")
    count = len(metrics)
    scales = NormalizationScales(
        fsum(item.time_s for item in metrics) / count,
        fsum(item.length_m for item in metrics) / count,
        max(1.0, fsum(item.complexity for item in metrics) / count),
    )
    if validate_manual_scales(scales).errors:
        raise ValueError("invalid automatic scales")
    return scales
