"""Длина, время и одна формула безразмерной стоимости маршрута."""

from dataclasses import dataclass, field
from math import fsum, isclose, isfinite
from typing import Mapping

from algorithm.genetic.validation import validate_route
from algorithm.evaluation.normalization import NormalizationScales, validate_manual_scales
from algorithm.models import EdgeId, RoadEdge, Route, RouteContext, RouteMetrics
from config import PRESET_WEIGHTS, TRAFFIC_SPEED_FACTORS
from data.traffic import speed_factor


Weights = tuple[float, float, float]


def _finite_number(value: object) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return isfinite(value)
    except OverflowError:
        return False


def _valid_weights(weights: object) -> bool:
    return (
        isinstance(weights, (tuple, list)) and len(weights) == 3
        and all(_finite_number(value) and 0 <= value <= 1
                for value in weights)
        and isclose(fsum(weights), 1.0, rel_tol=0, abs_tol=1e-9)
    )


def resolve_preset(name: str, custom_weights: Weights | None = None) -> Weights:
    """Вернуть веса T/L/C; готовые пресеты не принимают ручную подмену."""

    if name in PRESET_WEIGHTS:
        weights = PRESET_WEIGHTS[name]
    elif name == "custom":
        weights = custom_weights
    else:
        raise ValueError("unknown fitness preset")
    if not _valid_weights(weights):
        raise ValueError("invalid fitness weights")
    return tuple(weights)


def evaluate_cost(metrics: RouteMetrics, weights: Weights,
                  scales: NormalizationScales) -> float:
    """Вычислить J=wT·T/Tref+wL·L/Lref+wC·C/Cref; меньший J лучше."""

    if not _valid_weights(weights):
        raise ValueError("invalid fitness weights")
    if validate_manual_scales(scales).errors:
        raise ValueError("invalid normalization scales")
    if (any(not _finite_number(value)
            for value in (metrics.time_s, metrics.length_m, metrics.complexity))
            or metrics.time_s <= 0 or metrics.length_m <= 0 or metrics.complexity < 0):
        raise ValueError("invalid route metrics")
    cost = fsum((weights[0] * metrics.time_s / scales.time_s,
                 weights[1] * metrics.length_m / scales.length_m,
                 weights[2] * metrics.complexity / scales.complexity))
    if not isfinite(cost) or cost < 0:
        raise ValueError("nonfinite fitness cost")
    return cost


@dataclass(frozen=True, slots=True)
class TimeModel:
    """Нормализованные рёбра и фактические уровни для одного расчёта."""

    roads: Mapping[EdgeId, RoadEdge]
    levels: Mapping[EdgeId, int] = field(default_factory=dict)
    factors: Mapping[int, float] = field(default_factory=lambda: TRAFFIC_SPEED_FACTORS.copy())


def _valid_edges(route: Route, context: RouteContext, model: TimeModel):
    report = validate_route(route, context)
    if not report.valid:
        raise ValueError(f"invalid route: {', '.join(report.codes)}")
    for edge_id in route.edges:
        edge = model.roads.get(edge_id)
        if edge is None:
            raise ValueError(f"missing normalized edge: {edge_id!r}")
        level = model.levels.get(edge_id, 1)
        if level == 11:
            raise ValueError(f"closed edge: {edge_id!r}")
        yield edge_id, edge, speed_factor(level, model.factors)


def calculate_length(route: Route, context: RouteContext, time_model: TimeModel) -> float:
    """Сумма нормализованных длин в метрах, только для допустимого маршрута."""

    total = 0.0
    for edge_id, edge, _ in _valid_edges(route, context, time_model):
        if not isfinite(edge.length_m) or edge.length_m <= 0:
            raise ValueError(f"invalid length: {edge_id!r}")
        total += edge.length_m
    if not isfinite(total):
        raise ValueError("nonfinite route length")
    return total


def calculate_travel_time(route: Route, context: RouteContext, time_model: TimeModel) -> float:
    """Σ 3.6·L/(v·k) секунд; дополнительных задержек Q02 нет."""

    total = 0.0
    for edge_id, edge, factor in _valid_edges(route, context, time_model):
        if not isfinite(edge.length_m) or edge.length_m <= 0:
            raise ValueError(f"invalid length: {edge_id!r}")
        speed = edge.speed_kph
        if speed is None or not isfinite(speed) or speed <= 0:
            raise ValueError(f"unknown or invalid speed: {edge_id!r}")
        total += 3.6 * edge.length_m / (speed * factor)
    if not isfinite(total):
        raise ValueError("nonfinite route time")
    return total


def unknown_required_speeds(context: RouteContext, time_model: TimeModel) -> tuple[EdgeId, ...]:
    """Список доступных рёбер области без обязательной скорости для блокировки запуска."""

    unknown: list[EdgeId] = []
    for edge_id in context.graph.edges(keys=True):
        if edge_id in context.closed_edges or time_model.levels.get(edge_id, 1) == 11:
            continue
        if context.graph[edge_id[0]][edge_id[1]][edge_id[2]].get("car_access") is not True:
            continue
        edge = time_model.roads.get(edge_id)
        if edge is None:
            unknown.append(edge_id)
            continue
        speed = edge.speed_kph
        if speed is None or not isfinite(speed) or speed <= 0:
            unknown.append(edge_id)
    return tuple(unknown)
