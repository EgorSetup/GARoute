"""Длина, время и одна формула безразмерной стоимости маршрута."""

from dataclasses import dataclass, field
from math import fsum, isclose, isfinite
from random import Random
from typing import Mapping

from algorithm.genetic.validation import validate_route
from algorithm.evaluation.normalization import NormalizationScales, validate_manual_scales
from algorithm.models import (
    EasyLimits, EasyReferences, EdgeId, EligibilityReport, RoadEdge, Route, RouteContext,
    RouteMetrics,
)
from config import EASY_DELTA_MAX, PRESET_WEIGHTS, TRAFFIC_SPEED_FACTORS
from data.traffic import speed_factor
from data.road_graph import find_connector


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


def check_easy_constraints(metrics: RouteMetrics, references: EasyReferences,
                           limits: EasyLimits) -> EligibilityReport:
    """Проверить оба жёстких предела; нарушение не превращать в штраф J."""

    values = (metrics.length_m, metrics.time_s, references.length_min_m,
              references.time_min_s, limits.length_delta, limits.time_delta)
    if not all(_finite_number(value) for value in values):
        raise ValueError("nonfinite easy constraints")
    if (metrics.length_m <= 0 or metrics.time_s <= 0
            or references.length_min_m <= 0 or references.time_min_s <= 0
            or not 0 <= limits.length_delta <= EASY_DELTA_MAX
            or not 0 <= limits.time_delta <= EASY_DELTA_MAX):
        raise ValueError("invalid easy constraints")
    length_bound = (1 + limits.length_delta) * references.length_min_m
    time_bound = (1 + limits.time_delta) * references.time_min_s
    if not isfinite(length_bound) or not isfinite(time_bound):
        raise ValueError("nonfinite easy bounds")
    reasons = []
    if metrics.length_m > length_bound:
        reasons.append("easy_length_exceeded")
    if metrics.time_s > time_bound:
        reasons.append("easy_time_exceeded")
    return EligibilityReport(not reasons, tuple(reasons))


@dataclass(frozen=True, slots=True)
class TimeModel:
    """Нормализованные рёбра и фактические уровни для одного расчёта."""

    roads: Mapping[EdgeId, RoadEdge]
    levels: Mapping[EdgeId, int] = field(default_factory=dict)
    factors: Mapping[int, float] = field(default_factory=lambda: TRAFFIC_SPEED_FACTORS.copy())
    modeled_speeds_enabled: bool = False
    modeled_speeds_kph: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedSpeed:
    """Расчётная скорость и её источник; исходный RoadEdge не меняется."""

    speed_kph: float
    source: str


def resolve_speed(edge: RoadEdge, model: TimeModel) -> ResolvedSpeed | None:
    """Модель действует лишь для отсутствующего тега и явно заданного класса."""

    speed = edge.speed_kph
    if type(speed) in (int, float) and isfinite(speed) and speed > 0:
        return ResolvedSpeed(float(speed), edge.speed_status)
    if not model.modeled_speeds_enabled or edge.raw_speed is not None:
        return None
    road_class = edge.raw_attributes.get("highway")
    if type(road_class) is not str:
        return None
    speed = model.modeled_speeds_kph.get(road_class)
    if type(speed) not in (int, float) or not isfinite(speed) or speed <= 0:
        return None
    return ResolvedSpeed(float(speed), f"manual_model:{road_class}")


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
        resolved = resolve_speed(edge, time_model)
        if resolved is None:
            raise ValueError(f"unknown or invalid speed: {edge_id!r}")
        total += 3.6 * edge.length_m / (resolved.speed_kph * factor)
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
        if resolve_speed(edge, time_model) is None:
            unknown.append(edge_id)
    return tuple(unknown)


def compute_easy_references(context: RouteContext, time_model: TimeModel) -> EasyReferences:
    """Найти L_min и T_min отдельными полными поисками при одних дорожных условиях."""

    unknown = unknown_required_speeds(context, time_model)
    if unknown:
        raise ValueError(f"unknown required speeds: {len(unknown)}")
    def by_length(edge_id: EdgeId, _attributes: Mapping[str, object]) -> float:
        return time_model.roads[edge_id].length_m

    shortest = find_connector(context.start, context.end, context, Random(0),
                              weight=by_length, complete=True)
    if shortest is None:
        raise ValueError("no valid route for L_min")
    length_min = calculate_length(shortest, context, time_model)

    def by_time(edge_id: EdgeId, _attributes: Mapping[str, object]) -> float:
        edge = time_model.roads[edge_id]
        level = time_model.levels.get(edge_id, 1)
        resolved = resolve_speed(edge, time_model)
        if resolved is None:
            raise ValueError(f"unknown required speed: {edge_id!r}")
        return 3.6 * edge.length_m / (resolved.speed_kph * speed_factor(level, time_model.factors))

    quickest = find_connector(context.start, context.end, context, Random(0),
                              weight=by_time, complete=True)
    if quickest is None:
        raise ValueError("no valid route for T_min")
    references = EasyReferences(length_min, calculate_travel_time(quickest, context, time_model))
    if not all(_finite_number(value) and value > 0 for value in
               (references.length_min_m, references.time_min_s)):
        raise ValueError("invalid easy references")
    return references
