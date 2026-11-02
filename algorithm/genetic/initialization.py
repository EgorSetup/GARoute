"""Одна попытка построения особи через случайные промежуточные вершины."""

from dataclasses import dataclass
from math import fsum, isfinite
from random import Random
from typing import Callable, Hashable, Mapping

from algorithm.genetic.validation import validate_route
from algorithm.models import EdgeId, GAConfig, Route, RouteContext, ValidationReport
from data.road_graph import find_connector


WeightStrategy = Callable[[EdgeId, Mapping[str, object], Random], float]


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """Результат одной конечной попытки без подмены неудачи эталоном."""

    route: Route | None
    reason: str | None
    waypoints: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class InitialLengthPolicy:
    """Метры эталона и утверждённый множитель предела P0."""

    minimum_length_m: float | None
    factor: float = 3.0


def minimum_valid_length(context: RouteContext) -> float | None:
    """Найти только длину кратчайшего допустимого пути, не особь P0."""

    def by_length(edge: EdgeId, attributes: Mapping[str, object]) -> float:
        try:
            return float(attributes["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("edge length must be known in metres") from error

    route = find_connector(context.start, context.end, context, Random(0), weight=by_length)
    if route is None:
        return None
    return fsum(float(context.graph[u][v][key]["length"]) for u, v, key in route.edges)


def check_initial_length(route_length_m: float, policy: InitialLengthPolicy) -> ValidationReport:
    """Проверить L≤k_init·L_min в метрах, не меняя маршрут."""

    base = policy.minimum_length_m
    factor = policy.factor
    if type(base) not in (int, float) or not isfinite(base) or base <= 0:
        return ValidationReport(errors=("L_min must be positive and finite",))
    if type(factor) not in (int, float) or not isfinite(factor) or factor <= 0:
        return ValidationReport(errors=("initial length factor must be positive and finite",))
    if type(route_length_m) not in (int, float) or not isfinite(route_length_m) or route_length_m <= 0:
        return ValidationReport(errors=("route length must be positive and finite",))
    limit = factor * base
    if not isfinite(limit):
        return ValidationReport(errors=("initial length limit is not finite",))
    if route_length_m > limit:
        return ValidationReport(errors=("initial route exceeds k_init × L_min",))
    return ValidationReport()


def generate_candidate(
    context: RouteContext,
    config: GAConfig,
    rng: Random,
    weight_strategy: WeightStrategy,
) -> CandidateOutcome:
    """Соединить 1–4 случайные вершины и проверить готовую особь."""

    if config.min_waypoints < 1 or config.max_waypoints > 4 or config.min_waypoints > config.max_waypoints:
        return CandidateOutcome(None, "invalid_waypoint_range", ())
    if context.start == context.end:
        return CandidateOutcome(None, "same_endpoints", ())
    available = [node for node in context.graph if node not in (context.start, context.end)]
    if len(available) < config.min_waypoints:
        return CandidateOutcome(None, "not_enough_waypoints", ())

    count = rng.randint(config.min_waypoints, min(config.max_waypoints, len(available)))
    waypoints = tuple(rng.sample(available, count))
    targets = (context.start, *waypoints, context.end)
    edges: list[EdgeId] = []

    def weight(edge: EdgeId, attributes: Mapping[str, object]) -> float:
        return weight_strategy(edge, attributes, rng)

    for start, end in zip(targets, targets[1:]):
        connector = find_connector(
            start,
            end,
            context,
            rng,
            incoming=edges[-1] if edges else None,
            weight=weight,
        )
        if connector is None:
            return CandidateOutcome(None, "connector_unavailable", waypoints)
        edges.extend(connector.edges)

    candidate = Route(tuple(edges))
    report = validate_route(candidate, context)
    if not report.valid:
        return CandidateOutcome(None, "invalid_route:" + ",".join(report.codes), waypoints)
    return CandidateOutcome(candidate, None, waypoints)
