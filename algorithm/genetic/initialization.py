"""Одна попытка построения особи через случайные промежуточные вершины."""

from dataclasses import dataclass
from random import Random
from typing import Callable, Hashable, Mapping

from algorithm.genetic.validation import validate_route
from algorithm.models import EdgeId, GAConfig, Route, RouteContext
from data.road_graph import find_connector


WeightStrategy = Callable[[EdgeId, Mapping[str, object], Random], float]


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """Результат одной конечной попытки без подмены неудачи эталоном."""

    route: Route | None
    reason: str | None
    waypoints: tuple[Hashable, ...]


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
