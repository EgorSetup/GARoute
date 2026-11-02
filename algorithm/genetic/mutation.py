"""Мутация маршрута без ремонта и изменения родителя."""

from dataclasses import dataclass
from random import Random
from typing import Callable, Mapping

from algorithm.genetic.validation import validate_route
from algorithm.models import EdgeId, Individual, OperatorOutcome, Route, RouteContext
from data.road_graph import find_connector


@dataclass(frozen=True, slots=True)
class MutationPolicy:
    """Весовая стратегия передаётся извне; производственного default пока нет."""

    weight_strategy: Callable[[EdgeId, Mapping[str, object], Random], float]


def mutate_subroute(
    individual: Individual, context: RouteContext, rng: Random, policy: MutationPolicy,
) -> OperatorOutcome:
    """Один раз выбрать две вершины и попробовать заменить участок пути."""

    parent = individual.route
    if not validate_route(parent, context).valid:
        raise ValueError("mutation parent must be a valid route")
    if not callable(policy.weight_strategy):
        raise ValueError("mutation weight strategy is required")

    start_index, end_index = sorted(rng.sample(range(len(parent.edges) + 1), 2))
    vertices = (context.start,) + tuple(edge[1] for edge in parent.edges)
    incoming = parent.edges[start_index - 1] if start_index else None
    outgoing = parent.edges[end_index] if end_index < len(parent.edges) else None

    def weight(edge: EdgeId, attributes: Mapping[str, object]) -> float:
        return policy.weight_strategy(edge, attributes, rng)

    connector = find_connector(
        vertices[start_index], vertices[end_index], context, rng,
        incoming=incoming, outgoing=outgoing, weight=weight,
    )
    if connector is None:
        return OperatorOutcome(Route(parent.edges), True, False, reason="connector_unavailable")
    candidate = Route(parent.edges[:start_index] + connector.edges + parent.edges[end_index:])
    if candidate == parent:
        return OperatorOutcome(Route(parent.edges), True, False, reason="no_change")
    if not validate_route(candidate, context).valid:
        return OperatorOutcome(Route(parent.edges), True, False, 1, "invalid_route")
    return OperatorOutcome(candidate, True, True)


def mutate_via_waypoint(
    individual: Individual, context: RouteContext, rng: Random, policy: MutationPolicy,
) -> OperatorOutcome:
    """Один раз заменить участок маршрутом через случайную вершину."""

    parent = individual.route
    if not validate_route(parent, context).valid:
        raise ValueError("mutation parent must be a valid route")
    if not callable(policy.weight_strategy):
        raise ValueError("mutation weight strategy is required")
    start_index, end_index = sorted(rng.sample(range(len(parent.edges) + 1), 2))
    vertices = (context.start,) + tuple(edge[1] for edge in parent.edges)
    start, end = vertices[start_index], vertices[end_index]
    choices = [node for node in context.graph if node not in (start, end)]
    if not choices:
        return OperatorOutcome(Route(parent.edges), True, False, reason="no_waypoint")
    waypoint = rng.choice(choices)
    incoming = parent.edges[start_index - 1] if start_index else None
    outgoing = parent.edges[end_index] if end_index < len(parent.edges) else None

    def weight(edge: EdgeId, attributes: Mapping[str, object]) -> float:
        return policy.weight_strategy(edge, attributes, rng)

    first = find_connector(start, waypoint, context, rng, incoming=incoming, weight=weight)
    if first is None:
        return OperatorOutcome(Route(parent.edges), True, False, reason="waypoint_unreachable")
    second = find_connector(
        waypoint, end, context, rng, incoming=first.edges[-1], outgoing=outgoing, weight=weight,
    )
    if second is None:
        return OperatorOutcome(Route(parent.edges), True, False, reason="waypoint_unreachable")
    candidate = Route(parent.edges[:start_index] + first.edges + second.edges + parent.edges[end_index:])
    if candidate == parent:
        return OperatorOutcome(Route(parent.edges), True, False, reason="no_change")
    if not validate_route(candidate, context).valid:
        return OperatorOutcome(Route(parent.edges), True, False, 1, "invalid_route")
    return OperatorOutcome(candidate, True, True)


def mutate(
    method: str, individual: Individual, context: RouteContext, rng: Random,
    policy: MutationPolicy,
) -> OperatorOutcome:
    """Вызвать настроенный метод с единым контрактом результата."""

    if method == "A":
        return mutate_subroute(individual, context, rng, policy)
    if method == "B":
        return mutate_via_waypoint(individual, context, rng, policy)
    raise ValueError("unknown mutation method")
