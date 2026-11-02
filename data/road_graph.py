"""Связующие пути с учётом входящего ребра и запретов перехода."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from random import Random
from typing import Callable, Hashable, Mapping

from algorithm.models import EdgeId, Route, RouteContext
from data.restrictions import is_turn_allowed


EdgeWeight = Callable[[EdgeId, Mapping[str, object]], float]


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    """Факт достижимости без выдачи готовой особи в P0."""

    reachable: bool
    reason: str | None = None


def _traversable(edge: EdgeId, context: RouteContext) -> bool:
    u, v, key = edge
    return (
        context.graph.has_edge(u, v, key)
        and context.graph[u][v][key].get("car_access") is True
        and edge not in context.closed_edges
    )


def _search(
    start: Hashable,
    end: Hashable,
    context: RouteContext,
    rng: Random | None,
    incoming: EdgeId | None,
    outgoing: EdgeId | None,
    weight: EdgeWeight | None,
) -> Route | None:
    if start not in context.graph or end not in context.graph or start == end:
        return None
    if incoming is not None and (incoming[1] != start or not _traversable(incoming, context)):
        return None
    if outgoing is not None and (outgoing[0] != end or not _traversable(outgoing, context)):
        return None

    order = count()
    queue = [(0.0, next(order), start, incoming, (), frozenset({start}))]
    while queue:
        cost, _, node, previous, path, visited = heappop(queue)
        if node == end:
            return Route(path)
        edges = list(context.graph.out_edges(node, keys=True))
        if rng is not None:
            rng.shuffle(edges)
        else:
            edges.sort(key=lambda edge: tuple(map(str, edge)))
        for edge in edges:
            _, neighbor, key = edge
            edge_id = (node, neighbor, key)
            if neighbor in visited:
                continue
            if not _traversable(edge_id, context):
                continue
            if previous is not None and not is_turn_allowed(previous, edge_id, context.turn_rules):
                continue
            attributes = context.graph[node][neighbor][key]
            edge_cost = 1.0 if weight is None else float(weight(edge_id, attributes))
            if not isfinite(edge_cost) or edge_cost <= 0:
                raise ValueError("edge weight must be positive and finite")
            if neighbor == end:
                if outgoing is None or is_turn_allowed(edge_id, outgoing, context.turn_rules):
                    heappush(queue, (cost + edge_cost, next(order), neighbor, edge_id, path + (edge_id,), visited | {neighbor}))
                continue
            heappush(queue, (cost + edge_cost, next(order), neighbor, edge_id, path + (edge_id,), visited | {neighbor}))
    return None


def find_connector(
    start: Hashable,
    end: Hashable,
    context: RouteContext,
    rng: Random,
    incoming: EdgeId | None = None,
    outgoing: EdgeId | None = None,
    weight: EdgeWeight | None = None,
) -> Route | None:
    """Найти один допустимый простой подмаршрут или явно вернуть None."""

    return _search(start, end, context, rng, incoming, outgoing, weight)


def check_reachability(a: Hashable, b: Hashable, context: RouteContext) -> ReachabilityResult:
    """Проверить доступность без записи найденного пути в популяцию."""

    if a == b:
        return ReachabilityResult(False, "same_endpoints")
    route = _search(a, b, context, None, None, None, None)
    return ReachabilityResult(route is not None, None if route is not None else "unreachable")
