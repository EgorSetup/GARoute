"""Скрещивание только по общей промежуточной вершине."""

from random import Random

from algorithm.genetic.validation import validate_route
from algorithm.models import Individual, OperatorOutcome, Route, RouteContext


def _internal_vertices(route: Route) -> tuple[object, ...]:
    """Вершины после каждого ребра, кроме конечной B."""

    return tuple(edge[1] for edge in route.edges[:-1])


def crossover(
    parent_a: Individual, parent_b: Individual, context: RouteContext, rng: Random,
) -> OperatorOutcome:
    """Обменять окончания на допустимой общей вершине или вернуть родителя A."""

    if not validate_route(parent_a.route, context).valid or not validate_route(parent_b.route, context).valid:
        raise ValueError("crossover parents must be valid routes")
    original = Route(parent_a.route.edges)
    positions_a = {vertex: index + 1 for index, vertex in enumerate(_internal_vertices(parent_a.route))}
    positions_b = {vertex: index + 1 for index, vertex in enumerate(_internal_vertices(parent_b.route))}
    common = [vertex for vertex in positions_a if vertex in positions_b]
    if not common:
        return OperatorOutcome(original, False, False, reason="no_common_internal_vertex")
    rng.shuffle(common)
    rejected = 0
    for vertex in common:
        candidate = Route(parent_a.route.edges[:positions_a[vertex]] + parent_b.route.edges[positions_b[vertex]:])
        if candidate == original:
            continue
        if validate_route(candidate, context).valid:
            return OperatorOutcome(candidate, True, True, rejected)
        rejected += 1
    return OperatorOutcome(
        original, True, False, rejected,
        "all_splices_invalid" if rejected else "no_change",
    )
