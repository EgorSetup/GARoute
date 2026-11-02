"""Чистая проверка маршрута перед включением в популяцию."""

from algorithm.models import Route, RouteContext, RouteValidationReport
from data.restrictions import is_turn_allowed


def validate_route(route: Route, context: RouteContext) -> RouteValidationReport:
    """Собрать жёсткие нарушения без исправления и изменения входных объектов."""

    errors: list[str] = []
    if context.start == context.end:
        errors.append("same_endpoints")
    if not route.edges:
        errors.append("empty_route")
        return RouteValidationReport(tuple(errors))

    if route.edges[0][0] != context.start:
        errors.append("wrong_start")
    if route.edges[-1][1] != context.end:
        errors.append("wrong_end")

    visited = {route.edges[0][0]}
    for index, edge in enumerate(route.edges):
        u, v, key = edge
        if not context.graph.has_edge(u, v, key):
            errors.append("missing_edge")
        else:
            attributes = context.graph[u][v][key]
            if attributes.get("car_access") is not True:
                errors.append("car_access_denied_or_unknown")
        if edge in context.closed_edges:
            errors.append("closed_edge")
        if index:
            previous = route.edges[index - 1]
            if previous[1] != u:
                errors.append("disconnected_edges")
            elif not is_turn_allowed(previous, edge, context.turn_rules):
                errors.append("forbidden_turn")
        if v in visited:
            errors.append("repeated_vertex")
        visited.add(v)

    return RouteValidationReport(tuple(dict.fromkeys(errors)))
