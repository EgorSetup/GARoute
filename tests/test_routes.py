"""Жёсткие ограничения и полный ключ направленного ребра."""

from dataclasses import replace

from algorithm.genetic.validation import validate_route
from algorithm.models import Route, RouteContext
from data.restrictions import TurnRules, is_turn_allowed


def report(context: RouteContext, *edges: tuple[int, int, int]):
    return validate_route(Route(edges), context)


def test_valid_route(route_context: RouteContext) -> None:
    assert report(route_context, (1, 2, 0), (2, 5, 0), (5, 6, 0)).valid
    assert report(route_context, (1, 2, 1), (2, 5, 0), (5, 6, 0)).valid
    assert not is_turn_allowed((1, 2, 0), (2, 3, 0), route_context.turn_rules)
    assert is_turn_allowed((1, 2, 1), (2, 3, 0), route_context.turn_rules)


def test_wrong_endpoints(route_context: RouteContext) -> None:
    assert "wrong_start" in report(route_context, (2, 5, 0), (5, 6, 0)).codes
    assert "wrong_end" in report(route_context, (1, 2, 0)).codes
    assert "same_endpoints" in report(replace(route_context, end=1), (1, 2, 0)).codes
    assert "empty_route" in report(route_context).codes


def test_disconnected_edges(route_context: RouteContext) -> None:
    assert "disconnected_edges" in report(route_context, (1, 2, 0), (3, 5, 0), (5, 6, 0)).codes


def test_missing_key(route_context: RouteContext) -> None:
    assert "missing_edge" in report(route_context, (1, 2, 99), (2, 5, 0), (5, 6, 0)).codes
    assert "missing_edge" in report(route_context, (2, 1, 0), (1, 3, 0), (3, 5, 0), (5, 6, 0)).codes


def test_cycle(route_context: RouteContext) -> None:
    assert "repeated_vertex" in report(route_context, (1, 2, 1), (2, 3, 0), (3, 2, 0), (2, 5, 0), (5, 6, 0)).codes


def test_forbidden_join(route_context: RouteContext) -> None:
    assert "forbidden_turn" in report(route_context, (1, 2, 0), (2, 3, 0), (3, 5, 0), (5, 6, 0)).codes
    assert report(route_context, (1, 2, 1), (2, 3, 0), (3, 5, 0), (5, 6, 0)).valid


def test_access_and_closure(route_context: RouteContext) -> None:
    context = replace(route_context, closed_edges=frozenset({(2, 5, 0)}))
    assert "closed_edge" in report(context, (1, 2, 0), (2, 5, 0), (5, 6, 0)).codes
    route_context.graph[1][3][0]["car_access"] = False
    assert "car_access_denied_or_unknown" in report(route_context, (1, 3, 0), (3, 5, 0), (5, 6, 0)).codes
    route_context.graph[1][3][0].pop("car_access")
    assert "car_access_denied_or_unknown" in report(route_context, (1, 3, 0), (3, 5, 0), (5, 6, 0)).codes


def test_no_mutation(route_context: RouteContext) -> None:
    candidate = Route(((1, 2, 0), (2, 3, 0), (3, 5, 0), (5, 6, 0)))
    before = candidate.edges
    rules = route_context.turn_rules
    validate_route(candidate, route_context)
    assert candidate.edges == before
    assert route_context.turn_rules == rules == TurnRules(frozenset({((1, 2, 0), (2, 3, 0))}))
