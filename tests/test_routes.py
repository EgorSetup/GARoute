"""Жёсткие ограничения и полный ключ направленного ребра."""

from dataclasses import replace
from random import Random
import pytest

from algorithm.genetic.validation import validate_route
from algorithm.models import Route, RouteContext
from data.road_graph import check_reachability, find_connector
from data.restrictions import TurnRules, is_turn_allowed


def report(context: RouteContext, *edges: tuple[int, int, int]):
    return validate_route(Route(edges), context)


def test_complete_search_keeps_nondominated_visited_sets() -> None:
    """Дешёвый путь к состоянию может уже посетить нужную позже вершину."""

    import networkx as nx

    graph = nx.MultiDiGraph()
    for u, v, length in (("s", "x", 1), ("x", "a", 1), ("s", "b", 2),
                         ("b", "a", 2), ("a", "y", 1), ("y", "x", 1),
                         ("x", "t", 1)):
        graph.add_edge(u, v, key=0, length=length, car_access=True)
    rules = TurnRules(frozenset({(("s", "x", 0), ("x", "t", 0))}))
    context = RouteContext(graph, "s", "t", rules)
    by_length = lambda edge, attrs: attrs["length"]
    route = find_connector("s", "t", context, Random(1), weight=by_length, complete=True)
    assert route == Route((("s", "b", 0), ("b", "a", 0), ("a", "y", 0),
                           ("y", "x", 0), ("x", "t", 0)))
    assert check_reachability("s", "t", context).reachable


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


def test_topological_path_but_forbidden_turn(route_context: RouteContext) -> None:
    context = replace(route_context, turn_rules=TurnRules(route_context.turn_rules.forbidden | frozenset({
        ((1, 2, 0), (2, 5, 0)), ((1, 2, 1), (2, 5, 0)),
        ((1, 3, 0), (3, 5, 0)), ((1, 4, 0), (4, 6, 0)),
        ((1, 2, 1), (2, 3, 0)), ((1, 3, 0), (3, 2, 0)),
    })))
    assert not check_reachability(1, 6, context).reachable
    assert find_connector(1, 6, context, Random(3)) is None


def test_connector_boundaries(route_context: RouteContext) -> None:
    assert find_connector(2, 3, route_context, Random(1), incoming=(1, 2, 0)) is None
    allowed = find_connector(2, 3, route_context, Random(1), incoming=(1, 2, 1))
    assert allowed == Route(((2, 3, 0),))
    rules = TurnRules(route_context.turn_rules.forbidden | {((2, 3, 0), (3, 5, 0))})
    context = replace(route_context, turn_rules=rules)
    assert find_connector(2, 3, context, Random(1), incoming=(1, 2, 1), outgoing=(3, 5, 0)) is None
    assert find_connector(2, 3, context, Random(1), incoming=(1, 2, 1), outgoing=(3, 2, 0)) == allowed
    assert find_connector(2, 3, context, Random(1), incoming=(1, 2, 99)) is None
    assert find_connector(2, 3, context, Random(1), outgoing=(3, 5, 99)) is None


def test_unreachable(route_context: RouteContext) -> None:
    result = check_reachability(6, 1, route_context)
    assert not result.reachable and result.reason == "unreachable"
    assert find_connector(6, 1, route_context, Random(0)) is None
    assert check_reachability(1, 1, route_context).reason == "same_endpoints"


def test_parallel_route(route_context: RouteContext) -> None:
    context = replace(route_context, closed_edges=frozenset({(1, 2, 0), (1, 3, 0), (1, 4, 0)}))
    connector = find_connector(1, 6, context, Random(0))
    assert connector == Route(((1, 2, 1), (2, 5, 0), (5, 6, 0)))
    assert validate_route(connector, context).valid


def test_connector_weight_and_invalid_weight(route_context: RouteContext) -> None:
    by_length = lambda edge, attrs: attrs["length"]
    connector = find_connector(1, 6, route_context, Random(0), weight=by_length)
    assert connector in (
        Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))),
        Route(((1, 2, 1), (2, 5, 0), (5, 6, 0))),
        Route(((1, 3, 0), (3, 5, 0), (5, 6, 0))),
    )
    with pytest.raises(ValueError, match="positive and finite"):
        find_connector(1, 6, route_context, Random(0), weight=lambda edge, attrs: 0)
