"""Граничные дорожные топологии перед общей приёмкой."""

from dataclasses import replace
from random import Random

import networkx as nx

from algorithm.models import Route, RouteContext
from data.restrictions import TurnRules
from data.road_graph import check_reachability, find_connector


def test_only_route_skips_dead_end_and_respects_direction() -> None:
    graph = nx.MultiDiGraph()
    for u, v in (("s", "a"), ("a", "t"), ("s", "dead")):
        graph.add_edge(u, v, key=0, length=1, car_access=True)
    context = RouteContext(graph, "s", "t", TurnRules())
    expected = Route((("s", "a", 0), ("a", "t", 0)))
    assert check_reachability("s", "t", context).reachable
    assert find_connector("s", "t", context, Random(3), complete=True) == expected
    assert not check_reachability("dead", "t", context).reachable
    assert find_connector("dead", "t", context, Random(3), complete=True) is None
    assert not check_reachability("t", "s", context).reachable

    closed = replace(context, closed_edges=frozenset(graph.edges(keys=True)))
    assert not check_reachability("s", "t", closed).reachable
    assert find_connector("s", "t", closed, Random(3), complete=True) is None
