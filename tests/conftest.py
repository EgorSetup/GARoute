"""Искусственный ориентированный граф для ранней проверки ядра."""

import pytest
import networkx as nx

from algorithm.models import RouteContext
from data.restrictions import TurnRules


@pytest.fixture
def route_context() -> RouteContext:
    graph = nx.MultiDiGraph()
    for u, v, key, length in (
        (1, 2, 0, 1),
        (1, 2, 1, 1),
        (2, 5, 0, 1),
        (5, 6, 0, 1),
        (1, 3, 0, 1),
        (3, 5, 0, 1),
        (1, 4, 0, 4),
        (4, 6, 0, 1),
        (2, 3, 0, 1),
        (3, 2, 0, 1),
    ):
        graph.add_edge(u, v, key=key, length=length, car_access=True)
    rules = TurnRules(frozenset({((1, 2, 0), (2, 3, 0))}))
    return RouteContext(graph=graph, start=1, end=6, turn_rules=rules)
