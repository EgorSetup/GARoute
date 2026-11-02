"""Привязка A/B и подтверждение изменения уже выбранной пары."""

import networkx as nx

from data.restrictions import TurnRules
from interface.controls import (
    UIState, cancel_change, confirm_change, request_condition_change, select_endpoint,
)
from interface.map_view import snap_to_node


def state_with_roads() -> UIState:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node, lon in ((1, 30.38), (2, 30.39), (3, 30.40), (4, 30.41)):
        graph.add_node(node, x=lon, y=60.01)
    graph.add_edge(1, 2, key=0, car_access=True)
    graph.add_edge(2, 3, key=0, car_access=True)
    graph.add_edge(2, 4, key=0, car_access=True)
    graph.add_edge(3, 1, key=0, car_access=False)
    return UIState(graph_loaded=True, routing_graph=graph, turn_rules=TurnRules())


def test_endpoint_snap() -> None:
    state = state_with_roads()
    assert snap_to_node((60.01, 30.3801), state.routing_graph, "A") == 1
    assert snap_to_node((60.01, 30.4001), state.routing_graph, "B") == 3
    assert snap_to_node((60.01, 30.4001), state.routing_graph, "A") == 2
    assert snap_to_node((60.01, 30.38), state.routing_graph, "unknown") is None


def test_endpoint_cancel() -> None:
    state = state_with_roads()
    request_condition_change(state, select_endpoint(state, "A", 1))
    request_condition_change(state, select_endpoint(state, "B", 3))
    state.run_results.append(object())
    change = select_endpoint(state, "B", 4)
    request_condition_change(state, change)
    assert state.pending_change is change and state.endpoint_b == 3
    cancel_change(state)
    assert state.endpoint_b == 3 and len(state.run_results) == 1
    request_condition_change(state, select_endpoint(state, "B", 4))
    confirm_change(state)
    assert state.endpoint_b == 4 and not state.run_results


def test_same_or_unreachable_pair_is_rejected() -> None:
    state = state_with_roads()
    request_condition_change(state, select_endpoint(state, "A", 3))
    assert "разными" in select_endpoint(state, "B", 3).errors[0]
    assert "нет направленного проезда" in select_endpoint(state, "B", 4).errors[0]
    assert state.endpoint_b is None
