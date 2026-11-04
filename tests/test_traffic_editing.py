"""Правка пробок проходит через общую транзакцию условий и доступ графа."""

import networkx as nx

from algorithm.models import RoadEdge
from data.road_graph import NormalizedRoadData
from data.restrictions import apply_access_to_graph
from data.traffic import CoveredInterval, SegmentCollection, VisualSegment
from interface.controls import (
    UIState, cancel_change, confirm_change, propose_traffic_edit,
    request_condition_change, reset_traffic,
)


EDGE = (1, 2, 0)


def editable_state() -> UIState:
    graph = nx.MultiDiGraph()
    graph.add_edge(*EDGE)
    edge = RoadEdge(
        EDGE, ((30.0, 60.0), (30.001, 60.0)), "fixture", 100,
        "fixture", (101,), {}, 36, "observed", "maxspeed", None, "36",
    )
    roads = NormalizedRoadData(graph, {EDGE: edge}, {}, (), (), {})
    routing, _ = apply_access_to_graph(roads)
    segment = VisualSegment(
        "s0", 101, "Улица", edge.geometry_lonlat, 100,
        (CoveredInterval(EDGE, 0, 100),),
    )
    return UIState(
        normalized_roads=roads, routing_graph=routing,
        traffic_segments=SegmentCollection((segment,), (), ((EDGE, 100),)),
        traffic_levels={EDGE: 1},
    )


def edit(state: UIState, delta: int) -> None:
    proposal = propose_traffic_edit(state, "s0", delta)
    assert not request_condition_change(state, proposal).errors


def test_level_bounds() -> None:
    state = editable_state()
    assert propose_traffic_edit(state, "s0", -1).errors
    assert propose_traffic_edit(state, "missing", 1).errors
    for _ in range(10):
        edit(state, 1)
    assert state.segment_levels == {"s0": 11}
    assert state.traffic_levels[EDGE] == 11
    assert propose_traffic_edit(state, "s0", 1).errors


def test_close_reopen() -> None:
    state = editable_state()
    for _ in range(10):
        edit(state, 1)
    assert state.routing_graph[1][2][0]["car_access"] is False
    edit(state, -1)
    assert state.traffic_levels[EDGE] == 10
    assert state.routing_graph[1][2][0]["car_access"] is True


def test_cancel_traffic_edit() -> None:
    state = editable_state()
    state.run_results.append(object())
    edit(state, 1)
    assert state.pending_change is not None
    assert state.segment_levels == {} and state.traffic_levels[EDGE] == 1
    cancel_change(state)
    assert state.pending_change is None and len(state.run_results) == 1
    assert state.routing_graph[1][2][0]["car_access"] is True


def test_reset_confirmation() -> None:
    state = editable_state()
    edit(state, 1)
    state.run_results.append(object())
    result = request_condition_change(state, reset_traffic(state))
    assert not result.errors and state.pending_change is not None
    assert state.segment_levels == {"s0": 2}
    confirm_change(state)
    assert state.segment_levels == {} and state.traffic_levels[EDGE] == 1
    assert state.run_results == [] and state.pending_change is None


def test_edit_confirm_clears_results() -> None:
    state = editable_state()
    state.run_results.append(object())
    edit(state, 1)
    confirm_change(state)
    assert state.segment_levels == {"s0": 2}
    assert state.run_results == []
