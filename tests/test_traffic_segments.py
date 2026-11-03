"""Нарезка направленных OSM way без изменения дорожного графа."""

from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

from algorithm.models import RoadEdge
from data.road_graph import NormalizedRoadData
from data.traffic import build_visual_segments


def road_data(*specs) -> NormalizedRoadData:
    graph = nx.MultiDiGraph()
    edges = {}
    for u, v, key, length, way, name in specs:
        graph.add_node(u, x=u * 0.001, y=0.0)
        graph.add_node(v, x=v * 0.001, y=0.0)
        graph.add_edge(u, v, key=key, car_access=True, length=length)
        edge_id = (u, v, key)
        edges[edge_id] = RoadEdge(
            edge_id, ((u * 0.001, 0.0), (v * 0.001, 0.0)), "fixture",
            length, "fixture", (way,), {"name": name} if name else {},
            36, "observed", "maxspeed", None, "36",
        )
    return NormalizedRoadData(graph, edges, {}, (), (), {})


def test_short_street() -> None:
    data = road_data((1, 2, 0, 120, 101, None))
    result = build_visual_segments(data, 500)
    assert result.issues == () and len(result.segments) == 1
    segment = result.segments[0]
    assert segment.length_m == 120 and segment.name == "way 101"
    assert [(item.edge_id, item.start_m, item.end_m) for item in segment.coverage] == [
        ((1, 2, 0), 0.0, 120),
    ]


def test_long_street_split_and_reverse_direction() -> None:
    data = road_data(
        (1, 2, 0, 300, 101, "Главная"), (2, 3, 0, 300, 101, "Главная"),
        (3, 4, 0, 300, 101, "Главная"),
        (2, 1, 0, 300, 101, "Главная"), (3, 2, 0, 300, 101, "Главная"),
        (4, 3, 0, 300, 101, "Главная"),
    )
    result = build_visual_segments(data, 500)
    assert result.issues == ()
    assert sorted(segment.length_m for segment in result.segments) == [400, 400, 500, 500]
    forward = [segment for segment in result.segments
               if any(item.edge_id[0] < item.edge_id[1] for item in segment.coverage)]
    assert len(forward) == 2
    intervals = [item for segment in forward for item in segment.coverage]
    assert [(item.edge_id, item.start_m, item.end_m) for item in intervals] == [
        ((1, 2, 0), 0.0, 300), ((2, 3, 0), 0.0, 200),
        ((2, 3, 0), 200, 300), ((3, 4, 0), 0.0, 300),
    ]


def test_unnamed_branch_policy_and_equal_name_does_not_join() -> None:
    data = road_data(
        (1, 2, 0, 100, 101, None), (2, 3, 0, 100, 101, None),
        (2, 4, 0, 100, 101, None), (3, 5, 0, 100, 102, "Главная"),
        (5, 6, 0, 100, 103, "Главная"),
    )
    result = build_visual_segments(data, 500)
    assert result.issues == ()
    assert len(result.segments) == 5
    assert {segment.way_id for segment in result.segments} == {101, 102, 103}
    assert all(len(segment.coverage) == 1 for segment in result.segments)


def test_graph_topology_unchanged() -> None:
    data = road_data((1, 2, 0, 700, 101, "A"), (2, 3, 0, 100, 101, "A"))
    before_nodes = deepcopy(dict(data.graph.nodes(data=True)))
    before_edges = deepcopy(tuple(data.graph.edges(keys=True, data=True)))
    result = build_visual_segments(data, 500)
    assert result.issues == ()
    assert dict(data.graph.nodes(data=True)) == before_nodes
    assert tuple(data.graph.edges(keys=True, data=True)) == before_edges


@pytest.mark.parametrize("geometry", (
    ((0.0, 0.0), (0.0, 0.0)),
    ((0.0, 0.0), (None, 0.0)),
))
def test_invalid_geometry_is_explicitly_uncovered(geometry) -> None:
    data = road_data((1, 2, 0, 100, 101, None))
    edge_id = (1, 2, 0)
    edges = dict(data.edges)
    edge = edges[edge_id]
    edges[edge_id] = replace(edge, geometry_lonlat=geometry)
    data = replace(data, edges=edges)
    result = build_visual_segments(data, 500)
    assert result.segments == ()
    assert [(issue.edge_id, issue.reason) for issue in result.issues] == [
        (edge_id, "invalid_geometry"),
    ]
