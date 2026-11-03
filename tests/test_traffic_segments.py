"""Нарезка направленных OSM way без изменения дорожного графа."""

from copy import deepcopy
from dataclasses import replace

import networkx as nx
import pytest

from algorithm.models import RoadEdge
from data.road_graph import NormalizedRoadData
from data.traffic import (
    CoveredInterval, SegmentCollection, SegmentationIssue, VisualSegment,
    assign_edge_levels, build_visual_segments,
)


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


def coverage_example(lengths: tuple[float, ...]) -> SegmentCollection:
    edge_id = (1, 2, 0)
    segments = []
    offset = 0.0
    for index, length in enumerate(lengths):
        segments.append(VisualSegment(
            f"s{index}", 101, "Пример", ((0.0, 0.0), (0.001, 0.0)), length,
            (CoveredInterval(edge_id, offset, offset + length),),
        ))
        offset += length
    return SegmentCollection(tuple(segments), (), ((edge_id, offset),))


def test_dominant_coverage_and_short_closed_part_not_dominant() -> None:
    coverage = coverage_example((150, 320, 30))
    levels = assign_edge_levels(coverage, {"s0": 3, "s1": 8, "s2": 11})
    assert levels == {(1, 2, 0): 8}


def test_tie_higher_level_without_tolerance() -> None:
    coverage = coverage_example((100, 100))
    assert assign_edge_levels(coverage, {"s0": 3, "s1": 8}) == {(1, 2, 0): 8}
    near_tie = coverage_example((100, 100.000000001))
    assert assign_edge_levels(near_tie, {"s0": 11, "s1": 1}) == {(1, 2, 0): 1}


def test_crossing_roads_independent() -> None:
    first, second = (1, 2, 0), (3, 4, 0)
    coverage = SegmentCollection((
        VisualSegment("a", 101, "A", ((0.0, 0.0), (1.0, 1.0)), 100,
                      (CoveredInterval(first, 0, 100),)),
        VisualSegment("b", 102, "B", ((0.0, 1.0), (1.0, 0.0)), 100,
                      (CoveredInterval(second, 0, 100),)),
    ), (), ((first, 100), (second, 100)))
    assert assign_edge_levels(coverage, {"a": 11, "b": 2}) == {first: 11, second: 2}


def test_missing_coverage_and_invalid_level_are_errors() -> None:
    edge_id = (1, 2, 0)
    missing = SegmentCollection((), (), ((edge_id, 100),))
    with pytest.raises(ValueError, match="missing or unexpected"):
        assign_edge_levels(missing, {})
    invalid = coverage_example((100,))
    with pytest.raises(ValueError, match="traffic levels"):
        assign_edge_levels(invalid, {"s0": 12})
    marked = replace(invalid, issues=(SegmentationIssue(edge_id, "missing_coverage"),))
    with pytest.raises(ValueError, match="uncovered"):
        assign_edge_levels(marked, {})
    with pytest.raises(ValueError, match="target edge lengths required"):
        assign_edge_levels(replace(invalid, target_lengths=()), {})
