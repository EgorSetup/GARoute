"""Сквозная связь уровней сегментов, допустимости перехода и времени маршрута."""

from random import Random

import networkx as nx
import pytest

from algorithm.evaluation.fitness import TimeModel, calculate_travel_time
from algorithm.models import RoadEdge, Route, RouteContext
from config import TRAFFIC_SPEED_FACTORS
from data.restrictions import TurnRules, apply_access_to_graph
from data.road_graph import NormalizedRoadData, check_reachability, find_connector
from data.traffic import CoveredInterval, SegmentCollection, VisualSegment, assign_edge_levels


SHORT = ((1, 2, 0), (2, 4, 0))
LONG = ((1, 3, 0), (3, 4, 0))


def roads() -> NormalizedRoadData:
    graph = nx.MultiDiGraph()
    edges = {}
    coords = {1: (30.0, 60.0), 2: (30.001, 60.0),
              3: (30.0, 60.001), 4: (30.001, 60.001)}
    for edge_id in (*SHORT, *LONG):
        length = 100 if edge_id in SHORT else 120
        u, v, key = edge_id
        graph.add_edge(u, v, key=key, length=length)
        edges[edge_id] = RoadEdge(
            edge_id, (coords[u], coords[v]), "fixture", length, "fixture",
            (100 + u,), {}, 36, "observed", "maxspeed", None, "36",
        )
    return NormalizedRoadData(graph, edges, {}, (), (), {})


def coverage(data: NormalizedRoadData, *, split_short: bool = False) -> SegmentCollection:
    segments = []
    targets = []
    for edge_id, edge in data.edges.items():
        targets.append((edge_id, edge.length_m))
        parts = (30, 70) if split_short and edge_id == SHORT[0] else (edge.length_m,)
        position = 0.0
        for index, length in enumerate(parts):
            segments.append(VisualSegment(
                f"{edge_id}:{index}", edge.osm_way_ids[0], "fixture",
                edge.geometry_lonlat, length,
                (CoveredInterval(edge_id, position, position + length),),
            ))
            position += length
    return SegmentCollection(tuple(segments), (), tuple(targets))


def context(data: NormalizedRoadData, levels: dict) -> RouteContext:
    graph, _ = apply_access_to_graph(data, levels)
    return RouteContext(graph, 1, 4, TurnRules())


def fastest(data: NormalizedRoadData, levels: dict) -> Route | None:
    current = context(data, levels)
    model = TimeModel(data.edges, levels)

    def weight(edge_id, attrs):
        edge = data.edges[edge_id]
        level = levels.get(edge_id, 1)
        return 3.6 * edge.length_m / (edge.speed_kph * TRAFFIC_SPEED_FACTORS[level])

    return find_connector(1, 4, current, Random(1), weight=weight, complete=True)


def test_slowdown_metrics() -> None:
    data = roads()
    segments = coverage(data)
    initial = assign_edge_levels(segments, {})
    route = fastest(data, initial)
    assert route == Route(SHORT)
    assert calculate_travel_time(route, context(data, initial), TimeModel(data.edges, initial)) == 20

    slowed = assign_edge_levels(segments, {f"{SHORT[0]}:0": 8})
    assert calculate_travel_time(route, context(data, slowed), TimeModel(data.edges, slowed)) == pytest.approx(
        10 / TRAFFIC_SPEED_FACTORS[8] + 10,
    )
    assert fastest(data, slowed) == Route(LONG)


def test_closed_path_excluded() -> None:
    data = roads()
    segments = coverage(data)
    levels = assign_edge_levels(segments, {f"{SHORT[0]}:0": 11})
    current = context(data, levels)
    assert current.graph[1][2][0]["car_access"] is False
    assert fastest(data, levels) == Route(LONG)
    with pytest.raises(ValueError, match="invalid route|closed edge"):
        calculate_travel_time(Route(SHORT), current, TimeModel(data.edges, levels))


def test_all_paths_closed() -> None:
    data = roads()
    segments = coverage(data)
    levels = assign_edge_levels(segments, {
        f"{SHORT[0]}:0": 11, f"{LONG[0]}:0": 11,
    })
    current = context(data, levels)
    assert check_reachability(1, 4, current).reason == "unreachable"
    assert fastest(data, levels) is None


def test_short_closed_part_not_dominant() -> None:
    data = roads()
    segments = coverage(data, split_short=True)
    levels = assign_edge_levels(segments, {f"{SHORT[0]}:0": 11})
    assert levels[SHORT[0]] == 1
    assert context(data, levels).graph[1][2][0]["car_access"] is True
    assert fastest(data, levels) == Route(SHORT)
