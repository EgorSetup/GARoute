"""Нормализация исходного графа и происхождения атрибутов."""

import networkx as nx
import pytest
from shapely.geometry import LineString

from algorithm.models import RoadEdge
from data.osm_loader import RoadData
from data.road_graph import build_road_data


def raw_data(graph: nx.MultiDiGraph) -> RoadData:
    return RoadData(graph, (), {"source": "test"}, False)


def graph_with_parallel_edges() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=30.38, y=60.01)
    graph.add_node(2, x=30.39, y=60.02)
    graph.add_edge(1, 2, key=0, osmid=100, length=120.0,
                   maxspeed="40", highway="residential", reversed=False)
    graph.add_edge(1, 2, key=1, osmid=200, length=140.0,
                   maxspeed="25 mph", highway="primary", reversed=False)
    graph.add_edge(2, 1, key=0, osmid=100, length=120.0,
                   maxspeed="RU:urban", highway="residential", reversed=True)
    return graph


def test_parallel_keys() -> None:
    data = build_road_data(raw_data(graph_with_parallel_edges()))
    assert set(data.edges) == {(1, 2, 0), (1, 2, 1), (2, 1, 0)}
    assert data.edges[1, 2, 0].osm_way_ids == (100,)
    assert data.edges[1, 2, 1].osm_way_ids == (200,)
    assert set(data.way_to_edges[100]) == {(1, 2, 0), (2, 1, 0)}
    assert data.edges[1, 2, 0].length_m == 120
    assert data.edges[1, 2, 1].length_m == 140
    assert data.edges[1, 2, 1].speed_kph == pytest.approx(40.2336)


def test_geometry_direction() -> None:
    graph = graph_with_parallel_edges()
    graph[1][2][0]["geometry"] = LineString([(30.39, 60.02), (30.385, 60.015), (30.38, 60.01)])
    data = build_road_data(raw_data(graph))
    edge = data.edges[1, 2, 0]
    assert isinstance(edge, RoadEdge)
    assert edge.geometry_source == "osm_geometry_reversed"
    assert edge.geometry_lonlat == ((30.38, 60.01), (30.385, 60.015), (30.39, 60.02))
    assert data.edges[2, 1, 0].geometry_lonlat == ((30.39, 60.02), (30.38, 60.01))


def test_missing_length_has_explicit_geodesic_source() -> None:
    graph = graph_with_parallel_edges()
    graph[1][2][0].pop("length")
    edge = build_road_data(raw_data(graph)).edges[1, 2, 0]
    assert 1200 < edge.length_m < 1300
    assert edge.length_source == "geodesic_geometry"


def test_missing_speed() -> None:
    graph = graph_with_parallel_edges()
    graph[1][2][0].pop("maxspeed")
    graph[1][2][1]["maxspeed"] = ["40", "60"]
    data = build_road_data(raw_data(graph))
    assert set(data.unknown_speed_edges) == {(1, 2, 0), (1, 2, 1)}
    assert all(data.edges[edge_id].speed_kph is None for edge_id in data.unknown_speed_edges)
    assert data.edges[2, 1, 0].speed_kph == 60
    assert data.edges[2, 1, 0].speed_status == "implicit_osm_verify"
    assert data.edges[2, 1, 0].speed_rule == "RU:urban_to_60_kph"
    assert data.edges[1, 2, 1].raw_speed == ["40", "60"]
    assert data.edges[1, 2, 0].tag_status("maxspeed") == "unknown"
    assert data.edges[2, 1, 0].tag_status("highway") == "observed"


def test_osm_way_mapping() -> None:
    graph = graph_with_parallel_edges()
    graph[1][2][0]["osmid"] = [100, 101, 100]
    graph[1][2][0]["maxspeed:forward"] = "30 km/h"
    graph[2][1][0]["maxspeed:backward"] = "45"
    data = build_road_data(raw_data(graph))
    assert data.edges[1, 2, 0].osm_way_ids == (100, 101)
    assert (1, 2, 0) in data.way_to_edges[101]
    assert data.edges[1, 2, 0].speed_kph == 30
    assert data.edges[1, 2, 0].speed_source == "maxspeed:forward"
    assert data.edges[2, 1, 0].speed_kph == 45
    assert data.edges[2, 1, 0].speed_source == "maxspeed:backward"
    assert graph[2][1][0]["maxspeed"] == "RU:urban"


@pytest.mark.parametrize("highway,raw_speed,expected,status", [
    ("residential", "30", 30, "observed"),
    ("residential", "RU:urban", 60, "implicit_osm_verify"),
    ("living_street", "RU:living_street", 20, "implicit_osm_verify"),
    ("living_street", None, 20, "assumed_rule"),
    ("unclassified", None, 60, "assumed_rule"),
    ("secondary_link", None, 60, "assumed_rule"),
    ("service", None, None, "unknown"),
    ("residential", "30;50", None, "unknown"),
])
def test_urban_speed_policy(highway, raw_speed, expected, status) -> None:
    graph = graph_with_parallel_edges()
    attrs = graph[1][2][0]
    attrs["highway"] = highway
    attrs.pop("maxspeed")
    if raw_speed is not None:
        attrs["maxspeed"] = raw_speed
    edge = build_road_data(raw_data(graph), urban_context=True).edges[1, 2, 0]
    assert edge.speed_kph == expected
    assert edge.speed_status == status
    if status == "assumed_rule":
        assert edge.speed_source == "PDD_10.2+OSM_highway"
    assert graph[1][2][0].get("maxspeed") == raw_speed
