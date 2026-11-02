"""Автомобильный доступ, направление и диагностика OSM-исключений."""

import networkx as nx

from data.osm_loader import RoadData
from data.restrictions import apply_access_to_graph, edge_access, is_edge_accessible
from data.road_graph import build_road_data


def road_data(*, reversed_way=False, **tags):
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=30.38, y=60.01)
    graph.add_node(2, x=30.39, y=60.01)
    graph.add_edge(1, 2, key=0, osmid=42, length=100, reversed=reversed_way, **tags)
    return build_road_data(RoadData(graph, (), {}, False))


def test_oneway() -> None:
    data = road_data(oneway=True)
    assert is_edge_accessible((1, 2, 0), data)
    assert not is_edge_accessible((2, 1, 0), data)
    assert edge_access((2, 1, 0), data).reason == "direction_not_in_graph"


def test_denied_car_access() -> None:
    data = road_data(access="no")
    assert not is_edge_accessible((1, 2, 0), data)
    assert edge_access((1, 2, 0), data).status == "denied"
    assert is_edge_accessible((1, 2, 0), road_data(access="no", motorcar="yes"))
    assert not is_edge_accessible((1, 2, 0), road_data(access="yes", motor_vehicle="no"))


def test_directional_access_respects_way_orientation() -> None:
    assert not is_edge_accessible(
        (1, 2, 0), road_data(**{"motorcar:forward": "no", "motorcar:backward": "yes"})
    )
    data = road_data(reversed_way=True, **{"motorcar:forward": "no", "motorcar:backward": "yes"})
    assert is_edge_accessible((1, 2, 0), data)


def test_closed_edge() -> None:
    data = road_data()
    assert not is_edge_accessible((1, 2, 0), data, {(1, 2, 0): 11})
    graph, report = apply_access_to_graph(data, {(1, 2, 0): 11})
    assert report.closed == ((1, 2, 0),)
    assert graph[1][2][0]["car_access"] is False
    assert "car_access" not in data.graph[1][2][0]


def test_unsupported_report() -> None:
    data = road_data(access="yes", **{"motor_vehicle:conditional": "no @ (Mo-Fr 08:00-09:00)"})
    graph, report = apply_access_to_graph(data)
    assert report.allowed == 0 and len(report.unsupported) == 1
    assert graph[1][2][0]["car_access"] is False
    assert not is_edge_accessible((1, 2, 0), road_data(motorcar="destination"))
    assert edge_access((1, 2, 0), road_data(motorcar="destination")).status == "unsupported"
