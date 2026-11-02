"""Автомобильный доступ, направление и диагностика OSM-исключений."""

import networkx as nx

from data.osm_loader import RoadData
from data.restrictions import (
    apply_access_to_graph, build_forbidden_turns, edge_access,
    is_edge_accessible, is_turn_allowed,
)
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


def turn_data():
    graph = nx.MultiDiGraph(crs="epsg:4326")
    for node, lon in ((1, 30.38), (2, 30.39), (3, 30.40), (4, 30.41), (5, 30.42)):
        graph.add_node(node, x=lon, y=60.01)
    for u, v, key, way in (
        (1, 2, 0, 10), (2, 3, 0, 20), (2, 4, 0, 30),
        (5, 2, 0, 40), (1, 2, 1, 50),
    ):
        graph.add_edge(u, v, key=key, osmid=way, length=100, reversed=False)
    return build_road_data(RoadData(graph, (), {}, False))


def relation(restriction="no_left_turn", via_type="node", extra_tags=None):
    return {"type": "relation", "id": 7,
            "members": [{"type": "way", "ref": 10, "role": "from"},
                        {"type": via_type, "ref": 2, "role": "via"},
                        {"type": "way", "ref": 20, "role": "to"}],
            "tags": {"type": "restriction", "restriction": restriction, **(extra_tags or {})}}


def test_no_turn() -> None:
    rules = build_forbidden_turns((relation(),), turn_data())
    assert rules.applied_relation_ids == (7,)
    assert not rules.issues
    assert not is_turn_allowed((1, 2, 0), (2, 3, 0), rules)
    assert is_turn_allowed((1, 2, 0), (2, 4, 0), rules)


def test_only_turn() -> None:
    rules = build_forbidden_turns((relation("only_straight_on"),), turn_data())
    assert is_turn_allowed((1, 2, 0), (2, 3, 0), rules)
    assert not is_turn_allowed((1, 2, 0), (2, 4, 0), rules)
    assert len(rules.forbidden) == 1


def test_reverse_direction() -> None:
    rules = build_forbidden_turns((relation(),), turn_data())
    assert is_turn_allowed((5, 2, 0), (2, 3, 0), rules)


def test_parallel_edge_rule() -> None:
    data = turn_data()
    rules = build_forbidden_turns((relation(),), data)
    assert is_turn_allowed((1, 2, 1), (2, 3, 0), rules)
    data.graph[1][2][1]["osmid"] = 10
    ambiguous = build_forbidden_turns((relation(),), build_road_data(RoadData(data.graph, (), {}, False)))
    assert not ambiguous.forbidden
    assert ambiguous.issues[0].reason == "ambiguous_way_mapping"


def test_via_way_report() -> None:
    rules = build_forbidden_turns((relation(via_type="way"),), turn_data())
    assert not rules.forbidden and not rules.applied_relation_ids
    assert rules.issues[0].reason == "via_way_unsupported"
    conditional = build_forbidden_turns((relation(extra_tags={"restriction:conditional": "no_left_turn @ (Mo-Fr)"}),), turn_data())
    assert conditional.issues[0].reason == "conditional_unsupported"
    special = build_forbidden_turns((relation("no_entry"),), turn_data())
    assert special.issues[0].reason == "restriction_type_unsupported"
