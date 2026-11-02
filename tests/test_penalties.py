"""P01–P15: факты OSM, направления, единицы и неизвестность отдельно от нуля."""

import networkx as nx
import pytest
from dataclasses import replace

from algorithm.evaluation.penalties import (
    ABSENT, PRESENT, UNKNOWN, UNSUPPORTED, PENALTY_IDS,
    PenaltyPolicy, extract_penalty_events,
)
from algorithm.models import RoadEdge, Route, RouteContext
from data.restrictions import TurnRules


def fixture_road(nodes, edges, *, extra_edges=(), node_tags=None):
    """Упорядоченный маршрут на графе с явной геометрией/OSM-происхождением."""

    graph = nx.MultiDiGraph()
    for node, point in nodes.items():
        graph.add_node(node, x=point[0], y=point[1], **(node_tags or {}).get(node, {}))
    roads = {}
    for u, v, key, attrs in (*edges, *extra_edges):
        graph.add_edge(u, v, key=key, car_access=True, **attrs)
        if (u, v, key, attrs) in extra_edges:
            continue
        edge_id = (u, v, key)
        shape = (nodes[u], nodes[v])
        roads[edge_id] = RoadEdge(
            edge_id, shape, "test", float(attrs.get("length", 100)), "test",
            (attrs["osmid"],), attrs, 60, "observed", "maxspeed", None, 60,
        )
    route = Route(tuple((u, v, key) for u, v, key, _ in edges))
    context = RouteContext(graph, edges[0][0], edges[-1][1], TurnRules())
    return route, context, PenaltyPolicy(roads)


def observations(result, category, status=PRESENT):
    return [item for item in result.observations if item.category == category and item.status == status]


def test_fifteen_categories_and_unknown_lane_change():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11, "reversed": False, "highway": "residential"})],
    )
    result = extract_penalty_events(route, context, policy)
    assert {item.category for item in result.observations} == set(PENALTY_IDS)
    assert observations(result, "P04", UNSUPPORTED)
    assert observations(result, "P05", UNKNOWN)
    assert observations(result, "P12", UNKNOWN)
    assert observations(result, "P13", UNKNOWN)
    assert not result.confirmed


@pytest.mark.parametrize("end,expected", [
    ((30.0, 59.99), "P01"), ((30.0, 60.01), "P02"), ((29.99, 60.0), "P03"),
])
def test_directional_turns(end, expected):
    route, context, policy = fixture_road(
        {1: (29.99, 60.0), 2: (30.0, 60.0), 3: end, 4: (30.01, 60.01)},
        [(1, 2, 0, {"osmid": 11, "reversed": False}),
         (2, 3, 0, {"osmid": 12, "reversed": False})],
        extra_edges=((4, 2, 0, {"osmid": 13}),),
    )
    result = extract_penalty_events(route, context, policy)
    assert len(observations(result, expected)) == 1
    assert observations(result, expected)[0].object_id == "osm:node:2"


def test_same_way_bend_is_not_a_maneuver():
    route, context, policy = fixture_road(
        {1: (29.99, 60.0), 2: (30.0, 60.0), 3: (30.0, 59.99), 4: (30.01, 60.01)},
        [(1, 2, 0, {"osmid": 11}), (2, 3, 0, {"osmid": 11})],
        extra_edges=((4, 2, 0, {"osmid": 13}),),
    )
    result = extract_penalty_events(route, context, policy)
    assert not observations(result, "P01")
    assert observations(result, "P01", ABSENT)


def test_no_double_count_and_distance_units():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0), 3: (30.02, 60.0)},
        [(1, 2, 0, {"osmid": 11, "junction": "roundabout", "highway": "residential",
                    "lanes:forward": "3", "width": "3 m", "hazard": "school_zone",
                    "reversed": False, "length": 100}),
         (2, 3, 0, {"osmid": 11, "junction": "roundabout", "highway": "residential",
                    "lanes:forward": "3", "width": "3 m", "hazard": "school_zone",
                    "reversed": False, "length": 200})],
    )
    result = extract_penalty_events(route, context, policy)
    assert len(observations(result, "P06")) == 1
    assert len(observations(result, "P14")) == 1
    assert sum(item.quantity for item in observations(result, "P12")) == 3
    assert sum(item.quantity for item in observations(result, "P13")) == 3


@pytest.mark.parametrize("tag,category", [
    ({"highway": "traffic_signals", "direction": "forward"}, "P07"),
    ({"highway": "stop", "stop:direction": "forward"}, "P08"),
    ({"highway": "give_way", "give_way:direction": "forward"}, "P09"),
    ({"railway": "tram_level_crossing"}, "P10"),
    ({"railway": "level_crossing"}, "P11"),
    ({"traffic_calming": "bump"}, "P15"),
])
def test_confirmed_point_features(tag, category):
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11, "reversed": False})], node_tags={2: tag},
    )
    result = extract_penalty_events(route, context, policy)
    assert len(observations(result, category)) == 1
    assert observations(result, category)[0].object_id == "osm:node:2"


def test_bridge_not_tram_crossing_and_direction_unknown():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11, "reversed": False, "bridge": "yes"})],
        node_tags={2: {"railway": "tram_level_crossing", "highway": "traffic_signals"}},
    )
    result = extract_penalty_events(route, context, policy)
    assert not observations(result, "P10")
    assert observations(result, "P10", UNKNOWN)
    assert observations(result, "P07", UNKNOWN)


def test_pedestrian_tram_crossing_not_car_event():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11})], node_tags={2: {"railway": "tram_crossing"}},
    )
    assert not observations(extract_penalty_events(route, context, policy), "P10")


def test_sign_direction_opposite_is_not_charged():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11, "reversed": False})],
        node_tags={2: {"highway": "stop", "stop:direction": "backward"}},
    )
    result = extract_penalty_events(route, context, policy)
    assert not observations(result, "P08")
    assert observations(result, "P08", ABSENT)


def test_nonfinite_policy_rejected():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11})],
    )
    with pytest.raises(ValueError, match="invalid penalty policy"):
        extract_penalty_events(route, context, replace(policy, distance_unit_m=float("inf")))
