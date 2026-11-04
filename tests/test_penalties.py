"""Актуальные штрафы: факты OSM, направления и неопределённость."""

import networkx as nx
import pytest
from dataclasses import replace

from algorithm.evaluation.penalties import (
    ABSENT, PRESENT, UNKNOWN, PENALTY_IDS,
    PenaltyEvent, PenaltyEvents, PenaltyPolicy, calculate_complexity,
    calculate_penalties, extract_penalty_events,
)
from config import working_penalty_defaults
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


def test_weighted_sum_breakdown_and_unknown():
    events = PenaltyEvents((
        PenaltyEvent("P02", PRESENT, "osm:node:1", 2, "turn"),
        PenaltyEvent("P12", PRESENT, "osm:way:2", 3.5, "lanes+length"),
        PenaltyEvent("P05", UNKNOWN, "osm:node:3", 0, "missing control"),
    ))
    breakdown = calculate_penalties(events, working_penalty_defaults())
    assert breakdown.lines[0].subtotal == 10
    assert next(line for line in breakdown.lines if line.category == "P12").subtotal == 1.75
    assert calculate_complexity(breakdown) == 11.75
    assert len(breakdown.uncertain) == 1
    assert next(line for line in breakdown.lines if line.category == "P05").uncertain[0].status == UNKNOWN


def test_invalid_penalty_values_and_uncertain_quantity():
    defaults = working_penalty_defaults()
    with pytest.raises(ValueError):
        calculate_penalties(PenaltyEvents(()), defaults[:-1])
    with pytest.raises(ValueError):
        calculate_penalties(PenaltyEvents(()), (16, *defaults[1:]))
    with pytest.raises(ValueError):
        calculate_penalties(PenaltyEvents((
            PenaltyEvent("P05", UNKNOWN, None, 1, "missing"),
        )), defaults)


def test_ten_categories_and_unknown_data():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11, "reversed": False, "highway": "residential"})],
    )
    result = extract_penalty_events(route, context, policy)
    assert {item.category for item in result.observations} == set(PENALTY_IDS)
    assert len(PENALTY_IDS) == 10
    assert observations(result, "P05", UNKNOWN)
    assert observations(result, "P12", UNKNOWN)
    assert observations(result, "P13", UNKNOWN)
    assert not result.confirmed


@pytest.mark.parametrize("end,expected", [
    ((30.0, 59.99), None), ((30.0, 60.01), "P02"), ((29.99, 60.0), "P03"),
])
def test_directional_turns(end, expected):
    route, context, policy = fixture_road(
        {1: (29.99, 60.0), 2: (30.0, 60.0), 3: end, 4: (30.01, 60.01)},
        [(1, 2, 0, {"osmid": 11, "reversed": False}),
         (2, 3, 0, {"osmid": 12, "reversed": False})],
        extra_edges=((4, 2, 0, {"osmid": 13}),),
    )
    result = extract_penalty_events(route, context, policy)
    if expected is None:
        assert not any(item.status == PRESENT and item.category in ("P02", "P03")
                       for item in result.observations)
    else:
        assert len(observations(result, expected)) == 1
        assert observations(result, expected)[0].object_id == "osm:node:2"


def test_same_way_bend_is_not_a_maneuver():
    route, context, policy = fixture_road(
        {1: (29.99, 60.0), 2: (30.0, 60.0), 3: (30.0, 59.99), 4: (30.01, 60.01)},
        [(1, 2, 0, {"osmid": 11}), (2, 3, 0, {"osmid": 11})],
        extra_edges=((4, 2, 0, {"osmid": 13}),),
    )
    result = extract_penalty_events(route, context, policy)
    assert not observations(result, "P02")
    assert observations(result, "P02", ABSENT)


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
    ({"highway": "stop", "stop:direction": "forward"}, "P08"),
    ({"highway": "give_way", "give_way:direction": "forward"}, "P08"),
    ({"railway": "tram_level_crossing"}, "P10"),
    ({"railway": "level_crossing"}, "P11"),
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
        node_tags={2: {"railway": "tram_level_crossing", "highway": "stop"}},
    )
    result = extract_penalty_events(route, context, policy)
    assert not observations(result, "P10")
    assert observations(result, "P10", UNKNOWN)
    assert observations(result, "P08", UNKNOWN)


def test_removed_events_are_not_charged():
    route, context, policy = fixture_road(
        {1: (30.0, 60.0), 2: (30.01, 60.0)},
        [(1, 2, 0, {"osmid": 11, "reversed": False})],
        node_tags={2: {"highway": "traffic_signals", "traffic_calming": "bump"}},
    )
    result = extract_penalty_events(route, context, policy)
    assert not set(PENALTY_IDS) & {"P01", "P04", "P07", "P09", "P15"}
    assert not result.confirmed


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
