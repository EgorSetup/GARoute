"""Ручная сверка метров/секунд и отказов модели времени."""

from dataclasses import replace

import pytest

from algorithm.evaluation.fitness import (
    TimeModel, calculate_length, calculate_travel_time, resolve_speed, unknown_required_speeds,
)
from algorithm.models import RoadEdge, Route
from data.traffic import speed_factor


def road(edge_id, length, speed):
    return RoadEdge(
        edge_id, ((0.0, 0.0), (0.001, 0.001)), "test", length, "test",
        (1,), {}, speed, "observed" if speed is not None else "unknown",
        "maxspeed" if speed is not None else None, None, speed,
    )


@pytest.fixture
def metrics_case(route_context):
    ids = ((1, 2, 0), (2, 5, 0), (5, 6, 0))
    route = Route(ids)
    roads = {edge_id: road(edge_id, 1, 36) for edge_id in route_context.graph.edges(keys=True)}
    roads.update({
        ids[0]: road(ids[0], 100, 36),
        ids[1]: road(ids[1], 200, 72),
        ids[2]: road(ids[2], 50, 18),
    })
    model = TimeModel(roads)
    return route_context, route, model


def test_length_sum(metrics_case):
    context, route, model = metrics_case
    assert calculate_length(route, context, model) == 350


def test_time_units(metrics_case):
    context, route, model = metrics_case
    # 100 м / 36 км/ч = 10 с; 200/72 = 10 с; 50/18 = 10 с.
    assert calculate_travel_time(route, context, model) == pytest.approx(30)


def test_traffic_slowdown(metrics_case):
    context, route, model = metrics_case
    model = replace(model, levels={(2, 5, 0): 6})
    assert speed_factor(1) == 1
    assert speed_factor(10) == pytest.approx(0.1)
    assert calculate_travel_time(route, context, model) == pytest.approx(40)
    with pytest.raises(ValueError, match="invalid traffic level"):
        speed_factor(0)
    assert speed_factor(6, {6: 0.25}) == 0.25


def test_closed_before_division(metrics_case):
    context, route, model = metrics_case
    model = replace(model, levels={(2, 5, 0): 11})
    with pytest.raises(ValueError, match="closed edge"):
        calculate_travel_time(route, context, model)
    with pytest.raises(ValueError, match="closed edge"):
        calculate_length(route, context, model)
    with pytest.raises(ValueError, match="closes"):
        speed_factor(11)


def test_missing_speed_policy(metrics_case):
    context, route, model = metrics_case
    roads = dict(model.roads)
    roads[(2, 5, 0)] = road((2, 5, 0), 200, None)
    model = replace(model, roads=roads)
    assert unknown_required_speeds(context, model) == ((2, 5, 0),)
    with pytest.raises(ValueError, match="unknown or invalid speed"):
        calculate_travel_time(route, context, model)
    assert unknown_required_speeds(context, replace(model, levels={(2, 5, 0): 11})) == ()


def test_manual_class_speed_needs_explicit_mode_and_missing_tag(metrics_case):
    context, route, model = metrics_case
    edge_id = (2, 5, 0)
    roads = dict(model.roads)
    roads[edge_id] = replace(road(edge_id, 200, None), raw_attributes={"highway": "living_street"})
    model = replace(model, roads=roads, modeled_speeds_kph={"living_street": 24})
    assert unknown_required_speeds(context, model) == (edge_id,)
    enabled = replace(model, modeled_speeds_enabled=True)
    assert resolve_speed(roads[edge_id], enabled).source == "manual_model:living_street"
    assert calculate_travel_time(route, context, enabled) == pytest.approx(10 + 30 + 10)
    assert unknown_required_speeds(context, enabled) == ()
    ambiguous = replace(roads[edge_id], raw_speed="30;50")
    assert resolve_speed(ambiguous, enabled) is None
    assert resolve_speed(replace(roads[edge_id], raw_attributes={"highway": ["living_street"]}), enabled) is None


def test_invalid_route_and_nonfinite(metrics_case):
    context, route, model = metrics_case
    with pytest.raises(ValueError, match="invalid route"):
        calculate_length(Route(((1, 2, 0), (5, 6, 0))), context, model)
    roads = dict(model.roads)
    roads[(1, 2, 0)] = road((1, 2, 0), float("inf"), 36)
    with pytest.raises(ValueError, match="invalid length"):
        calculate_travel_time(route, context, replace(model, roads=roads))
