"""Одна попытка P0 с явными искусственными весами и RNG."""

from dataclasses import replace
from random import Random

import pytest

from algorithm.genetic.initialization import generate_candidate
from algorithm.genetic.validation import validate_route
from algorithm.models import GAConfig, Route, RouteContext


def by_length(edge, attributes, rng):
    return attributes["length"]


class FixedWaypointRng(Random):
    def __init__(self, waypoints):
        super().__init__(0)
        self.waypoints = waypoints

    def randint(self, a, b):
        return len(self.waypoints)

    def sample(self, population, k):
        assert all(node in population for node in self.waypoints)
        assert len(self.waypoints) == k
        return list(self.waypoints)


def test_waypoint_count(route_context: RouteContext) -> None:
    config = replace(GAConfig(), min_waypoints=2, max_waypoints=2)
    outcome = generate_candidate(route_context, config, Random(7), by_length)
    assert len(outcome.waypoints) == 2
    assert set(outcome.waypoints).isdisjoint({1, 6})
    assert outcome.route is not None or outcome.reason is not None


def test_candidate_validation(route_context: RouteContext) -> None:
    outcome = generate_candidate(route_context, GAConfig(), FixedWaypointRng((2,)), by_length)
    assert outcome.reason is None
    assert outcome.route is not None
    assert validate_route(outcome.route, route_context).valid
    blocked = replace(route_context, closed_edges=frozenset({(1, 2, 1)}))
    failure = generate_candidate(blocked, GAConfig(), FixedWaypointRng((2, 3)), by_length)
    assert failure.route is None and failure.reason == "connector_unavailable"


def test_seed_repeatability(route_context: RouteContext) -> None:
    first = generate_candidate(route_context, GAConfig(), Random(42), by_length)
    second = generate_candidate(route_context, GAConfig(), Random(42), by_length)
    assert first == second


def test_no_baseline_injection(route_context: RouteContext) -> None:
    outcome = generate_candidate(route_context, GAConfig(), FixedWaypointRng((4,)), by_length)
    assert outcome.route == Route(((1, 4, 0), (4, 6, 0)))
    assert sum(route_context.graph[u][v][key]["length"] for u, v, key in outcome.route.edges) == 5


def test_invalid_weight_and_range(route_context: RouteContext) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        generate_candidate(route_context, GAConfig(), FixedWaypointRng((4,)), lambda edge, attrs, rng: 0)
    invalid = replace(GAConfig(), min_waypoints=0)
    assert generate_candidate(route_context, invalid, Random(0), by_length).reason == "invalid_waypoint_range"
