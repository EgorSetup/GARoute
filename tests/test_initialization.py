"""Одна попытка P0 с явными искусственными весами и RNG."""

from dataclasses import replace
from random import Random

import pytest

from algorithm.genetic.initialization import (
    InitialLengthPolicy,
    check_initial_length,
    generate_candidate,
    generate_initial_population,
    minimum_valid_length,
    production_weight_strategy,
    route_signature,
)
from algorithm.genetic.validation import validate_route
from algorithm.models import EligibilityReport, GAConfig, Route, RouteContext


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


def test_initial_length_boundary(route_context: RouteContext) -> None:
    minimum = minimum_valid_length(route_context)
    assert minimum == 3
    assert check_initial_length(9, InitialLengthPolicy(minimum, GAConfig().initial_length_factor)).is_valid


def test_initial_length_rejected(route_context: RouteContext) -> None:
    minimum = minimum_valid_length(route_context)
    assert not check_initial_length(9.01, InitialLengthPolicy(minimum)).is_valid
    assert check_initial_length(5, InitialLengthPolicy(minimum)).is_valid


def test_initial_reference_invalid(route_context: RouteContext) -> None:
    for base in (None, 0, float("nan")):
        assert not check_initial_length(5, InitialLengthPolicy(base)).is_valid
    assert not check_initial_length(5, InitialLengthPolicy(3, 0)).is_valid
    assert not check_initial_length(float("inf"), InitialLengthPolicy(3)).is_valid
    assert not check_initial_length(True, InitialLengthPolicy(3)).is_valid
    assert not check_initial_length(5, InitialLengthPolicy(1e308, 3)).is_valid
    context = replace(route_context, closed_edges=frozenset(route_context.graph.edges(keys=True)))
    assert minimum_valid_length(context) is None


def test_fixed_population_size(route_context: RouteContext) -> None:
    config = replace(GAConfig(), population_size=5, attempts_per_slot=100)
    result = generate_initial_population(route_context, config, Random(11), by_length)
    assert result.error is None
    assert len(result.population) == result.accepted_count == 5
    assert all(validate_route(route, route_context).valid for route in result.population)


def test_easy_derives_from_common_base_without_mutation(route_context: RouteContext) -> None:
    config = replace(GAConfig(), population_size=4, max_copies=4, attempts_per_slot=100)
    base = generate_initial_population(route_context, config, Random(11), by_length)
    assert base.error is None

    def eligible(route: Route) -> EligibilityReport:
        length = sum(route_context.graph[u][v][key]["length"] for u, v, key in route.edges)
        return EligibilityReport(length <= 3, () if length <= 3 else ("easy_length",))

    assert any(not eligible(route).eligible for route in base.population)

    derived = generate_initial_population(
        route_context, config, Random(12), by_length, eligible, base.population,
    )
    assert derived.error is None and len(derived.population) == 4
    assert derived.attempts > 0
    assert all(eligible(route).eligible for route in derived.population)
    assert base.population == generate_initial_population(
        route_context, config, Random(11), by_length,
    ).population


def test_easy_base_shortage_does_not_publish_partial_population(route_context: RouteContext) -> None:
    config = replace(GAConfig(), population_size=3, attempts_per_slot=2)
    base = (Route(((1, 4, 0), (4, 6, 0))),)
    result = generate_initial_population(
        route_context, config, FixedWaypointRng((4,)), by_length,
        lambda route: EligibilityReport(False, ("easy_time",)), base,
    )
    assert result.error == "attempts_exhausted"
    assert result.population == () and result.accepted_count == 0
    assert dict(result.rejections)["easy_time"] >= 1


def test_duplicate_limit(route_context: RouteContext) -> None:
    config = replace(GAConfig(), population_size=3, max_copies=2, attempts_per_slot=3)
    result = generate_initial_population(route_context, config, FixedWaypointRng((4,)), by_length)
    assert result.error == "attempts_exhausted"
    assert result.population == () and result.accepted_count == 2
    assert result.attempts == 5
    assert dict(result.rejections)["duplicate_limit"] == 3
    assert route_signature(Route(((1, 2, 0),))) != route_signature(Route(((1, 2, 1),)))


def test_attempt_limit(route_context: RouteContext) -> None:
    config = replace(GAConfig(), population_size=2, max_copies=1, attempts_per_slot=2)
    result = generate_initial_population(route_context, config, FixedWaypointRng((4,)), by_length)
    assert result.error == "attempts_exhausted"
    assert result.attempts == 3


def test_no_auto_relaxation(route_context: RouteContext) -> None:
    config = replace(GAConfig(), population_size=3, max_copies=1, attempts_per_slot=1)
    result = generate_initial_population(route_context, config, FixedWaypointRng((4,)), by_length)
    assert result.error == "attempts_exhausted"
    assert result.population == () and result.accepted_count == 1
    assert dict(result.rejections) == {"duplicate_limit": 1}
    invalid = generate_initial_population(route_context, replace(config, population_size=0), Random(0), by_length)
    assert invalid.error == "invalid_population_config"
    short_only = replace(config, population_size=1, initial_length_factor=1, attempts_per_slot=2)
    too_long = generate_initial_population(route_context, short_only, FixedWaypointRng((4,)), by_length)
    assert too_long.error == "attempts_exhausted"
    assert dict(too_long.rejections) == {"length_limit": 2}


def test_production_weights_and_seed(route_context: RouteContext) -> None:
    strategy = production_weight_strategy(1.0)
    weights_a = [strategy((1, 2, 0), {"length": 100}, Random(seed)) for seed in (1, 2, 3)]
    weights_b = [strategy((1, 2, 0), {"length": 100}, Random(seed)) for seed in (1, 2, 3)]
    assert weights_a == weights_b
    assert all(50 <= weight <= 200 for weight in weights_a)
    with pytest.raises(ValueError):
        production_weight_strategy(0)
    config = replace(GAConfig(), population_size=5)
    first = generate_initial_population(route_context, config, Random(17), strategy)
    second = generate_initial_population(route_context, config, Random(17), strategy)
    assert first == second
    assert first.error is None
