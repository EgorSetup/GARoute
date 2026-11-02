"""Одна смена поколения на известном графе без производственного Fitness."""

from dataclasses import replace
from random import Random

import pytest

from algorithm.genetic.evolution import (
    GenerationContext,
    build_next_generation,
    process_invalid_candidate,
)
from algorithm.genetic.initialization import generate_initial_population
from algorithm.genetic.mutation import MutationPolicy
from algorithm.genetic.validation import validate_route
from algorithm.models import GAConfig, Individual, Route, RouteContext


def length_weight(edge, attributes, rng):
    return attributes["length"]


def evaluate_for(context: RouteContext):
    def evaluate(route: Route) -> Individual:
        length = sum(context.graph[u][v][key]["length"] for u, v, key in route.edges)
        return Individual(route, length)
    return evaluate


def known_population(context: RouteContext) -> tuple[Individual, ...]:
    evaluate = evaluate_for(context)
    routes = (
        Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))),
        Route(((1, 2, 1), (2, 5, 0), (5, 6, 0))),
        Route(((1, 3, 0), (3, 5, 0), (5, 6, 0))),
        Route(((1, 4, 0), (4, 6, 0))),
    )
    return tuple(map(evaluate, routes))


def run_context(context: RouteContext, **changes) -> GenerationContext:
    base = replace(GAConfig(), population_size=4, elite_count=1, max_copies=4,
                   attempts_per_slot=20, tournament_size=2)
    config = replace(base, **changes)
    return GenerationContext(context, config, MutationPolicy(length_weight))


def test_known_route_costs(route_context: RouteContext) -> None:
    assert [person.j for person in known_population(route_context)] == [3, 3, 3, 5]


def test_elites_preserved(route_context: RouteContext) -> None:
    current = known_population(route_context)
    context = run_context(route_context, elite_count=2)
    result = build_next_generation(current, context, Random(2), evaluate_for(route_context))
    assert result.error is None
    assert result.population[0] is current[0]
    assert result.population[1] is current[1]
    assert current == known_population(route_context)


def test_exact_size(route_context: RouteContext) -> None:
    context = run_context(route_context, crossover_probability=0, mutation_probability=0)
    result = build_next_generation(known_population(route_context), context, Random(4), evaluate_for(route_context))
    assert result.error is None and len(result.population) == 4
    assert result.crossover_attempts == result.mutation_attempts == 0
    assert all(validate_route(person.route, route_context).valid for person in result.population)


def test_duplicate_limit_with_elites(route_context: RouteContext) -> None:
    config = replace(GAConfig(), population_size=4, elite_count=1, max_copies=1,
                     attempts_per_slot=2, tournament_size=4,
                     crossover_probability=0, mutation_probability=0)
    context = GenerationContext(route_context, config, MutationPolicy(length_weight))
    result = build_next_generation(known_population(route_context), context, Random(0), evaluate_for(route_context))
    assert result.error == "attempts_exhausted"
    assert result.population == () and result.accepted_count == 3
    assert dict(result.rejections)["duplicate_limit"] == 2


def test_no_repair_called(route_context: RouteContext) -> None:
    candidate = Route(((1, 2, 0), (2, 3, 0), (3, 5, 0), (5, 6, 0)))
    report = validate_route(candidate, route_context)
    assert not report.valid
    assert process_invalid_candidate(candidate, report) is None
    assert candidate.edges == ((1, 2, 0), (2, 3, 0), (3, 5, 0), (5, 6, 0))


def test_generation_exhaustion(route_context: RouteContext) -> None:
    context = run_context(route_context, attempts_per_slot=1,
                          crossover_probability=0, mutation_probability=0)
    bad_eval = lambda route: Individual(route, float("nan"))
    result = build_next_generation(known_population(route_context), context, Random(0), bad_eval)
    assert result.error == "attempts_exhausted"
    assert result.population == () and result.accepted_count == 1
    assert dict(result.rejections)["invalid_evaluation"] == 1


@pytest.mark.parametrize("method", ("A", "B"))
def test_synthetic_generation_pipeline(route_context: RouteContext, method: str) -> None:
    config = replace(GAConfig(), population_size=4, elite_count=1, max_copies=4,
                     attempts_per_slot=100, tournament_size=2,
                     crossover_probability=1, mutation_probability=1,
                     mutation_method=method)
    initial = generate_initial_population(route_context, config, Random(9), length_weight)
    assert initial.error is None and len(initial.population) == 4
    evaluate = evaluate_for(route_context)
    current = tuple(map(evaluate, initial.population))
    before = tuple(person.route.edges for person in current)
    result = build_next_generation(
        current, GenerationContext(route_context, config, MutationPolicy(length_weight)),
        Random(5), evaluate,
    )
    assert result.error is None and len(result.population) == 4
    assert result.crossover_attempts > 0 and result.mutation_attempts > 0
    assert all(validate_route(person.route, route_context).valid for person in result.population)
    assert tuple(person.route.edges for person in current) == before
