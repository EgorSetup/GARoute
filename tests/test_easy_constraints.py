"""Два независимых жёстких ограничения «Лёгкого» Q05 VERIFY."""

from dataclasses import replace
from math import inf
from random import Random

import pytest

from algorithm.evaluation.fitness import (
    TimeModel, check_easy_constraints, compute_easy_references,
)
from algorithm.genetic.evolution import GenerationContext, build_next_generation
from algorithm.genetic.initialization import generate_initial_population
from algorithm.genetic.mutation import MutationPolicy
from algorithm.models import (
    EasyLimits, EasyReferences, EligibilityReport, GAConfig, Individual, RoadEdge,
    Route, RouteMetrics,
)
from config import default_config, validate_config
from interface.controls import UIState, propose_fitness_change


REFS = EasyReferences(1000, 100)
LIMITS = EasyLimits(0.5, 0.5)


def test_both_limits_and_exact_boundaries():
    assert check_easy_constraints(RouteMetrics(1500, 150, 99), REFS, LIMITS).eligible
    assert check_easy_constraints(RouteMetrics(1501, 150, 0), REFS, LIMITS).reasons == (
        "easy_length_exceeded",)
    assert check_easy_constraints(RouteMetrics(1500, 151, 0), REFS, LIMITS).reasons == (
        "easy_time_exceeded",)
    assert check_easy_constraints(RouteMetrics(1501, 151, 0), REFS, LIMITS).reasons == (
        "easy_length_exceeded", "easy_time_exceeded")


def test_no_relaxation_when_no_route_is_eligible():
    population = [RouteMetrics(1600, 80, 0), RouteMetrics(800, 160, 0)]
    accepted = [item for item in population if check_easy_constraints(item, REFS, LIMITS).eligible]
    assert accepted == []
    assert LIMITS == EasyLimits(0.5, 0.5)


@pytest.mark.parametrize("references,limits", [
    (EasyReferences(0, 100), LIMITS),
    (REFS, EasyLimits(-0.1, 0.5)),
    (REFS, EasyLimits(0.5, 2.1)),
    (REFS, EasyLimits(inf, 0.5)),
])
def test_invalid_references_and_limits(references, limits):
    with pytest.raises(ValueError):
        check_easy_constraints(RouteMetrics(1000, 100, 0), references, limits)


def test_working_defaults_and_ui_range():
    config = default_config()
    assert config.fitness.easy_length_delta == config.fitness.easy_time_delta == 0.5
    assert validate_config(replace(config, fitness=replace(config.fitness, easy_length_delta=2.1))).errors
    state = UIState()
    change = propose_fitness_change(state, "easy_length_delta", 200)
    assert change.candidate.fitness.easy_length_delta == 2
    assert propose_fitness_change(state, "easy_time_delta", 201).errors


def test_easy_references_are_separate_from_normalization(route_context):
    roads = {}
    for edge_id in route_context.graph.edges(keys=True):
        length = route_context.graph[edge_id[0]][edge_id[1]][edge_id[2]]["length"]
        speed = 72 if edge_id[0] == 3 else 36
        roads[edge_id] = RoadEdge(
            edge_id, ((0.0, 0.0), (0.001, 0.001)), "test", length, "test",
            (1,), {}, speed, "observed", "maxspeed", None, speed,
        )
    references = compute_easy_references(route_context, TimeModel(roads))
    assert references.length_min_m == 3
    assert references.time_min_s > 0
    roads[(1, 2, 0)] = replace(roads[(1, 2, 0)], speed_kph=None)
    with pytest.raises(ValueError, match="unknown required speeds"):
        compute_easy_references(route_context, TimeModel(roads))


def test_p0_rejects_ineligible_without_partial_success(route_context):
    config = replace(GAConfig(), population_size=4, attempts_per_slot=20)
    result = generate_initial_population(
        route_context, config, Random(17), lambda edge, attrs, rng: attrs["length"],
        eligible=lambda route: EligibilityReport(False, ("easy_length_exceeded",)),
    )
    assert result.error == "attempts_exhausted"
    assert result.population == () and result.accepted_count == 0
    assert result.attempts == 20
    assert dict(result.rejections)["easy_length_exceeded"] > 0


def test_generation_rejects_ineligible_candidates(route_context):
    routes = (
        Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))),
        Route(((1, 2, 1), (2, 5, 0), (5, 6, 0))),
        Route(((1, 3, 0), (3, 5, 0), (5, 6, 0))),
        Route(((1, 4, 0), (4, 6, 0))),
    )
    current = tuple(Individual(route, 1, RouteMetrics(1000, 100, 0)) for route in routes)
    config = replace(GAConfig(), population_size=4, elite_count=1, max_copies=4,
                     attempts_per_slot=2, tournament_size=2,
                     crossover_probability=0, mutation_probability=0)
    admissible = lambda metrics: check_easy_constraints(metrics, REFS, LIMITS)
    context = GenerationContext(route_context, config,
                                MutationPolicy(lambda edge, attrs, rng: attrs["length"]),
                                admissible)
    result = build_next_generation(
        current, context, Random(1),
        lambda route: Individual(route, 1, RouteMetrics(2000, 200, 0)),
    )
    assert result.error == "attempts_exhausted"
    assert result.population == () and result.accepted_count == 1
    assert dict(result.rejections)["easy_length_exceeded"] == 2
    assert dict(result.rejections)["easy_time_exceeded"] == 2
