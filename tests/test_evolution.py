"""Одна смена поколения на известном графе без производственного Fitness."""

from dataclasses import replace
from random import Random

import pytest

from algorithm.genetic.evolution import (
    GenerationContext,
    RunInput,
    build_next_generation,
    process_invalid_candidate,
    run_evolution,
    should_stop,
)
from algorithm.genetic.initialization import generate_initial_population
from algorithm.evaluation.fitness import resolve_preset
from algorithm.genetic.mutation import MutationPolicy
from algorithm.genetic.selection import tournament_select
from algorithm.genetic.validation import validate_route
from algorithm.models import GAConfig, Individual, RoadEdge, Route, RouteContext
from config import default_config


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


def production_input(context: RouteContext, **ga_changes) -> RunInput:
    """Полный производственный расчёт на малом известном графе."""

    ga_values = dict(population_size=4, elite_count=1, max_copies=4,
                     attempts_per_slot=20, tournament_size=2, max_generations=3,
                     crossover_probability=0, mutation_probability=0, seed=7)
    ga_values.update(ga_changes)
    ga = replace(GAConfig(), **ga_values)
    base = default_config()
    fitness = replace(base.fitness, preset="short", weights=resolve_preset("short"),
                      normalization_mode="auto")
    roads = {}
    for u, v, key, attributes in context.graph.edges(keys=True, data=True):
        edge_id = (u, v, key)
        roads[edge_id] = RoadEdge(edge_id, ((0.0, 0.0), (0.001, 0.001)),
                                  "test", attributes["length"], "test", (1,), {},
                                  36.0, "observed", "maxspeed", None, "36")
    return RunInput(tuple(person.route for person in known_population(context)),
                    replace(base, ga=ga, fitness=fitness), context, roads)


def test_generation_zero_and_max_transitions(route_context: RouteContext) -> None:
    run = run_evolution(production_input(route_context))
    assert run.status == "completed" and run.stop_reason == "generation_limit", run.error
    assert run.transitions == 3
    assert [snapshot.generation for snapshot in run.history] == [0, 1, 2, 3]
    assert all(len(snapshot.population) == 4 for snapshot in run.history)
    assert all(snapshot.population[snapshot.best_index].j == run.best.j
               for snapshot in run.history)


def test_easy_keeps_ineligible_p0_and_never_selects_it(route_context: RouteContext) -> None:
    source = production_input(route_context, max_generations=1)
    easy = replace(source.config.fitness, preset="easy", weights=resolve_preset("easy"),
                   easy_length_delta=0, easy_time_delta=0)
    outcome = run_evolution(replace(source, config=replace(source.config, fitness=easy)))
    assert outcome.history and len(outcome.history[0].population) == 4
    assert tuple(person.route for person in outcome.history[0].population) == source.initial_population
    assert outcome.best is None or outcome.best.eligible
    assert all(not person.eligible for person in outcome.history[0].population
               if person.metrics.length_m > 3)


def test_easy_without_eligible_route_keeps_history_but_has_no_winner(route_context: RouteContext) -> None:
    source = production_input(route_context, max_generations=1)
    easy = replace(source.config.fitness, preset="easy", weights=resolve_preset("easy"),
                   easy_length_delta=0, easy_time_delta=0)
    long_route = source.initial_population[-1]
    outcome = run_evolution(replace(
        source, initial_population=(long_route,) * 4,
        config=replace(source.config, fitness=easy),
    ))
    assert outcome.status == "completed" and outcome.stop_reason == "no_eligible_route"
    assert outcome.best is None and len(outcome.history[0].population) == 4
    assert all(not person.eligible for person in outcome.history[0].population)


def test_eligible_ranked_before_lower_j_ineligible(route_context: RouteContext) -> None:
    route = production_input(route_context).initial_population[0]
    invalid = Individual(route, 0.01, eligible=False)
    valid = Individual(route, 2.0)
    assert tournament_select((invalid, valid), 2, Random(7)) is valid


def test_stagnation_on_off(route_context: RouteContext) -> None:
    enabled = run_evolution(production_input(route_context, stagnation_enabled=True,
                                             stagnation_window=2, max_generations=5))
    disabled = run_evolution(production_input(route_context, stagnation_enabled=False,
                                              stagnation_window=2, max_generations=5))
    assert enabled.status == disabled.status == "completed"
    assert (enabled.stop_reason, enabled.transitions) == ("stagnation", 2)
    assert (disabled.stop_reason, disabled.transitions) == ("generation_limit", 5)


def test_repeatable_seed_and_frozen_history(route_context: RouteContext) -> None:
    run_input = production_input(route_context, crossover_probability=1,
                                 mutation_probability=1, max_generations=2)
    first = run_evolution(run_input)
    second = run_evolution(run_input)
    assert first.status == second.status == "completed"
    assert first.history == second.history
    assert first.best == second.best
    assert first.history[0].population[0].route == run_input.initial_population[0]


def test_partial_failure_preserves_only_complete_history(route_context: RouteContext) -> None:
    run = run_evolution(production_input(route_context, max_copies=1,
                                         attempts_per_slot=1))
    assert run.status == "failed" and run.stop_reason == "generation_failed"
    assert run.history and run.history[0].generation == 0
    assert run.failed_generation is not None
    assert run.failed_generation.population == ()


def test_unknown_speed_blocks_production(route_context: RouteContext) -> None:
    run_input = production_input(route_context)
    roads = dict(run_input.roads)
    edge_id = (1, 2, 0)
    roads[edge_id] = replace(roads[edge_id], speed_kph=None, speed_status="unknown")
    run = run_evolution(replace(run_input, roads=roads))
    assert run.status == "failed" and run.stop_reason == "unknown_required_speeds"
    assert run.history == ()


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
