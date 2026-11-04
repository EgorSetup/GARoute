"""Внутренняя воспроизводимость ГА и генератора без пользовательского seed."""

from dataclasses import replace
from random import Random

from algorithm.genetic.evolution import run_evolution
from algorithm.genetic.initialization import (
    generate_initial_population, production_weight_strategy,
)
from algorithm.models import RouteContext
from interface.run_state import start_run
from test_evolution import production_input
from test_run_state import prepared_state


def test_same_seed_same_history(route_context: RouteContext) -> None:
    run_input = production_input(route_context, crossover_probability=1,
                                 mutation_probability=1, max_generations=3)
    first = run_evolution(run_input)
    second = run_evolution(run_input)
    assert first.status == second.status == "completed"
    assert first.history == second.history
    assert first.scales == second.scales
    assert first.best == second.best
    assert first.stop_reason == second.stop_reason


def test_internal_rng_seed_replay(route_context: RouteContext) -> None:
    original = production_input(route_context, max_generations=2)
    first = run_evolution(original)
    replay = replace(original, rng_seed=7)
    second = run_evolution(replay)
    assert first.history == second.history
    assert first.scales == second.scales
    assert first.stop_reason == second.stop_reason


def test_initial_population_internal_rng_and_ui_snapshot(route_context: RouteContext) -> None:
    run_input = production_input(route_context, attempts_per_slot=100)
    seed = 11
    first = generate_initial_population(route_context, run_input.config.ga,
                                        Random(seed), production_weight_strategy(1.0))
    second = generate_initial_population(route_context, run_input.config.ga,
                                         Random(seed), production_weight_strategy(1.0))
    assert first.error is None and first == second
    state = prepared_state(route_context)
    state.initial_population = first
    handle, error = start_run(state)
    assert error is None and handle.run_input.initial_population == first.population
    assert not hasattr(handle.run_input.config.ga, "seed")


def test_elitist_best_bound(route_context: RouteContext) -> None:
    result = run_evolution(production_input(route_context, max_generations=5,
                                            elite_count=1, max_copies=4))
    assert result.status == "completed"
    best_costs = [snapshot.population[snapshot.best_index].j
                  for snapshot in result.history]
    assert len(best_costs) == 6
    assert all(later <= earlier for earlier, later in zip(best_costs, best_costs[1:]))
    assert result.best.j == min(best_costs)
