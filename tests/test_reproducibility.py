"""Повтор всей вычислительной истории при фиксированных условиях и seed."""

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
    assert first.seed_used == second.seed_used == 7
    assert first.history == second.history
    assert first.scales == second.scales
    assert first.best == second.best
    assert first.stop_reason == second.stop_reason


def test_generated_seed_replay(route_context: RouteContext) -> None:
    original = production_input(route_context, max_generations=2)
    no_seed = replace(original, config=replace(
        original.config, ga=replace(original.config.ga, seed=None),
    ))
    first = run_evolution(no_seed)
    assert first.status == "completed" and type(first.seed_used) is int
    replay = replace(no_seed, config=replace(
        no_seed.config, ga=replace(no_seed.config.ga, seed=first.seed_used),
    ))
    second = run_evolution(replay)
    assert first.history == second.history
    assert first.scales == second.scales
    assert first.stop_reason == second.stop_reason


def test_initial_population_seed_replay_and_ui_transfer(route_context: RouteContext) -> None:
    run_input = production_input(route_context, attempts_per_slot=100)
    seed = 11
    first = generate_initial_population(route_context, run_input.config.ga,
                                        Random(seed), production_weight_strategy(1.0))
    second = generate_initial_population(route_context, run_input.config.ga,
                                         Random(seed), production_weight_strategy(1.0))
    assert first.error is None and first == second
    state = prepared_state(route_context)
    state.config = replace(state.config, ga=replace(state.config.ga, seed=None))
    state.initial_population = first
    state.initial_seed_used = seed
    handle, error = start_run(state)
    assert error is None and handle.run_input.config.ga.seed == seed


def test_elitist_best_bound(route_context: RouteContext) -> None:
    result = run_evolution(production_input(route_context, max_generations=5,
                                            elite_count=1, max_copies=4))
    assert result.status == "completed"
    best_costs = [snapshot.population[snapshot.best_index].j
                  for snapshot in result.history]
    assert len(best_costs) == 6
    assert all(later <= earlier for earlier, later in zip(best_costs, best_costs[1:]))
    assert result.best.j == min(best_costs)
