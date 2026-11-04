"""Независимые запуски пресетов и выбор их сохранённой истории."""

from dataclasses import replace

from algorithm.genetic.evolution import run_evolution
from algorithm.genetic.initialization import PopulationResult
from algorithm.evaluation.fitness import resolve_preset
from algorithm.evaluation.normalization import NormalizationScales
from algorithm.models import RouteContext
from data.road_graph import NormalizedRoadData
from interface.controls import UIState
from interface.run_state import apply_run_completion, select_run, set_run_visibility, start_run
from test_evolution import production_input


def prepared_state(context: RouteContext) -> UIState:
    source = production_input(context)
    state = UIState(config=source.config, routing_graph=context.graph,
                    turn_rules=context.turn_rules, endpoint_a=context.start, endpoint_b=context.end)
    state.normalized_roads = NormalizedRoadData(context.graph, source.roads, {}, (), (), {})
    state.base_population = PopulationResult(source.initial_population, 4, 4, ())
    state.initial_population = state.base_population
    state.base_seed_used = 7
    state.initial_seed_used = 7
    state.base_scales = NormalizationScales(1, 3.5, 1)
    return state


def completed(state: UIState) -> str:
    handle, error = start_run(state)
    assert handle is not None and error is None
    result = run_evolution(handle.run_input)
    assert result.status == "completed", result.error
    assert apply_run_completion(state, handle.run_id, result)
    return handle.run_id


def switch_preset(state: UIState, preset: str) -> None:
    state.config = replace(state.config, fitness=replace(
        state.config.fitness, preset=preset, weights=resolve_preset(preset),
    ))


def test_presets_accumulate(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    first = completed(state)
    switch_preset(state, "fast")
    second = completed(state)
    assert first != second and len(state.saved_runs) == len(state.run_results) == 2
    assert [run.config.fitness.preset for run in state.saved_runs] == ["short", "fast"]
    assert state.saved_runs[0].outcome.history[0].population[0].route == state.saved_runs[1].outcome.history[0].population[0].route
    assert state.saved_runs[0].outcome.scales == state.saved_runs[1].outcome.scales == state.base_scales


def test_run_metadata(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    run_id = completed(state)
    record = state.saved_runs[0]
    assert (record.run_id, record.start, record.end, record.seed_used) == (run_id, 1, 6, 7)
    assert record.outcome.best is not None and record.outcome.best.metrics is not None
    assert record.outcome.stop_reason == "generation_limit"
    assert record.uncertain_penalties >= 0


def test_same_route_different_presets(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    completed(state)
    switch_preset(state, "fast")
    completed(state)
    assert state.saved_runs[0].outcome.best.route == state.saved_runs[1].outcome.best.route
    assert state.saved_runs[0].run_id != state.saved_runs[1].run_id


def test_select_own_history(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    first = completed(state)
    switch_preset(state, "fast")
    second = completed(state)
    refreshed = []
    state.refresh_charts = lambda: refreshed.append(state.selected_run_id)
    assert select_run(state, first)
    assert state.generation_history == list(state.saved_runs[0].outcome.history)
    assert state.playback.generation_index == 0
    assert refreshed == [first]
    assert not select_run(state, "missing")
    assert set_run_visibility(state, second, False)
    assert state.saved_runs[1].visible is False and state.saved_runs[0].visible is True
    assert not set_run_visibility(state, "missing", True)


def test_repeated_preset_policy(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    first = completed(state)
    second = completed(state)
    assert first != second and len(state.saved_runs) == 2
    assert all(run.config.fitness.preset == "short" for run in state.saved_runs)
    assert state.base_population.population == tuple(
        person.route for person in state.saved_runs[0].outcome.history[0].population
    )


def test_easy_uses_filtered_p0_and_shared_scales(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    base = state.base_population.population
    state.initial_population = PopulationResult((base[0], base[1], base[2], base[0]), 4, 1, ())
    state.initial_population_preset = "easy"
    switch_preset(state, "easy")
    completed(state)
    record = state.saved_runs[0]
    assert record.outcome.scales == state.base_scales
    assert tuple(person.route for person in record.outcome.history[0].population) == state.initial_population.population
    assert state.base_population.population == base
