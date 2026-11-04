"""Один фоновый запуск, копия условий и приём результата по run_id."""

from dataclasses import replace

from algorithm.genetic.evolution import run_evolution
from algorithm.genetic.initialization import PopulationResult
from algorithm.models import RouteContext
from data.road_graph import NormalizedRoadData
from interface.controls import UIState, PendingChange, request_condition_change
from interface.run_state import start_run, apply_run_completion
from test_evolution import production_input


def prepared_state(context: RouteContext) -> UIState:
    run_input = production_input(context)
    state = UIState(config=run_input.config, routing_graph=context.graph,
                    turn_rules=context.turn_rules, endpoint_a=1, endpoint_b=6)
    state.normalized_roads = NormalizedRoadData(context.graph, run_input.roads,
                                                {}, (), (), {})
    state.initial_population = PopulationResult(run_input.initial_population, 4, 4, ())
    return state


def test_single_run_frozen_inputs_and_stale_completion_ignored(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    handle, error = start_run(state)
    assert error is None and handle is not None
    assert start_run(state)[0] is None
    candidate = PendingChange("seed", 3,
                              replace(state.config, ga=replace(state.config.ga, seed=3)))
    assert request_condition_change(state, candidate).errors
    route_context.graph[1][2][0]["length"] = 99
    assert handle.run_input.routes.graph[1][2][0]["length"] == 1
    outcome = run_evolution(handle.run_input)
    assert not apply_run_completion(state, "old-id", outcome)
    assert state.active_run_id == handle.run_id
    refreshed = []
    state.refresh_playback = lambda: refreshed.append(state.playback.generation_index)
    state.refresh_charts = lambda: refreshed.append(len(state.generation_history))
    assert apply_run_completion(state, handle.run_id, outcome)
    assert state.run_status == "completed" and len(state.run_results) == 1
    assert len(state.generation_history) == len(outcome.history)
    assert state.playback.generation_count == len(outcome.history)
    assert state.playback.generation_index == 0 and not state.playback.playing
    assert refreshed == [0, len(outcome.history)]


def test_failure_not_success(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    handle, _ = start_run(state)
    failure = replace(run_evolution(handle.run_input), status="failed",
                      stop_reason="generation_failed", error="attempts_exhausted")
    assert apply_run_completion(state, handle.run_id, failure)
    assert state.run_results == [] and state.generation_history == []
    assert state.run_error == "attempts_exhausted"


def test_preflight_rejects_busy_p0_pending_change_and_unknown_speed(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    state.initial_population_running = True
    assert "P0" in start_run(state)[1]
    state.initial_population_running = False
    state.pending_change = PendingChange("seed", 3, state.config)
    assert "изменение" in start_run(state)[1]
    state.pending_change = None
    edge_id = (1, 2, 0)
    roads = dict(state.normalized_roads.edges)
    roads[edge_id] = replace(roads[edge_id], speed_kph=None, speed_status="unknown")
    state.normalized_roads = replace(state.normalized_roads, edges=roads)
    handle, error = start_run(state)
    assert handle is None and "Неизвестна скорость" in error
    assert state.active_run_id is None
