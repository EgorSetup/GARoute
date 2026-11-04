"""Транзакция условий на искусственной истории, не видимой в приложении."""

import pytest

from algorithm.genetic.evolution import run_evolution
from algorithm.models import RouteContext
from data.traffic import CoveredInterval, SegmentCollection, VisualSegment
from interface.controls import (
    UIState,
    cancel_change,
    confirm_change,
    propose_display_change,
    propose_fitness_change,
    propose_penalty_change,
    propose_setting_change,
    propose_traffic_edit,
    request_condition_change,
    select_endpoint,
)
from interface.run_state import apply_run_completion, start_run
from interface.map_view import clear_run_layers
from test_map_layers import FakeMap
from test_results import completed, prepared_state, switch_preset
from visualization.animation import advance
from visualization.animation import PlaybackState


def fixture_state() -> UIState:
    state = UIState()
    state.run_results.append(object())
    state.generation_history.append(object())
    state.chart_data["convergence"] = [1.0]
    state.selected_run_id = "test-run"
    state.highlighted_route_ids.add("test-route")
    state.playback = PlaybackState(2, 1, True, 2.0)
    return state


def test_cancel_preserves_all() -> None:
    state = fixture_state()
    old = state.config
    result = state.run_results[0]
    change = propose_setting_change(state, "population_size", 12)
    request_condition_change(state, change)
    assert state.pending_change is not None
    assert state.config == old
    cancel_change(state)
    assert state.pending_change is None
    assert state.config == old
    assert state.run_results == [result]
    assert state.playback.playing
    assert state.selected_run_id == "test-run"


def test_confirm_clears_all() -> None:
    state = fixture_state()
    change = propose_setting_change(state, "population_size", 12)
    request_condition_change(state, change)
    confirm_change(state)
    assert state.config.ga.population_size == 12
    assert state.pending_change is None
    assert state.run_results == []
    assert state.generation_history == []
    assert state.chart_data == {}
    assert state.selected_run_id is None
    assert state.highlighted_route_ids == set()
    assert not state.playback.has_history
    assert not state.playback.playing


def test_preset_keeps_results() -> None:
    state = fixture_state()
    change = propose_fitness_change(state, "preset", "fast")
    request_condition_change(state, change)
    assert state.pending_change is None
    assert state.config.fitness.preset == "fast"
    assert len(state.run_results) == 1


def test_display_change_keeps_results() -> None:
    state = fixture_state()
    change = propose_display_change(state, "playback_speed", 1.5)
    request_condition_change(state, change)
    assert state.pending_change is None
    assert state.playback.speed == 1.5
    assert len(state.run_results) == 1
    assert state.generation_history


def test_reselect_manual_mode_keeps_references() -> None:
    state = UIState()
    request_condition_change(state, propose_fitness_change(state, "normalization_mode", "manual"))
    request_condition_change(state, propose_fitness_change(state, "time_reference_s", 600))
    state.run_results.append(object())
    request_condition_change(state, propose_fitness_change(state, "normalization_mode", "manual"))
    assert state.pending_change is None
    assert state.config.fitness.time_reference_s == 600
    assert len(state.run_results) == 1


def two_run_state(context: RouteContext) -> UIState:
    state = prepared_state(context)
    completed(state)
    switch_preset(state, "fast")
    completed(state)
    state.playback.playing = True
    state.chart_data["old"] = [1.0]
    return state


@pytest.mark.parametrize("kind", (
    "endpoint_a", "endpoint_b", "traffic", "ga", "penalty", "easy_limit", "normalization",
))
def test_real_series_cancel_and_confirm_each_condition(route_context: RouteContext,
                                                        kind: str) -> None:
    state = two_run_state(route_context)
    if kind == "traffic":
        edge = state.normalized_roads.edges[(1, 2, 0)]
        segment = VisualSegment("test", 1, "Тест", edge.geometry_lonlat, edge.length_m,
                                (CoveredInterval(edge.edge_id, 0, edge.length_m),))
        state.traffic_segments = SegmentCollection((segment,), (), ((edge.edge_id, edge.length_m),))

    def proposal():
        if kind == "endpoint_a":
            return select_endpoint(state, "A", 2)
        if kind == "endpoint_b":
            return select_endpoint(state, "B", 5)
        if kind == "traffic":
            return propose_traffic_edit(state, "test", 1)
        if kind == "ga":
            return propose_setting_change(state, "population_size", 5)
        if kind == "penalty":
            return propose_penalty_change(state, 0, 3)
        if kind == "easy_limit":
            return propose_fitness_change(state, "easy_length_delta", 60)
        return propose_fitness_change(state, "normalization_mode", "manual")

    previous_config = state.config
    previous_runs = tuple(state.saved_runs)
    previous_history = tuple(state.generation_history)
    previous_key = state.series_condition_key
    change = proposal()
    assert not change.errors
    request_condition_change(state, change)
    assert state.pending_change is not None
    cancel_change(state)
    assert state.config == previous_config
    assert tuple(state.saved_runs) == previous_runs
    assert tuple(state.generation_history) == previous_history
    assert state.series_condition_key == previous_key
    assert state.playback.playing and state.selected_run_id == previous_runs[-1].run_id

    observed = []
    state.refresh_playback = lambda: observed.append(("playback", len(state.generation_history)))
    state.refresh_charts = lambda: observed.append(("charts", len(state.generation_history)))
    state.refresh_runs = lambda: observed.append(("cards", len(state.saved_runs)))
    state.refresh_run_overlays = lambda: observed.append(("layers", len(state.saved_runs)))
    request_condition_change(state, proposal())
    confirm_change(state)
    assert not state.saved_runs and not state.run_results and not state.generation_history
    assert state.base_population is None and state.base_scales is None
    assert state.selected_run_id is None and state.series_condition_key is None
    assert not state.playback.playing and not advance(state.playback, 100)
    assert not state.chart_data
    assert ("playback", 0) in observed and ("charts", 0) in observed
    assert ("cards", 0) in observed and ("layers", 0) in observed


def test_late_result_cannot_restore_cleared_series(route_context: RouteContext) -> None:
    state = two_run_state(route_context)
    handle, error = start_run(state)
    assert error is None and handle is not None
    outcome = run_evolution(handle.run_input)
    assert not apply_run_completion(state, "obsolete-id", outcome)
    assert state.active_run_id == handle.run_id
    assert request_condition_change(
        state, propose_setting_change(state, "population_size", 5),
    ).errors
    assert apply_run_completion(state, handle.run_id, outcome)
    request_condition_change(state, propose_setting_change(state, "population_size", 5))
    confirm_change(state)
    assert not apply_run_completion(state, handle.run_id, outcome)
    assert not state.saved_runs and not state.generation_history


def test_confirm_removes_run_playback_and_population_layers(route_context: RouteContext) -> None:
    state = two_run_state(route_context)
    map_element = FakeMap()
    run_layers = [object(), object()]
    playback_layers = [object()]
    population_layers = [object()]
    state.refresh_run_overlays = lambda: clear_run_layers(map_element, run_layers)
    state.show_generation_frame = lambda frame: clear_run_layers(map_element, playback_layers) if frame is None else None
    state.refresh_playback = lambda: state.show_generation_frame(None) if not state.generation_history else None
    state.clear_population_layers = lambda: clear_run_layers(map_element, population_layers)

    request_condition_change(state, propose_setting_change(state, "population_size", 5))
    cancel_change(state)
    assert len(map_element.removed) == 0
    assert len(run_layers) == 2 and len(playback_layers) == len(population_layers) == 1

    request_condition_change(state, propose_setting_change(state, "population_size", 5))
    confirm_change(state)
    assert run_layers == playback_layers == population_layers == []
    assert len(map_element.removed) == 4


def test_preset_and_display_keep_real_series(route_context: RouteContext) -> None:
    state = two_run_state(route_context)
    saved = tuple(state.saved_runs)
    history = tuple(state.generation_history)
    request_condition_change(state, propose_fitness_change(state, "preset", "short"))
    request_condition_change(state, propose_display_change(state, "playback_speed", 2))
    assert state.pending_change is None
    assert tuple(state.saved_runs) == saved and tuple(state.generation_history) == history
    assert state.playback.speed == 2 and state.playback.playing
