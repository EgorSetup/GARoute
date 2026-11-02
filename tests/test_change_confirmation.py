"""Транзакция условий на искусственной истории, не видимой в приложении."""

from interface.controls import (
    UIState,
    cancel_change,
    confirm_change,
    propose_display_change,
    propose_fitness_change,
    propose_setting_change,
    request_condition_change,
)
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
