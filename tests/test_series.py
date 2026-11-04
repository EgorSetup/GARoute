"""Ключ общих условий и снимок входа серии не смешивают результаты."""

from dataclasses import replace

from algorithm.evaluation.fitness import resolve_preset
from algorithm.evaluation.normalization import NormalizationScales
from algorithm.models import RouteContext
from interface.controls import (
    confirm_change, propose_setting_change, request_condition_change,
)
from interface.run_state import build_condition_key, build_run_snapshot, start_run
from test_results import completed, prepared_state


def test_shared_condition_key(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    key = build_condition_key(state)
    state.playback.speed = 2.5
    state.selected_run_id = "old"
    assert build_condition_key(state) == key
    state.traffic_levels[(1, 2, 0)] = 8
    assert build_condition_key(state) != key
    state.traffic_levels.clear()
    state.graph_version += 1
    assert build_condition_key(state) != key


def test_preset_exception(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    key = build_condition_key(state)
    state.config = replace(state.config, fitness=replace(
        state.config.fitness, preset="fast", weights=resolve_preset("fast"),
    ))
    assert build_condition_key(state) == key
    state.config = replace(state.config, fitness=replace(
        state.config.fitness, preset="custom", weights=(0.3, 0.3, 0.4),
    ))
    assert build_condition_key(state) != key
    custom = build_condition_key(state)
    state.custom_weights = (0.3, 0.3, 0.4)
    state.config = replace(state.config, fitness=replace(
        state.config.fitness, preset="short", weights=resolve_preset("short"),
    ))
    assert build_condition_key(state) == custom


def test_auto_scales_not_user_edit(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    key = build_condition_key(state)
    state.base_scales = NormalizationScales(5, 8, 2)
    assert build_condition_key(state) == key
    state.config = replace(state.config, fitness=replace(
        state.config.fitness, normalization_mode="manual",
        time_reference_s=5, length_reference_m=8, complexity_reference=2,
    ))
    assert build_condition_key(state) != key


def test_population_policy(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    snapshot = build_run_snapshot(state)
    base = state.base_population.population
    state.routing_graph[1][2][0]["length"] = 99
    state.traffic_levels[(1, 2, 0)] = 8
    assert snapshot.routes.graph[1][2][0]["length"] == 1
    assert snapshot.traffic_levels == {}
    assert snapshot.initial_population == base
    assert state.base_population.population == base


def test_incompatible_series_requires_confirmed_clear(route_context: RouteContext) -> None:
    state = prepared_state(route_context)
    completed(state)
    assert state.series_condition_key == build_condition_key(state)
    state.graph_version += 1  # Защита даже при обходе обычного UI-диалога.
    handle, error = start_run(state)
    assert handle is None and "Общие условия серии изменились" in error
    state.graph_version -= 1
    request_condition_change(state, propose_setting_change(state, "population_size", 5))
    assert state.pending_change is not None and state.saved_runs
    confirm_change(state)
    assert not state.saved_runs and state.series_condition_key is None
