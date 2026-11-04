"""Явная модель скоростей и общий ключ серии без изменения исходного OSM."""

from dataclasses import replace

from algorithm.genetic.evolution import run_evolution
from algorithm.evaluation.fitness import TimeModel, resolve_speed, unknown_required_speeds
from interface.controls import cancel_change, confirm_change, propose_fitness_change, request_condition_change
from interface.run_state import apply_run_completion, build_condition_key, start_run
from test_results import completed, prepared_state


def test_mode_settings_require_confirmation_after_run(route_context):
    state = prepared_state(route_context)
    completed(state)
    before = build_condition_key(state)
    proposal = propose_fitness_change(state, "modeled_speeds_enabled", True)
    assert not request_condition_change(state, proposal).errors
    assert state.pending_change is proposal
    cancel_change(state)
    assert not state.config.fitness.modeled_speeds_enabled and state.saved_runs
    request_condition_change(state, proposal)
    confirm_change(state)
    assert state.config.fitness.modeled_speeds_enabled
    assert not state.saved_runs and build_condition_key(state) != before


def test_manual_class_speed_survives_run_snapshot(route_context):
    state = prepared_state(route_context)
    edge_id = (1, 2, 0)
    roads = dict(state.normalized_roads.edges)
    roads[edge_id] = replace(roads[edge_id], speed_kph=None, speed_status="unknown",
                             speed_source=None, raw_speed=None,
                             raw_attributes={"highway": "living_street"})
    state.normalized_roads = replace(state.normalized_roads, edges=roads)
    population, scales = state.base_population, state.base_scales
    assert unknown_required_speeds(route_context, TimeModel(roads)) == (edge_id,)
    assert start_run(state)[0] is None
    for key, value in (("modeled_speed", ("living_street", 24)),
                       ("modeled_speeds_enabled", True)):
        proposal = propose_fitness_change(state, key, value)
        assert not request_condition_change(state, proposal).errors
        if state.pending_change is not None:
            confirm_change(state)
    state.base_population = state.initial_population = population
    state.base_scales = scales
    handle, error = start_run(state)
    assert error is None and handle is not None
    assert handle.run_input.roads[edge_id].speed_status == "unknown"
    policy = handle.run_input.config.fitness
    model = TimeModel(handle.run_input.roads, modeled_speeds_enabled=policy.modeled_speeds_enabled,
                      modeled_speeds_kph=dict(policy.modeled_speeds_kph))
    assert resolve_speed(handle.run_input.roads[edge_id], model).source == "manual_model:living_street"
    outcome = run_evolution(handle.run_input)
    assert outcome.status == "completed", outcome.error
    repeated = run_evolution(handle.run_input)
    assert repeated.status == "completed"
    assert repeated.best == outcome.best and repeated.history == outcome.history
    assert apply_run_completion(state, handle.run_id, outcome)
    assert state.saved_runs[0].config.fitness.modeled_speeds_kph == (("living_street", 24.0),)
    expected = sum(edge.edge_id == edge_id for edge in (
        handle.run_input.roads[key] for key in outcome.best.route.edges
    ))
    assert state.saved_runs[0].modeled_speed_usage == (
        (("living_street", expected),) if expected else ()
    )


def test_uncovered_or_ambiguous_speed_stays_blocked(route_context):
    state = prepared_state(route_context)
    edge_id = (1, 2, 0)
    roads = dict(state.normalized_roads.edges)
    roads[edge_id] = replace(roads[edge_id], speed_kph=None, speed_status="unknown",
                             raw_speed=None, raw_attributes={"highway": "living_street"})
    state.normalized_roads = replace(state.normalized_roads, edges=roads)
    state.config = replace(state.config, fitness=replace(
        state.config.fitness, modeled_speeds_enabled=True,
        modeled_speeds_kph=(("unclassified", 30.0),),
    ))
    assert "living_street: 1" in start_run(state)[1]
    roads[edge_id] = replace(roads[edge_id], raw_speed="30;50")
    assert "неоднозначный maxspeed: 1" in start_run(state)[1]
