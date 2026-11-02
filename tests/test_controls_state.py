"""Изменение настроек ГА без зависимости от браузерных виджетов."""

from interface.controls import (
    UIState,
    apply_setting_change,
    propose_fitness_change,
    propose_setting_change,
    reset_ga_defaults,
    reset_approved_penalties,
    validate_scales,
)
from config import penalty_catalog


def test_config_roundtrip() -> None:
    state = UIState()
    pending = propose_setting_change(state, "population_size", 16)

    assert pending.errors == ()
    assert state.config.ga.population_size == 10  # Предложение ещё не применено.
    apply_setting_change(state, pending)
    assert state.config.ga.population_size == 16

    probability = propose_setting_change(state, "crossover_percent", 65)
    apply_setting_change(state, probability)
    assert state.config.ga.crossover_probability == 0.65


def test_reset_known_defaults() -> None:
    state = UIState()
    apply_setting_change(state, propose_setting_change(state, "elite_count", 1))

    reset_ga_defaults(state)

    assert state.config.ga.elite_count == 2
    assert state.config.ga.random_weight_strength is None
    assert state.config.ga.stagnation_window is None


def test_mutation_choice() -> None:
    state = UIState()
    apply_setting_change(state, propose_setting_change(state, "mutation_method", "B"))
    assert state.config.ga.mutation_method == "B"

    invalid = propose_setting_change(state, "mutation_method", "C")
    assert invalid.errors
    assert state.config.ga.mutation_method == "B"


def test_bad_input_does_not_change_state() -> None:
    state = UIState()
    invalid = propose_setting_change(state, "elite_count", 10)

    assert invalid.errors
    assert not apply_setting_change(state, invalid)
    assert state.config.ga.elite_count == 2


def test_preset_locks_weights() -> None:
    state = UIState()
    assert apply_setting_change(state, propose_fitness_change(state, "preset", "fast"))
    assert state.config.fitness.weights == (1.0, 0.0, 0.0)
    assert propose_fitness_change(state, "weight_time", 40).errors
    assert apply_setting_change(state, propose_fitness_change(state, "preset", "easy"))
    assert state.config.fitness.weights is None
    assert state.config.fitness.easy_length_delta is None
    assert state.config.fitness.easy_time_delta is None

    assert apply_setting_change(state, propose_fitness_change(state, "preset", "custom"))
    assert apply_setting_change(state, propose_fitness_change(state, "weights", (20, 30, 50)))
    assert state.config.fitness.weights == (0.2, 0.3, 0.5)
    assert propose_fitness_change(state, "weights", (20, 30, 40)).errors
    assert state.config.fitness.weights == (0.2, 0.3, 0.5)


def test_manual_scales_positive() -> None:
    state = UIState()
    assert apply_setting_change(state, propose_fitness_change(state, "normalization_mode", "manual"))
    assert not validate_scales((600, 3000, 20)).errors
    for values in ((0, 3000, 20), (float("inf"), 3000, 20), (600, -1, 20)):
        assert validate_scales(values).errors
    assert propose_fitness_change(state, "time_reference_s", 0).errors
    assert apply_setting_change(state, propose_fitness_change(state, "time_reference_s", 600))
    assert state.config.fitness.time_reference_s == 600
    assert state.config.fitness.length_reference_m is None


def test_auto_scales_pending() -> None:
    state = UIState()
    assert apply_setting_change(state, propose_fitness_change(state, "normalization_mode", "auto"))
    assert state.config.fitness.time_reference_s is None
    assert state.config.fitness.length_reference_m is None
    assert state.config.fitness.complexity_reference is None
    assert propose_fitness_change(state, "time_reference_s", 600).errors


def test_exact_fifteen_penalties() -> None:
    catalog = penalty_catalog()
    assert tuple(item.id for item in catalog) == tuple(f"P{i:02d}" for i in range(1, 16))
    assert len({item.name for item in catalog}) == 15
    assert all(item.evidence_note for item in catalog)
    assert catalog[3].name == "Смена полосы"
    assert catalog[3].feature_available is False


def test_unknown_penalty_not_zero() -> None:
    state = UIState()
    assert state.config.fitness.penalties is None
    assert not reset_approved_penalties(state)
    assert state.config.fitness.penalties is None
    assert not state.config.fitness.penalties == (0.0,) * 15
