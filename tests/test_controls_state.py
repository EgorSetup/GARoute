"""Изменение настроек ГА без зависимости от браузерных виджетов."""

from interface.controls import (
    UIState,
    apply_setting_change,
    propose_setting_change,
    reset_ga_defaults,
)


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
