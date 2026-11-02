"""Проверки стартовой конфигурации приложения."""

from dataclasses import replace
from math import inf, nan

from config import default_config, validate_config, working_penalty_defaults


def test_known_defaults() -> None:
    config = default_config()

    assert config.ga.population_size == 10
    assert config.ga.elite_count == 2
    assert config.ga.crossover_probability == 0.8
    assert config.ga.mutation_probability == 0.3
    assert config.ga.max_generations == 50
    assert config.ga.max_copies == 2
    assert config.ga.attempts_per_slot == 100
    assert config.ga.tournament_size == 3
    assert config.ga.initial_length_factor == 3.0
    assert (config.ga.min_waypoints, config.ga.max_waypoints) == (1, 4)
    assert config.ga.segment_length_m == 500.0
    assert config.ga.mutation_method == "A"
    assert validate_config(config).errors == ()


def test_invalid_elitism() -> None:
    config = default_config()
    invalid = replace(config, ga=replace(config.ga, elite_count=10))

    report = validate_config(invalid)

    assert any("elite_count" in error for error in report.errors)
    assert not report.ready_for_run


def test_bad_integer_input_is_reported() -> None:
    config = default_config()
    invalid = replace(config, ga=replace(config.ga, population_size="10", min_waypoints="1"))

    report = validate_config(invalid)

    assert any("population_size" in error for error in report.errors)
    assert any("min_waypoints" in error for error in report.errors)


def test_nonfinite_values() -> None:
    config = default_config()
    invalid_ga = replace(config.ga, mutation_probability=nan)
    invalid_fitness = replace(config.fitness, time_reference_s=inf)

    report = validate_config(replace(config, ga=invalid_ga, fitness=invalid_fitness))

    assert any("mutation_probability" in error for error in report.errors)
    assert any("time_reference_s" in error for error in report.errors)
    assert not report.ready_for_run


def test_unresolved_not_defaulted() -> None:
    config = default_config()
    report = validate_config(config)

    assert config.ga.random_weight_strength == 1.0  # Q06-C VERIFY
    assert config.ga.stagnation_window == 10
    assert config.fitness.preset is None
    assert config.fitness.weights is None
    assert config.fitness.penalties == working_penalty_defaults()
    assert config.fitness.time_reference_s is None
    assert report.unresolved
    assert not report.ready_for_run
