"""Одна смена поколения без производственной оценки и без ремонта."""

from collections import Counter
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Callable, Sequence

from algorithm.genetic.crossover import crossover
from algorithm.genetic.initialization import route_signature
from algorithm.genetic.mutation import MutationPolicy, mutate
from algorithm.genetic.selection import tournament_select
from algorithm.genetic.validation import validate_route
from algorithm.models import GAConfig, Individual, Route, RouteContext, RouteValidationReport


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Условия одного шага; стратегия оценки передаётся отдельно."""

    routes: RouteContext
    config: GAConfig
    mutation_policy: MutationPolicy


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Опубликовано только полное поколение; частичный прогресс — диагностика."""

    population: tuple[Individual, ...]
    accepted_count: int
    attempts: int
    rejections: tuple[tuple[str, int], ...]
    crossover_attempts: int
    crossover_changes: int
    mutation_attempts: int
    mutation_changes: int
    error: str | None = None


def process_invalid_candidate(candidate: Route, report: RouteValidationReport) -> None:
    """Точка расширения: сейчас недопустимый кандидат только отбрасывается."""

    return None


def build_next_generation(
    current: Sequence[Individual],
    run_context: GenerationContext,
    rng: Random,
    evaluate: Callable[[Route], Individual],
) -> GenerationResult:
    """Сохранить элиту и заполнить остальные места за конечное число попыток."""

    config = run_context.config
    routes = run_context.routes
    empty = lambda error: GenerationResult((), 0, 0, (), 0, 0, 0, 0, error)
    if (
        len(current) != config.population_size
        or type(config.population_size) is not int or config.population_size < 1
        or type(config.elite_count) is not int or not 0 <= config.elite_count <= config.population_size
        or type(config.max_copies) is not int or config.max_copies < 1
        or type(config.attempts_per_slot) is not int or config.attempts_per_slot < 1
        or type(config.tournament_size) is not int or not 1 <= config.tournament_size <= config.population_size
        or config.mutation_method not in ("A", "B")
        or any(type(p) not in (int, float) or not isfinite(p) or not 0 <= p <= 1 for p in (
            config.crossover_probability, config.mutation_probability,
        ))
    ):
        return empty("invalid_generation_config")
    if any(
        not validate_route(individual.route, routes).valid
        or type(individual.j) not in (int, float)
        or not isfinite(individual.j)
        for individual in current
    ):
        return empty("invalid_current_population")
    if max(Counter(route_signature(individual.route) for individual in current).values()) > config.max_copies:
        return empty("invalid_current_copies")

    elites = sorted(current, key=lambda individual: individual.j)[:config.elite_count]
    accepted = list(elites)
    copies = Counter(route_signature(individual.route) for individual in elites)
    rejected: Counter[str] = Counter()
    attempts = cross_attempts = cross_changes = mutation_attempts = mutation_changes = 0

    def result(error: str | None = None) -> GenerationResult:
        return GenerationResult(
            tuple(accepted) if error is None else (), len(accepted), attempts,
            tuple(sorted(rejected.items())), cross_attempts, cross_changes,
            mutation_attempts, mutation_changes, error,
        )

    for _ in range(config.population_size - config.elite_count):
        for _ in range(config.attempts_per_slot):
            attempts += 1
            first = tournament_select(current, config.tournament_size, rng)
            candidate = first.route
            if rng.random() < config.crossover_probability:
                second = tournament_select(current, config.tournament_size, rng)
                crossed = crossover(first, second, routes, rng)
                cross_attempts += 1
                cross_changes += int(crossed.changed)
                if crossed.rejected_count:
                    rejected["crossover_invalid"] += crossed.rejected_count
                candidate = crossed.route
            if rng.random() < config.mutation_probability:
                mutation = mutate(
                    config.mutation_method, Individual(candidate, first.j), routes, rng,
                    run_context.mutation_policy,
                )
                mutation_attempts += 1
                mutation_changes += int(mutation.changed)
                if mutation.rejected_count:
                    rejected["mutation_invalid"] += mutation.rejected_count
                candidate = mutation.route
            report = validate_route(candidate, routes)
            if not report.valid:
                process_invalid_candidate(candidate, report)
                for code in report.codes:
                    rejected[code] += 1
                continue
            signature = route_signature(candidate)
            if copies[signature] >= config.max_copies:
                rejected["duplicate_limit"] += 1
                continue
            try:
                evaluated = evaluate(candidate)
            except (ValueError, TypeError, ArithmeticError):
                rejected["evaluation_error"] += 1
                continue
            if (
                not isinstance(evaluated, Individual)
                or evaluated.route != candidate
                or type(evaluated.j) not in (int, float)
                or not isfinite(evaluated.j)
            ):
                rejected["invalid_evaluation"] += 1
                continue
            accepted.append(evaluated)
            copies[signature] += 1
            break
        else:
            return result("attempts_exhausted")
    return result()
