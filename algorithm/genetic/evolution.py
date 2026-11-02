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
from algorithm.models import GAConfig, GenerationEvent, Individual, Route, RouteContext, RouteMetrics, RouteValidationReport


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
    events: tuple[GenerationEvent, ...] = ()


@dataclass(frozen=True, slots=True)
class OperationStats:
    """Полные счётчики поколения, независимо от глубины визуализации."""

    crossover_attempts: int
    crossover_changes: int
    mutation_attempts: int
    mutation_changes: int
    operator_failed: int
    accepted: int
    rejected_candidates: int
    retired: int
    rejection_reasons: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class GenerationSnapshot:
    """Независимая копия принятой популяции и событий без геометрий отказов."""

    population: tuple[Individual, ...]
    best_index: int
    stats: OperationStats
    events: tuple[GenerationEvent, ...]


def summarize_events(events: Sequence[GenerationEvent]) -> OperationStats:
    """Считать каждое отклонение кандидата один раз, сохраняя все причины."""

    kinds = tuple(events)
    reasons: Counter[str] = Counter()
    for event in kinds:
        if event.kind == "admission" and event.accepted is False:
            reasons.update(set(event.reasons))
    cross = [event for event in kinds if event.kind == "crossover"]
    mutations = [event for event in kinds if event.kind == "mutation"]
    return OperationStats(
        len(cross), sum(event.changed for event in cross),
        len(mutations), sum(event.changed for event in mutations),
        sum(not event.changed for event in (*cross, *mutations)),
        sum(event.kind == "elite" or event.kind == "admission" and event.accepted is True for event in kinds),
        sum(event.kind == "admission" and event.accepted is False for event in kinds),
        sum(event.kind == "retired" for event in kinds),
        tuple(sorted(reasons.items())),
    )


def snapshot_generation(
    population: Sequence[Individual], stats: OperationStats,
    events: Sequence[GenerationEvent],
) -> GenerationSnapshot:
    """Скопировать полную принятую популяцию с J/L/T/C и метаданными событий."""

    if not population:
        raise ValueError("cannot snapshot an empty population")
    if stats.accepted != len(population):
        raise ValueError("snapshot stats do not match population size")
    copied: list[Individual] = []
    for person in population:
        metrics = person.metrics
        if metrics is None or any(
            type(value) not in (int, float) or not isfinite(value)
            for value in (person.j, metrics.length_m, metrics.time_s, metrics.complexity)
        ):
            raise ValueError("snapshot requires finite J/L/T/C")
        copied.append(Individual(
            Route(tuple(tuple(edge) for edge in person.route.edges)), person.j,
            RouteMetrics(metrics.length_m, metrics.time_s, metrics.complexity),
        ))
    copied_events = tuple(GenerationEvent(
        event.kind, tuple(event.parents), event.child_index, event.changed,
        event.accepted, tuple(event.reasons), event.origin,
    ) for event in events)
    return GenerationSnapshot(tuple(copied), min(range(len(copied)), key=lambda i: copied[i].j), stats, copied_events)


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

    elite_pairs = sorted(enumerate(current), key=lambda pair: pair[1].j)[:config.elite_count]
    accepted = [individual for _, individual in elite_pairs]
    copies = Counter(route_signature(individual.route) for individual in accepted)
    rejected: Counter[str] = Counter()
    attempts = cross_attempts = cross_changes = mutation_attempts = mutation_changes = 0
    events = [GenerationEvent("elite", (parent_index,), child_index, accepted=True, origin="elite")
              for child_index, (parent_index, _) in enumerate(elite_pairs)]

    def result(error: str | None = None) -> GenerationResult:
        return GenerationResult(
            tuple(accepted) if error is None else (), len(accepted), attempts,
            tuple(sorted(rejected.items())), cross_attempts, cross_changes,
            mutation_attempts, mutation_changes, error, tuple(events),
        )

    for _ in range(config.population_size - config.elite_count):
        for _ in range(config.attempts_per_slot):
            attempts += 1
            first = tournament_select(current, config.tournament_size, rng)
            first_index = next(index for index, person in enumerate(current) if person is first)
            parents = (first_index,)
            candidate = first.route
            crossed_changed = mutation_changed = False
            if rng.random() < config.crossover_probability:
                second = tournament_select(current, config.tournament_size, rng)
                second_index = next(index for index, person in enumerate(current) if person is second)
                parents = (first_index, second_index)
                crossed = crossover(first, second, routes, rng)
                cross_attempts += 1
                cross_changes += int(crossed.changed)
                crossed_changed = crossed.changed
                events.append(GenerationEvent(
                    "crossover", parents, changed=crossed.changed,
                    reasons=(crossed.reason,) if crossed.reason else (),
                ))
                candidate = crossed.route
            if rng.random() < config.mutation_probability:
                mutation = mutate(
                    config.mutation_method, Individual(candidate, first.j), routes, rng,
                    run_context.mutation_policy,
                )
                mutation_attempts += 1
                mutation_changes += int(mutation.changed)
                mutation_changed = mutation.changed
                events.append(GenerationEvent(
                    "mutation", parents, changed=mutation.changed,
                    reasons=(mutation.reason,) if mutation.reason else (),
                ))
                candidate = mutation.route
            report = validate_route(candidate, routes)
            if not report.valid:
                process_invalid_candidate(candidate, report)
                for code in report.codes:
                    rejected[code] += 1
                events.append(GenerationEvent("admission", parents, accepted=False, reasons=report.codes))
                continue
            signature = route_signature(candidate)
            if copies[signature] >= config.max_copies:
                rejected["duplicate_limit"] += 1
                events.append(GenerationEvent("admission", parents, accepted=False, reasons=("duplicate_limit",)))
                continue
            try:
                evaluated = evaluate(candidate)
            except (ValueError, TypeError, ArithmeticError):
                rejected["evaluation_error"] += 1
                events.append(GenerationEvent("admission", parents, accepted=False, reasons=("evaluation_error",)))
                continue
            if (
                not isinstance(evaluated, Individual)
                or evaluated.route != candidate
                or type(evaluated.j) not in (int, float)
                or not isfinite(evaluated.j)
            ):
                rejected["invalid_evaluation"] += 1
                events.append(GenerationEvent("admission", parents, accepted=False, reasons=("invalid_evaluation",)))
                continue
            origin = "mutation" if mutation_changed else "crossover" if crossed_changed else "copy"
            events.append(GenerationEvent(
                "admission", parents, len(accepted), changed=crossed_changed or mutation_changed,
                accepted=True, origin=origin,
            ))
            accepted.append(evaluated)
            copies[signature] += 1
            break
        else:
            return result("attempts_exhausted")
    elite_indices = {index for index, _ in elite_pairs}
    events.extend(GenerationEvent("retired", (index,), origin="retired")
                  for index in range(len(current)) if index not in elite_indices)
    return result()
