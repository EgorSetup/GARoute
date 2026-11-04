"""Смена поколения и конечный запуск со снимком условий и полной историей."""

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import isfinite
from random import Random
from typing import Callable, Mapping, Sequence

from algorithm.evaluation.fitness import (
    TimeModel, calculate_length, calculate_travel_time, check_easy_constraints,
    compute_easy_references, evaluate_cost, resolve_preset, unknown_required_speeds,
)
from algorithm.evaluation.normalization import NormalizationScales, compute_scales, validate_manual_scales
from algorithm.evaluation.penalties import (
    PenaltyPolicy, calculate_complexity, calculate_penalties, extract_penalty_events,
)
from algorithm.genetic.crossover import crossover
from algorithm.genetic.initialization import choose_seed, production_weight_strategy, route_signature
from algorithm.genetic.mutation import MutationPolicy, mutate
from algorithm.genetic.selection import tournament_select
from algorithm.genetic.validation import validate_route
from algorithm.models import (
    AppConfig, EasyLimits, EdgeId, EligibilityReport, GAConfig, GenerationEvent,
    Individual, RoadEdge, Route, RouteContext, RouteMetrics, RouteValidationReport,
)
from config import validate_config


@dataclass(frozen=True, slots=True)
class GenerationContext:
    """Условия одного шага; стратегия оценки передаётся отдельно."""

    routes: RouteContext
    config: GAConfig
    mutation_policy: MutationPolicy
    eligible: Callable[[RouteMetrics], EligibilityReport] | None = None


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
    generation: int = 0


@dataclass(frozen=True, slots=True)
class RunInput:
    """Переданные условия; при запуске граф и дороги копируются один раз."""

    initial_population: tuple[Route, ...]
    config: AppConfig
    routes: RouteContext
    roads: Mapping[EdgeId, RoadEdge]
    traffic_levels: Mapping[EdgeId, int] = field(default_factory=dict)
    auto_scales: NormalizationScales | None = None


@dataclass(frozen=True, slots=True)
class RunResult:
    """Успешный результат или диагностическая история failed без подмены статуса."""

    status: str
    stop_reason: str
    history: tuple[GenerationSnapshot, ...]
    best: Individual | None
    scales: NormalizationScales | None
    error: str | None = None
    failed_generation: GenerationResult | None = None
    seed_used: int | None = None

    @property
    def transitions(self) -> int:
        return max(0, len(self.history) - 1)


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
    generation: int = 0,
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
    return GenerationSnapshot(tuple(copied), min(range(len(copied)), key=lambda i: copied[i].j),
                              stats, copied_events, generation)


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
    if run_context.eligible is not None:
        for individual in current:
            if individual.metrics is None:
                return empty("ineligible_current_population")
            try:
                report = run_context.eligible(individual.metrics)
            except (ValueError, TypeError, ArithmeticError):
                return empty("ineligible_current_population")
            if not isinstance(report, EligibilityReport) or not report.eligible:
                return empty("ineligible_current_population")

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
            if run_context.eligible is not None:
                if evaluated.metrics is None:
                    eligibility = EligibilityReport(False, ("missing_metrics",))
                else:
                    try:
                        eligibility = run_context.eligible(evaluated.metrics)
                    except (ValueError, TypeError, ArithmeticError):
                        eligibility = EligibilityReport(False, ("eligibility_error",))
                if not isinstance(eligibility, EligibilityReport):
                    eligibility = EligibilityReport(False, ("eligibility_error",))
                if not eligibility.eligible:
                    reasons = eligibility.reasons or ("ineligible_route",)
                    rejected.update(set(reasons))
                    events.append(GenerationEvent("admission", parents, accepted=False, reasons=tuple(reasons)))
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


def should_stop(history: Sequence[GenerationSnapshot], config: GAConfig) -> str | None:
    """Сначала жёсткий предел переходов, затем включённая стагнация по строгому J."""

    if not history:
        return None
    transitions = len(history) - 1
    if transitions >= config.max_generations:
        return "generation_limit"
    if not config.stagnation_enabled:
        return None
    best = history[0].population[history[0].best_index].j
    last_improvement = 0
    for index, snapshot in enumerate(history[1:], 1):
        current = snapshot.population[snapshot.best_index].j
        if current < best:
            best, last_improvement = current, index
    return "stagnation" if transitions - last_improvement >= config.stagnation_window else None


def run_evolution(run_input: RunInput) -> RunResult:
    """Оценить P0 и выполнить не более G_max переходов на копии условий."""

    seed_used: int | None = None

    def failed(reason: str, detail: str, history=(), best=None, scales=None,
               generation=None) -> RunResult:
        return RunResult("failed", reason, tuple(history), best, scales, detail, generation,
                         seed_used)

    report = validate_config(run_input.config)
    if not report.ready_for_run:
        return failed("invalid_input", "; ".join((*report.errors, *report.unresolved)))
    config = run_input.config
    ga = config.ga
    fitness = config.fitness
    if any(type(level) is not int or not 1 <= level <= 11
           for level in run_input.traffic_levels.values()):
        return failed("invalid_input", "invalid traffic level")
    history: list[GenerationSnapshot] = []
    best: Individual | None = None
    scales: NormalizationScales | None = None
    try:
        graph = deepcopy(run_input.routes.graph)
        roads = deepcopy(dict(run_input.roads))
        levels = dict(run_input.traffic_levels)
        closed = frozenset(run_input.routes.closed_edges) | frozenset(
            edge_id for edge_id, level in levels.items() if level == 11
        )
        routes = replace(run_input.routes, graph=graph, closed_edges=closed)
        time_model = TimeModel(roads, levels)
        penalties = PenaltyPolicy(roads)
        unknown = unknown_required_speeds(routes, time_model)
        if unknown:
            return failed("unknown_required_speeds", f"{len(unknown)} рёбер: {unknown[:5]!r}")
        initial = tuple(Route(tuple(route.edges)) for route in run_input.initial_population)
        if len(initial) != ga.population_size:
            return failed("invalid_initial_population", "P0 must contain exactly N routes")
        if max(Counter(route_signature(route) for route in initial).values()) > ga.max_copies:
            return failed("invalid_initial_population", "P0 exceeds max copies")

        def measure(route: Route) -> RouteMetrics:
            length = calculate_length(route, routes, time_model)
            time = calculate_travel_time(route, routes, time_model)
            events = extract_penalty_events(route, routes, penalties)
            breakdown = calculate_penalties(events, fitness.penalties)
            return RouteMetrics(length, time, calculate_complexity(breakdown))

        initial_metrics = tuple(measure(route) for route in initial)
        easy_references = compute_easy_references(routes, time_model) if fitness.preset == "easy" else None
        easy_limits = EasyLimits(fitness.easy_length_delta, fitness.easy_time_delta)
        eligible = (lambda metrics: check_easy_constraints(metrics, easy_references, easy_limits)) if (
            fitness.preset == "easy") else None
        if eligible is not None and any(not eligible(metrics).eligible for metrics in initial_metrics):
            return failed("invalid_initial_population", "P0 contains route outside easy limits")

        if fitness.normalization_mode == "auto":
            scales = run_input.auto_scales or compute_scales(initial_metrics)
            if validate_manual_scales(scales).errors:
                return failed("invalid_input", "invalid shared auto scales")
        else:
            scales = NormalizationScales(
                fitness.time_reference_s, fitness.length_reference_m, fitness.complexity_reference,
            )
            if validate_manual_scales(scales).errors:
                return failed("invalid_input", "invalid manual scales")
        weights = resolve_preset(fitness.preset, fitness.weights)

        def evaluate(route: Route) -> Individual:
            metrics = measure(route)
            return Individual(route, evaluate_cost(metrics, weights, scales), metrics)

        population = tuple(Individual(route, evaluate_cost(metrics, weights, scales), metrics)
                           for route, metrics in zip(initial, initial_metrics))
        events = tuple(GenerationEvent("admission", (), index, accepted=True, origin="initial")
                       for index in range(len(population)))
        history = [snapshot_generation(population, summarize_events(events), events, generation=0)]
        best = history[0].population[history[0].best_index]
        generation_context = GenerationContext(
            routes, ga, MutationPolicy(production_weight_strategy(ga.random_weight_strength)), eligible,
        )
        seed_used = choose_seed(ga.seed)
        rng = Random(seed_used)
        while (stop := should_stop(history, ga)) is None:
            step = build_next_generation(population, generation_context, rng, evaluate)
            if step.error is not None:
                return failed("generation_failed", step.error, history, best, scales, step)
            if len(step.population) != ga.population_size:
                return failed("generation_failed", "generation size differs from N", history, best, scales, step)
            snapshot = snapshot_generation(step.population, summarize_events(step.events),
                                           step.events, generation=len(history))
            history.append(snapshot)
            population = step.population
            candidate = snapshot.population[snapshot.best_index]
            if candidate.j < best.j:
                best = candidate
        return RunResult("completed", stop, tuple(history), best, scales, seed_used=seed_used)
    except (ValueError, TypeError, KeyError, OverflowError) as error:
        reason = "generation_failed" if history else "invalid_input"
        return failed(reason, str(error), history, best, scales)
