"""Одна попытка построения особи через случайные промежуточные вершины."""

from dataclasses import dataclass
from collections import Counter
from math import fsum, isfinite
from random import Random
from secrets import randbits
from typing import Callable, Hashable, Mapping, Sequence

from algorithm.genetic.validation import validate_route
from algorithm.models import EdgeId, EligibilityReport, GAConfig, Route, RouteContext, ValidationReport
from data.road_graph import find_connector


WeightStrategy = Callable[[EdgeId, Mapping[str, object], Random], float]


def choose_seed(configured: int | None) -> int:
    """Вернуть заданный seed либо создать и сохранить число для точного повтора."""

    if configured is not None:
        if type(configured) is not int:
            raise ValueError("seed must be an integer or empty")
        return configured
    return randbits(64)


def production_weight_strategy(strength: float) -> WeightStrategy:
    """Q06-C VERIFY: положительный множитель 2^U(-s,s) от длины ребра."""

    if type(strength) not in (int, float) or not isfinite(strength) or strength <= 0:
        raise ValueError("random weight strength must be positive and finite")

    def weight(edge: EdgeId, attributes: Mapping[str, object], rng: Random) -> float:
        try:
            length = float(attributes["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("edge length must be known in metres") from error
        if not isfinite(length) or length <= 0:
            raise ValueError("edge length must be positive and finite")
        result = length * 2 ** rng.uniform(-strength, strength)
        if not isfinite(result) or result <= 0:
            raise ValueError("edge weight must be positive and finite")
        return result

    return weight


@dataclass(frozen=True, slots=True)
class CandidateOutcome:
    """Результат одной конечной попытки без подмены неудачи эталоном."""

    route: Route | None
    reason: str | None
    waypoints: tuple[Hashable, ...]


@dataclass(frozen=True, slots=True)
class InitialLengthPolicy:
    """Метры эталона и утверждённый множитель предела P0."""

    minimum_length_m: float | None
    factor: float = 3.0


@dataclass(frozen=True, slots=True)
class PopulationResult:
    """Только полная популяция; частичный прогресс доступен как диагностика."""

    population: tuple[Route, ...]
    accepted_count: int
    attempts: int
    rejections: tuple[tuple[str, int], ...]
    error: str | None = None


def route_signature(route: Route) -> tuple[EdgeId, ...]:
    """Сравнить всю упорядоченную хромосому, включая ключи параллельных рёбер."""

    return route.edges


def minimum_valid_length(context: RouteContext) -> float | None:
    """Найти только длину кратчайшего допустимого пути, не особь P0."""

    def by_length(edge: EdgeId, attributes: Mapping[str, object]) -> float:
        try:
            return float(attributes["length"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("edge length must be known in metres") from error

    route = find_connector(context.start, context.end, context, Random(0), weight=by_length, complete=True)
    if route is None:
        return None
    return fsum(float(context.graph[u][v][key]["length"]) for u, v, key in route.edges)


def check_initial_length(route_length_m: float, policy: InitialLengthPolicy) -> ValidationReport:
    """Проверить L≤k_init·L_min в метрах, не меняя маршрут."""

    base = policy.minimum_length_m
    factor = policy.factor
    if type(base) not in (int, float) or not isfinite(base) or base <= 0:
        return ValidationReport(errors=("L_min must be positive and finite",))
    if type(factor) not in (int, float) or not isfinite(factor) or factor <= 0:
        return ValidationReport(errors=("initial length factor must be positive and finite",))
    if type(route_length_m) not in (int, float) or not isfinite(route_length_m) or route_length_m <= 0:
        return ValidationReport(errors=("route length must be positive and finite",))
    limit = factor * base
    if not isfinite(limit):
        return ValidationReport(errors=("initial length limit is not finite",))
    if route_length_m > limit:
        return ValidationReport(errors=("initial route exceeds k_init × L_min",))
    return ValidationReport()


def generate_candidate(
    context: RouteContext,
    config: GAConfig,
    rng: Random,
    weight_strategy: WeightStrategy,
) -> CandidateOutcome:
    """Соединить 1–4 случайные вершины и проверить готовую особь."""

    if config.min_waypoints < 1 or config.max_waypoints > 4 or config.min_waypoints > config.max_waypoints:
        return CandidateOutcome(None, "invalid_waypoint_range", ())
    if context.start == context.end:
        return CandidateOutcome(None, "same_endpoints", ())
    available = [node for node in context.graph if node not in (context.start, context.end)]
    if len(available) < config.min_waypoints:
        return CandidateOutcome(None, "not_enough_waypoints", ())

    count = rng.randint(config.min_waypoints, min(config.max_waypoints, len(available)))
    waypoints = tuple(rng.sample(available, count))
    targets = (context.start, *waypoints, context.end)
    edges: list[EdgeId] = []

    for start, end in zip(targets, targets[1:]):
        search_weights: dict[EdgeId, float] = {}

        def weight(edge: EdgeId, attributes: Mapping[str, object]) -> float:
            if edge not in search_weights:
                search_weights[edge] = weight_strategy(edge, attributes, rng)
            return search_weights[edge]

        connector = find_connector(
            start,
            end,
            context,
            rng,
            incoming=edges[-1] if edges else None,
            weight=weight,
        )
        if connector is None:
            return CandidateOutcome(None, "connector_unavailable", waypoints)
        edges.extend(connector.edges)

    candidate = Route(tuple(edges))
    report = validate_route(candidate, context)
    if not report.valid:
        return CandidateOutcome(None, "invalid_route:" + ",".join(report.codes), waypoints)
    return CandidateOutcome(candidate, None, waypoints)


def generate_initial_population(
    context: RouteContext,
    config: GAConfig,
    rng: Random,
    weight_strategy: WeightStrategy,
    eligible: Callable[[Route], EligibilityReport] | None = None,
    initial_routes: Sequence[Route] = (),
) -> PopulationResult:
    """Сохранить пригодные базовые маршруты и заполнить N конечными попытками."""

    if any(type(value) is not int or value < 1 for value in (
        config.population_size, config.max_copies, config.attempts_per_slot,
    )):
        return PopulationResult((), 0, 0, (), "invalid_population_config")
    try:
        minimum = minimum_valid_length(context)
    except (ValueError, TypeError, KeyError, OverflowError):
        return PopulationResult((), 0, 0, (), "invalid_reference")
    if minimum is None:
        return PopulationResult((), 0, 0, (), "reference_unavailable")
    policy = InitialLengthPolicy(minimum, config.initial_length_factor)
    if not check_initial_length(minimum, policy).is_valid:
        return PopulationResult((), 0, 0, (), "invalid_reference")

    accepted: list[Route] = []
    copies: Counter[tuple[EdgeId, ...]] = Counter()
    rejections: Counter[str] = Counter()
    attempts = 0
    for route in initial_routes:
        if len(accepted) >= config.population_size:
            break
        report = validate_route(route, context)
        if not report.valid:
            rejections["invalid_base_route"] += 1
            continue
        try:
            length = fsum(float(context.graph[u][v][key]["length"]) for u, v, key in route.edges)
        except (KeyError, TypeError, ValueError, OverflowError):
            rejections["invalid_base_length"] += 1
            continue
        if not check_initial_length(length, policy).is_valid:
            rejections["base_length_limit"] += 1
            continue
        if eligible is not None:
            try:
                eligibility = eligible(route)
            except (ValueError, TypeError, ArithmeticError, KeyError):
                rejections["base_eligibility_error"] += 1
                continue
            if not isinstance(eligibility, EligibilityReport):
                rejections["base_eligibility_error"] += 1
                continue
            if not eligibility.eligible:
                rejections.update(eligibility.reasons or ("ineligible_base_route",))
                continue
        signature = route_signature(route)
        if copies[signature] >= config.max_copies:
            rejections["base_duplicate_limit"] += 1
            continue
        accepted.append(route)
        copies[signature] += 1
    for _ in range(config.population_size - len(accepted)):
        for _ in range(config.attempts_per_slot):
            attempts += 1
            candidate = generate_candidate(context, config, rng, weight_strategy)
            if candidate.route is None:
                rejections[candidate.reason or "candidate_unavailable"] += 1
                continue
            route = candidate.route
            try:
                length = fsum(float(context.graph[u][v][key]["length"]) for u, v, key in route.edges)
            except (KeyError, TypeError, ValueError, OverflowError):
                rejections["invalid_route_length"] += 1
                continue
            length_report = check_initial_length(length, policy)
            if not length_report.is_valid:
                rejections["length_limit"] += 1
                continue
            if eligible is not None:
                try:
                    eligibility = eligible(route)
                except (ValueError, TypeError, ArithmeticError, KeyError):
                    rejections["eligibility_error"] += 1
                    continue
                if not isinstance(eligibility, EligibilityReport):
                    rejections["eligibility_error"] += 1
                    continue
                if not eligibility.eligible:
                    rejections.update(eligibility.reasons or ("ineligible_route",))
                    continue
            signature = route_signature(route)
            if copies[signature] >= config.max_copies:
                rejections["duplicate_limit"] += 1
                continue
            accepted.append(route)
            copies[signature] += 1
            break
        else:
            return PopulationResult((), len(accepted), attempts, tuple(sorted(rejections.items())), "attempts_exhausted")
    return PopulationResult(tuple(accepted), len(accepted), attempts, tuple(sorted(rejections.items())))
