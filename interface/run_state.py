"""Подготовка независимого расчёта и приём только актуального результата."""

from copy import deepcopy
from dataclasses import dataclass, replace
from uuid import uuid4

from algorithm.genetic.evolution import RunInput, RunResult
from algorithm.genetic.initialization import PopulationResult
from algorithm.evaluation.fitness import TimeModel, unknown_required_speeds
from algorithm.evaluation.penalties import PenaltyPolicy, calculate_penalties, extract_penalty_events
from algorithm.models import AppConfig, RouteContext
from config import validate_config
from interface.controls import UIState
from visualization.animation import load_history


@dataclass(frozen=True, slots=True)
class RunHandle:
    """Идентификатор и неизменяемый для UI вход фонового расчёта."""

    run_id: str
    run_input: RunInput


RUN_COLORS = ("#d67b35", "#268b74", "#8064b5", "#cf5571", "#397db7", "#a87e27")


@dataclass(frozen=True, slots=True)
class SavedRun:
    """Отдельный успешный запуск с собственными настройками и историей."""

    run_id: str
    config: AppConfig
    seed_used: int | None
    start: object
    end: object
    outcome: RunResult
    color: str
    uncertain_penalties: int = 0
    visible: bool = True


def start_run(state: UIState) -> tuple[RunHandle | None, str | None]:
    """Проверить готовность и снять копию всех условий до запуска потока."""

    if state.active_run_id is not None:
        return None, "Расчёт уже выполняется"
    if state.initial_population_running:
        return None, "Формирование P0 ещё выполняется"
    if state.pending_change is not None:
        return None, "Завершите изменение условий перед расчётом"
    report = validate_config(state.config)
    if not report.ready_for_run:
        return None, "; ".join((*report.errors, *report.unresolved))
    population = (state.initial_population if state.config.fitness.preset == "easy"
                  else state.base_population or state.initial_population)
    if not isinstance(population, PopulationResult) or population.error is not None:
        return None, "Сначала сформируйте полную P0"
    if (state.routing_graph is None or state.normalized_roads is None
            or state.endpoint_a is None or state.endpoint_b is None):
        return None, "Дорожный граф или точки A/B не готовы"
    if len(population.population) != state.config.ga.population_size:
        return None, "Размер P0 не совпадает с N"
    if state.config.fitness.preset == "easy" and state.initial_population_preset != "easy":
        return None, "Сформируйте P0 для пресета «Лёгкий»"
    frozen_config = deepcopy(state.config)
    seed_used = state.base_seed_used if state.base_seed_used is not None else state.initial_seed_used
    if frozen_config.ga.seed is None and seed_used is not None:
        frozen_config = replace(
            frozen_config, ga=replace(frozen_config.ga, seed=seed_used),
        )
    run_input = RunInput(
        tuple(deepcopy(population.population)), frozen_config,
        RouteContext(deepcopy(state.routing_graph), state.endpoint_a, state.endpoint_b,
                     deepcopy(state.turn_rules)),
        deepcopy(dict(state.normalized_roads.edges)),
        traffic_levels=deepcopy(state.traffic_levels),
        auto_scales=deepcopy(state.base_scales),
    )
    missing_speeds = unknown_required_speeds(
        run_input.routes, TimeModel(run_input.roads, run_input.traffic_levels),
    )
    if missing_speeds:
        return None, f"Неизвестна скорость {len(missing_speeds)} доступных рёбер дороги"
    run_id = uuid4().hex
    state.active_run_id = run_id
    state.active_run_input = run_input
    state.run_status = "running"
    state.run_error = None
    if state.refresh_run is not None:
        state.refresh_run()
    return RunHandle(run_id, run_input), None


def apply_run_completion(state: UIState, run_id: str, outcome: RunResult) -> bool:
    """Сохранить успех или ошибку лишь для текущего запуска страницы."""

    if state.active_run_id != run_id:
        return False
    run_input = state.active_run_input
    state.active_run_id = None
    state.active_run_input = None
    state.run_status = outcome.status
    state.run_error = outcome.error
    state.latest_run = outcome
    if outcome.status == "completed":
        state.run_results.append(outcome)
        if isinstance(run_input, RunInput):
            append_run_result(state, run_id, run_input, outcome)
        state.generation_history = list(outcome.history)
        state.selected_run_id = run_id
        load_history(state.playback, state.generation_history)
    if state.refresh_run is not None:
        state.refresh_run()
    if outcome.status == "completed" and state.refresh_run_overlays is not None:
        state.refresh_run_overlays()
    if outcome.status == "completed" and state.refresh_playback is not None:
        state.refresh_playback()
    if outcome.status == "completed" and state.refresh_charts is not None:
        state.refresh_charts()
    if outcome.status == "completed" and state.refresh_runs is not None:
        state.refresh_runs()
    return True


def append_run_result(state: UIState, run_id: str, run_input: RunInput,
                      outcome: RunResult) -> SavedRun:
    """Добавить завершённый запуск с новым ID, включая повтор того же пресета."""

    if outcome.status != "completed":
        raise ValueError("only completed runs can be compared")
    if any(run.run_id == run_id for run in state.saved_runs):
        raise ValueError("duplicate run ID")
    uncertain = 0
    if outcome.best is not None and run_input.config.fitness.penalties is not None:
        breakdown = calculate_penalties(
            extract_penalty_events(outcome.best.route, run_input.routes,
                                   PenaltyPolicy(run_input.roads)),
            run_input.config.fitness.penalties,
        )
        uncertain = len(breakdown.uncertain)
    record = SavedRun(
        run_id, run_input.config, outcome.seed_used,
        run_input.routes.start, run_input.routes.end, outcome,
        RUN_COLORS[len(state.saved_runs) % len(RUN_COLORS)], uncertain,
    )
    state.saved_runs.append(record)
    return record


def select_run(state: UIState, run_id: str) -> bool:
    """Переключить только сохранённую историю выбранного запуска."""

    if state.active_run_id is not None:
        return False
    record = next((run for run in state.saved_runs if run.run_id == run_id), None)
    if record is None:
        return False
    state.selected_run_id = run_id
    state.latest_run = record.outcome
    state.run_status = "completed"
    state.run_error = None
    state.generation_history = list(record.outcome.history)
    load_history(state.playback, state.generation_history)
    if state.refresh_playback is not None:
        state.refresh_playback()
    if state.refresh_charts is not None:
        state.refresh_charts()
    if state.refresh_runs is not None:
        state.refresh_runs()
    if state.refresh_run is not None:
        state.refresh_run()
    return True


def set_run_visibility(state: UIState, run_id: str, visible: bool) -> bool:
    """Менять только слой результата нужного запуска, не его историю."""

    if type(visible) is not bool:
        return False
    for index, record in enumerate(state.saved_runs):
        if record.run_id == run_id:
            state.saved_runs[index] = replace(record, visible=visible)
            if state.refresh_run_overlays is not None:
                state.refresh_run_overlays()
            return True
    return False
