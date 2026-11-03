"""Подготовка независимого расчёта и приём только актуального результата."""

from copy import deepcopy
from dataclasses import dataclass
from uuid import uuid4

from algorithm.genetic.evolution import RunInput, RunResult
from algorithm.genetic.initialization import PopulationResult
from algorithm.evaluation.fitness import TimeModel, unknown_required_speeds
from algorithm.models import RouteContext
from config import validate_config
from interface.controls import UIState


@dataclass(frozen=True, slots=True)
class RunHandle:
    """Идентификатор и неизменяемый для UI вход фонового расчёта."""

    run_id: str
    run_input: RunInput


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
    population = state.initial_population
    if not isinstance(population, PopulationResult) or population.error is not None:
        return None, "Сначала сформируйте полную P0"
    if (state.routing_graph is None or state.normalized_roads is None
            or state.endpoint_a is None or state.endpoint_b is None):
        return None, "Дорожный граф или точки A/B не готовы"
    if len(population.population) != state.config.ga.population_size:
        return None, "Размер P0 не совпадает с N"
    if state.config.fitness.preset == "easy" and state.initial_population_preset != "easy":
        return None, "Сформируйте P0 для пресета «Лёгкий»"
    run_input = RunInput(
        tuple(deepcopy(population.population)), deepcopy(state.config),
        RouteContext(deepcopy(state.routing_graph), state.endpoint_a, state.endpoint_b,
                     deepcopy(state.turn_rules)),
        deepcopy(dict(state.normalized_roads.edges)),
    )
    missing_speeds = unknown_required_speeds(run_input.routes, TimeModel(run_input.roads))
    if missing_speeds:
        return None, f"Неизвестна скорость {len(missing_speeds)} доступных рёбер дороги"
    run_id = uuid4().hex
    state.active_run_id = run_id
    state.run_status = "running"
    state.run_error = None
    if state.refresh_run is not None:
        state.refresh_run()
    return RunHandle(run_id, run_input), None


def apply_run_completion(state: UIState, run_id: str, outcome: RunResult) -> bool:
    """Сохранить успех или ошибку лишь для текущего запуска страницы."""

    if state.active_run_id != run_id:
        return False
    state.active_run_id = None
    state.run_status = outcome.status
    state.run_error = outcome.error
    state.latest_run = outcome
    if outcome.status == "completed":
        state.run_results.append(outcome)
        state.generation_history = list(outcome.history)
        state.selected_run_id = run_id
    if state.refresh_run is not None:
        state.refresh_run()
    return True
