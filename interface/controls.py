"""Состояние интерфейса, которое позже свяжет виджеты с расчётом."""

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Callable

import networkx as nx
from nicegui import ui
from nicegui.elements.number import Number

from algorithm.evaluation.penalties import PenaltyBreakdown
from algorithm.evaluation.normalization import NormalizationScales, validate_manual_scales
from algorithm.models import AppConfig, GAConfig, RouteContext, ValidationReport
from config import PRESET_WEIGHTS, default_config, penalty_catalog, validate_config, working_penalty_defaults
from data.osm_loader import RoadData
from data.restrictions import TurnRules
from data.road_graph import NormalizedRoadData, check_reachability
from visualization.animation import PlaybackState


@dataclass(slots=True)
class UIState:
    """Хранить настройки и готовность графа отдельно от элементов NiceGUI."""

    config: AppConfig = field(default_factory=default_config)
    graph_loaded: bool = False
    road_data: RoadData | None = None
    normalized_roads: NormalizedRoadData | None = None
    routing_graph: nx.MultiDiGraph | None = None
    turn_rules: TurnRules = field(default_factory=TurnRules)
    endpoint_a: object | None = None
    endpoint_b: object | None = None
    initial_population: object | None = None
    initial_complexities: tuple[PenaltyBreakdown, ...] = ()
    initial_scales: NormalizationScales | None = None
    initial_scale_error: str | None = None
    refresh_population_results: Callable[[], None] | None = None
    refresh_normalization: Callable[[], None] | None = None
    clear_population_layers: Callable[[], None] | None = None
    custom_weights: tuple[float, float, float] | None = None
    map_mode: str = "view"
    demo_mode: bool = False
    demo_point_a: tuple[float, float] | None = None
    demo_point_b: tuple[float, float] | None = None
    demo_segment_level: int = 1
    demo_map_clicks: int = 0
    demo_segment_clicks: int = 0
    playback: PlaybackState = field(default_factory=PlaybackState)
    run_results: list[object] = field(default_factory=list)
    generation_history: list[object] = field(default_factory=list)
    chart_data: dict[str, object] = field(default_factory=dict)
    selected_run_id: str | None = None
    highlighted_route_ids: set[str] = field(default_factory=set)
    pending_change: "PendingChange | None" = None
    pending_base_config: AppConfig | None = None
    change_demo_mode: bool = False


@dataclass(frozen=True, slots=True)
class PendingChange:
    """Предложение изменения до подтверждения общих условий расчёта."""

    key: str
    value: object
    candidate: AppConfig | None
    errors: tuple[str, ...] = ()


SubmitChange = Callable[[PendingChange, Callable[[], None], Callable[[], None]], None]


INTEGER_FIELDS = {
    "population_size", "elite_count", "max_generations", "max_copies",
    "attempts_per_slot", "tournament_size", "min_waypoints", "max_waypoints",
    "seed",
}
FLOAT_FIELDS = {"initial_length_factor", "segment_length_m"}
PERCENT_FIELDS = {
    "crossover_percent": "crossover_probability",
    "mutation_percent": "mutation_probability",
}


def _numeric_input(value: object) -> float | None:
    """Прочитать число из виджета без принятия NaN и бесконечности."""

    if type(value) in (int, float):
        try:
            number = float(value)
        except OverflowError:
            return None
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    return number if isfinite(number) else None


def propose_setting_change(state: UIState, key: str, value: object) -> PendingChange:
    """Проверить изменение, не затрагивая действующие настройки."""

    if key == "mutation_method":
        parsed = value
    elif key == "seed" and value in (None, ""):
        parsed = None
    elif key in INTEGER_FIELDS | FLOAT_FIELDS | PERCENT_FIELDS.keys():
        number = _numeric_input(value)
        if number is None:
            return PendingChange(key, value, None, (f"{key}: введите конечное число",))
        if key in INTEGER_FIELDS:
            if not number.is_integer():
                return PendingChange(key, value, None, (f"{key}: нужно целое число",))
            parsed = int(number)
        elif key in PERCENT_FIELDS:
            parsed = number / 100.0
        else:
            parsed = number
    else:
        return PendingChange(key, value, None, (f"{key}: неизвестная настройка",))

    field_name = PERCENT_FIELDS.get(key, key)
    candidate_ga = replace(state.config.ga, **{field_name: parsed})
    candidate = replace(state.config, ga=candidate_ga)
    errors = validate_config(candidate).errors
    return PendingChange(key, parsed, candidate if not errors else None, errors)


def apply_setting_change(state: UIState, pending: PendingChange) -> bool:
    """Применить проверенное предложение, если оно ещё допустимо."""

    if pending.candidate is None or pending.errors:
        return False
    if pending.key in ("endpoint_a", "endpoint_b"):
        setattr(state, pending.key, pending.value)
        return True
    state.config = pending.candidate
    if pending.key == "weights" and state.config.fitness.preset == "custom":
        state.custom_weights = state.config.fitness.weights
    return True


def propose_display_change(state: UIState, key: str, value: object) -> PendingChange:
    """Проверить параметр показа, который не удаляет расчётную историю."""

    if key != "playback_speed":
        return PendingChange(key, value, None, ("Неизвестный параметр показа",))
    speed = _numeric_input(value)
    if speed is None or speed <= 0:
        return PendingChange(key, value, None, ("Скорость должна быть положительной",))
    return PendingChange(key, speed, state.config)


def request_condition_change(state: UIState, change: PendingChange) -> PendingChange:
    """Применить безопасную правку сразу либо сохранить предложение до ответа."""

    if change.errors or change.candidate is None:
        return change
    if state.pending_change is not None:
        return PendingChange(change.key, change.value, None, ("Есть незавершённое изменение",))
    if change.key == "playback_speed":
        state.playback.speed = change.value
    elif change.key in ("endpoint_a", "endpoint_b"):
        old = getattr(state, change.key)
        if old != change.value and state.endpoint_a is not None and state.endpoint_b is not None:
            state.pending_change = change
            state.pending_base_config = state.config
        else:
            apply_setting_change(state, change)
    elif change.key == "preset" or change.candidate == state.config or not (
        state.run_results or state.generation_history or state.chart_data or state.initial_population
    ):
        apply_setting_change(state, change)
    else:
        state.pending_change = change
        state.pending_base_config = state.config
    return change


def select_endpoint(state: UIState, endpoint: str, node: object) -> PendingChange:
    """Предложить A/B после проверки узла и достижимости с запретами поворотов."""

    if endpoint not in ("A", "B"):
        return PendingChange("endpoint", node, None, ("Неизвестная точка маршрута",))
    graph = state.routing_graph
    if graph is None or node not in graph:
        return PendingChange(f"endpoint_{endpoint.lower()}", node, None, ("Дорожный граф не готов",))
    start = node if endpoint == "A" else state.endpoint_a
    end = node if endpoint == "B" else state.endpoint_b
    if start is not None and end is not None:
        if start == end:
            return PendingChange(f"endpoint_{endpoint.lower()}", node, None, ("A и B должны быть разными вершинами",))
        accessible = nx.subgraph_view(
            graph, filter_edge=lambda u, v, key: graph[u][v][key].get("car_access") is True,
        )
        if not nx.has_path(accessible, start, end):
            return PendingChange(f"endpoint_{endpoint.lower()}", node, None, ("Между A и B нет направленного проезда",))
        context = RouteContext(graph, start, end, state.turn_rules)
        if not check_reachability(start, end, context).reachable:
            return PendingChange(f"endpoint_{endpoint.lower()}", node, None, ("Проезд между A и B запрещён правилами поворотов",))
    return PendingChange(f"endpoint_{endpoint.lower()}", node, state.config)


def cancel_change(state: UIState) -> None:
    """Отменить предложение, не трогая действующие условия и историю."""

    state.pending_change = None
    state.pending_base_config = None


def confirm_change(state: UIState) -> UIState:
    """Атомарно применить предложение и удалить данные прежних условий."""

    change = state.pending_change
    if change is None or state.pending_base_config != state.config:
        cancel_change(state)
        return state
    if not apply_setting_change(state, change):
        cancel_change(state)
        return state
    state.run_results.clear()
    state.generation_history.clear()
    state.chart_data.clear()
    state.selected_run_id = None
    state.highlighted_route_ids.clear()
    state.playback = PlaybackState()
    state.initial_population = None
    state.initial_complexities = ()
    state.initial_scales = None
    state.initial_scale_error = None
    if state.clear_population_layers is not None:
        state.clear_population_layers()
    elif state.refresh_population_results is not None:
        state.refresh_population_results()
    if state.clear_population_layers is None and state.refresh_normalization is not None:
        state.refresh_normalization()
    cancel_change(state)
    return state


def validate_scales(values: tuple[object, object, object]) -> ValidationReport:
    """Проверить три ручных эталона без автоподстановки чисел."""

    if len(values) != 3:
        return ValidationReport(errors=("Нужны три эталона",))
    return validate_manual_scales(NormalizationScales(*values))


def propose_fitness_change(state: UIState, key: str, value: object) -> PendingChange:
    """Подготовить изменение Fitness, сохранив неизвестные параметры пустыми."""

    fitness = state.config.fitness
    if key == "preset":
        if value not in ("fast", "short", "easy", "custom"):
            return PendingChange(key, value, None, ("Неизвестный пресет",))
        weights = PRESET_WEIGHTS.get(value, state.custom_weights if value == "custom" else None)
        updated = replace(fitness, preset=value, weights=weights)
    elif key == "weights":
        if fitness.preset != "custom":
            return PendingChange(key, value, None, ("Веса меняются только в пользовательском пресете",))
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            return PendingChange(key, value, None, ("Нужны три веса",))
        numbers = tuple(_numeric_input(v) for v in value)
        if any(v is None or v < 0 for v in numbers):
            return PendingChange(key, value, None, ("Веса должны быть конечными и неотрицательными",))
        if abs(sum(numbers) - 100) > 1e-7:
            return PendingChange(key, value, None, ("Сумма весов должна быть 100%",))
        updated = replace(fitness, weights=tuple(v / 100 for v in numbers))
    elif key == "normalization_mode":
        if value not in ("auto", "manual"):
            return PendingChange(key, value, None, ("Неизвестный режим нормализации",))
        updated = fitness if value == fitness.normalization_mode else replace(
            fitness, normalization_mode=value,
            time_reference_s=None, length_reference_m=None,
            complexity_reference=None,
        )
    elif key in ("time_reference_s", "length_reference_m", "complexity_reference"):
        if fitness.normalization_mode != "manual":
            return PendingChange(key, value, None, ("Эталоны вручную задаются только в ручном режиме",))
        number = None if value in (None, "") else _numeric_input(value)
        if value not in (None, "") and (number is None or number <= 0):
            return PendingChange(key, value, None, ("Эталон должен быть положительным конечным числом",))
        updated = replace(fitness, **{key: number})
    else:
        return PendingChange(key, value, None, ("Неизвестная настройка Fitness",))

    candidate = replace(state.config, fitness=updated)
    errors = validate_config(candidate).errors
    return PendingChange(key, value, candidate if not errors else None, errors)


def reset_ga_defaults(state: UIState) -> None:
    """Сбросить лишь принятые настройки ГА, сохранив будущие настройки Fitness."""

    state.config = replace(state.config, ga=GAConfig())


def propose_penalty_change(state: UIState, index: int, value: object) -> PendingChange:
    """Проверить коэффициент, оставив подтверждение общих условий общему механизму."""

    if type(index) is not int or not 0 <= index < len(penalty_catalog()):
        return PendingChange("penalties", value, None, ("Неизвестный штраф",))
    number = _numeric_input(value)
    if number is None:
        return PendingChange("penalties", value, None, ("Введите конечное число",))
    values = list(state.config.fitness.penalties or working_penalty_defaults())
    values[index] = number
    candidate = replace(state.config, fitness=replace(state.config.fitness, penalties=tuple(values)))
    errors = validate_config(candidate).errors
    return PendingChange("penalties", number, candidate if not errors else None, errors)


def propose_penalty_reset(state: UIState) -> PendingChange:
    """Предложить возврат к рабочим Q04 VERIFY без немедленной очистки P0."""

    defaults = working_penalty_defaults()
    candidate = replace(state.config, fitness=replace(state.config.fitness, penalties=defaults))
    return PendingChange("penalties_reset", defaults, candidate)


def build_penalty_controls(state: UIState, submit: SubmitChange) -> None:
    """Показать рабочие коэффициенты, ограничения и подтверждаемое изменение."""

    with ui.expansion("Штрафы сложности · P01–P15", value=False).classes("w-full"):
        ui.label("Рабочие коэффициенты Q04 VERIFY; начисляются только подтверждённые события.").classes(
            "panel-copy"
        )
        ui.label("Неизвестный признак не означает нулевую сложность маршрута.").classes(
            "setting-error"
        )
        values = state.config.fitness.penalties
        error_label = ui.label("").classes("setting-error")
        widgets: dict[int, Number] = {}

        def change(index: int, value: object) -> None:
            pending = propose_penalty_change(state, index, value)
            error_label.set_text("; ".join(pending.errors))
            if not pending.errors:
                old = state.config.fitness.penalties[index]
                submit(pending, lambda: widgets[index].set_value(old), lambda: None)
        for index, item in enumerate(penalty_catalog()):
            value = values[index] if values is not None and index < len(values) else None
            with ui.card().classes("w-full penalty-card"):
                ui.label(f"{item.id} · {item.name}").classes("penalty-title")
                ui.label(item.evidence_note).classes("panel-copy")
                if item.feature_available is False:
                    ui.label("Признак недоступен в графе уровня дорог.").classes("setting-error")
                else:
                    ui.label("Нужен подтверждённый OSM-признак; пробелы видны в результатах.").classes("panel-copy")
                widgets[index] = ui.number(
                    f"Баллы (0–{item.maximum_points:g})", value=value,
                    min=0, max=item.maximum_points, step=0.5,
                    on_change=lambda event, i=index: change(i, event.value),
                ).props("outlined dense").classes("setting-field")

        def reset() -> None:
            pending = propose_penalty_reset(state)
            def update_widgets() -> None:
                for index, widget in widgets.items():
                    widget.set_value(state.config.fitness.penalties[index])
                error_label.set_text("")
            submit(pending, lambda: None, update_widgets)

        ui.button("Сбросить к рабочим Q04 VERIFY", on_click=reset).props("outline").classes("w-full")


def build_fitness_controls(state: UIState, submit: SubmitChange) -> None:
    """Показать пресеты и веса без выдуманных значений для открытых решений."""

    with ui.expansion("Пресет и веса Fitness", value=True).classes("w-full"):
        ui.label("Fitness / стоимость: меньше — лучше. Веса T / L / C, сумма 100%.").classes(
            "panel-copy"
        )
        error_label = ui.label("").classes("setting-error")
        details = None

        def render_weights() -> None:
            assert details is not None
            details.clear()
            fitness = state.config.fitness
            with details:
                if fitness.preset in PRESET_WEIGHTS:
                    if fitness.preset == "easy":
                        ui.label("Рабочие веса Q04 VERIFY; ограничения Лёгкого Q05 ещё открыты.").classes("setting-error")
                    for name, value in zip(("Время T", "Длина L", "Сложность C"), fitness.weights):
                        ui.label(f"{name}: {value * 100:g}%").classes("panel-copy")
                        ui.slider(min=0, max=100, value=value * 100).disable().classes("w-full")
                elif fitness.preset == "custom":
                    draft = [v * 100 for v in fitness.weights] if fitness.weights else [None] * 3

                    def change_weight(index: int, value: object) -> None:
                        draft[index] = value
                        if any(v is None for v in draft):
                            error_label.set_text("Заполните все три веса; сумма должна быть 100%")
                            return
                        pending = propose_fitness_change(state, "weights", tuple(draft))
                        error_label.set_text("; ".join(pending.errors))
                        if not pending.errors:
                            submit(pending, render_weights, lambda: None)

                    for index, name in enumerate(("Время T", "Длина L", "Сложность C")):
                        ui.number(
                            name + ", %", value=draft[index], min=0, max=100, step=1,
                            on_change=lambda event, i=index: change_weight(i, event.value),
                        ).props("outlined dense").classes("setting-field")
                else:
                    ui.label("Выберите пресет; веса пока не заданы.").classes("panel-copy")

        def select_preset(value: object) -> None:
            if value == state.config.fitness.preset:
                error_label.set_text("")
                render_weights()
                return
            pending = propose_fitness_change(state, "preset", value)
            error_label.set_text("; ".join(pending.errors))
            if not pending.errors:
                old = state.config.fitness.preset
                submit(pending, lambda: restore_preset(old), render_weights)

        def restore_preset(old: str | None) -> None:
            preset_select.set_value(old)
            render_weights()
            error_label.set_text("")

        preset_select = ui.select(
            {"fast": "Быстрый", "short": "Экономичный", "easy": "Лёгкий",
             "custom": "Пользовательский"},
            label="Пресет", value=state.config.fitness.preset,
            on_change=lambda event: select_preset(event.value),
        ).props("outlined dense").classes("setting-field")
        details = ui.column().classes("w-full")
        render_weights()
        ui.label("Лёгкий: L ≤ (1 + δL)·L_min и T ≤ (1 + δT)·T_min.").classes("panel-copy")
        ui.label("Пороги δL/δT и место фильтрации ещё не согласованы (Q05-C).").classes(
            "panel-copy"
        )
        ui.number("Превышение длины δL, %", value=None).props(
            "outlined dense disable"
        ).classes("setting-field")
        ui.number("Превышение времени δT, %", value=None).props(
            "outlined dense disable"
        ).classes("setting-field")


def build_normalization_controls(state: UIState, submit: SubmitChange) -> None:
    """Показать выбор режима и ручные эталоны с единицами."""

    with ui.expansion("Нормализация и эталоны", value=True).classes("w-full"):
        ui.label("J = wT·T/Tref + wL·L/Lref + wC·C/Cref.").classes("panel-copy")
        error_label = ui.label("").classes("setting-error")
        details = None
        fields = (
            ("time_reference_s", "Tref, секунды"),
            ("length_reference_m", "Lref, метры"),
            ("complexity_reference", "Cref, баллы"),
        )

        def render_references() -> None:
            assert details is not None
            details.clear()
            fitness = state.config.fitness
            with details:
                if fitness.normalization_mode == "auto":
                    ui.label("По P0: Tref=mean(T), Lref=mean(L), Cref=max(1, mean(C)).").classes(
                        "panel-copy"
                    )
                    scales = state.initial_scales
                    if scales is None:
                        ui.label(state.initial_scale_error or "Будет вычислено после полной P0.").classes(
                            "setting-error" if state.initial_scale_error else "panel-copy"
                        )
                    else:
                        ui.label("Предварительно по P0; эталоны этого снимка не меняются при новых поколениях.").classes("panel-copy")
                        for label, value in (("Tref, секунды", scales.time_s),
                                             ("Lref, метры", scales.length_m),
                                             ("Cref, баллы", scales.complexity)):
                            ui.label(f"{label}: {value:.2f}").classes("panel-copy")
                        if any(item.uncertain for item in state.initial_complexities):
                            ui.label("Cref основан на C_observed: часть признаков неизвестна/недоступна.").classes("setting-error")
                elif fitness.normalization_mode == "manual":
                    ui.label("Введите все три положительных эталона.").classes("panel-copy")

                    def change_reference(key: str, value: object) -> None:
                        pending = propose_fitness_change(state, key, value)
                        error_label.set_text("; ".join(pending.errors))
                        if not pending.errors:
                            old = getattr(state.config.fitness, key)
                            submit(
                                pending,
                                lambda field=key, previous=old: reference_widgets[field].set_value(previous),
                                lambda: None,
                            )

                    reference_widgets: dict[str, Number] = {}
                    for key, label in fields:
                        reference_widgets[key] = ui.number(
                            label, value=getattr(fitness, key), min=0, step=1,
                            on_change=lambda event, field=key: change_reference(field, event.value),
                        ).props("outlined dense").classes("setting-field")
                else:
                    ui.label("Выберите режим нормализации.").classes("panel-copy")

        def select_mode(value: object) -> None:
            if value == state.config.fitness.normalization_mode:
                error_label.set_text("")
                render_references()
                return
            pending = propose_fitness_change(state, "normalization_mode", value)
            error_label.set_text("; ".join(pending.errors))
            if not pending.errors:
                old = state.config.fitness.normalization_mode
                submit(pending, lambda: restore_mode(old), render_references)

        def restore_mode(old: str | None) -> None:
            mode_select.set_value(old)
            render_references()
            error_label.set_text("")

        mode_select = ui.select(
            {"auto": "По начальной популяции", "manual": "Вручную"},
            label="Режим нормализации", value=state.config.fitness.normalization_mode,
            on_change=lambda event: select_mode(event.value),
        ).props("outlined dense").classes("setting-field")
        details = ui.column().classes("w-full")
        state.refresh_normalization = render_references
        render_references()


def build_ga_controls(state: UIState, submit: SubmitChange) -> None:
    """Построить поля ГА; виджеты лишь предлагают изменения состояния."""

    widgets: dict[str, Number] = {}
    error_label = ui.label("").classes("setting-error")

    def change(key: str, value: object) -> None:
        pending = propose_setting_change(state, key, value)
        error_label.set_text("; ".join(pending.errors))
        if not pending.errors:
            field_name = PERCENT_FIELDS.get(key, key)
            old = getattr(state.config.ga, field_name)
            if key in PERCENT_FIELDS:
                old *= 100
            if key == "mutation_method":
                restore = lambda previous=old: mutation_select.set_value(previous)
            else:
                restore = lambda field=key, previous=old: widgets[field].set_value(previous)
            submit(pending, restore, lambda: None)

    def add_number(
        key: str, label: str, value: float | int | None, *,
        minimum: float | None = None, maximum: float | None = None,
        step: float = 1, help_text: str = "",
    ) -> None:
        widget = ui.number(
            label, value=value, min=minimum, max=maximum, step=step,
            on_change=lambda event, field=key: change(field, event.value),
        ).props("outlined dense").classes("setting-field")
        if help_text:
            widget.tooltip(help_text)
        widgets[key] = widget

    ga = state.config.ga
    with ui.expansion("Параметры ГА", value=True).classes("w-full"):
        ui.label("Поиск маршрутов, а не свойства дорог.").classes("panel-copy")
        add_number("population_size", "Размер популяции N", ga.population_size,
                   minimum=5, maximum=50, help_text="Диапазон интерфейса 5–50")
        add_number("elite_count", "Элитизм E", ga.elite_count, minimum=0)
        add_number("crossover_percent", "Скрещивание, %",
                   ga.crossover_probability * 100, minimum=0, maximum=100)
        add_number("mutation_percent", "Мутация, %",
                   ga.mutation_probability * 100, minimum=0, maximum=100)
        add_number("max_generations", "Переходы поколений", ga.max_generations,
                   minimum=1, help_text="P0 считается отдельно")
        add_number("max_copies", "Максимум копий особи", ga.max_copies, minimum=1)
        add_number("attempts_per_slot", "Попытки на свободное место",
                   ga.attempts_per_slot, minimum=1)
        add_number("seed", "Seed (можно оставить пустым)", ga.seed)

    with ui.expansion("Операторы и ограничения", value=True).classes("w-full"):
        mutation_select = ui.select({"A": "A — заменить участок", "B": "B — через промежуточную точку"},
                  label="Метод мутации", value=ga.mutation_method,
                  on_change=lambda event: change("mutation_method", event.value)
                  ).props("outlined dense").classes("setting-field")
        add_number("tournament_size", "Размер турнира", ga.tournament_size,
                   minimum=1)
        add_number("initial_length_factor", "Предел начальной длины, ×L_min",
                   ga.initial_length_factor, minimum=1, step=0.1)
        ui.label("L_min — кратчайший допустимый путь при текущих условиях; он нужен только для ограничения P0 и не добавляется в популяцию.").classes("panel-copy")
        add_number("min_waypoints", "Минимум промежуточных вершин",
                   ga.min_waypoints, minimum=1)
        add_number("max_waypoints", "Максимум промежуточных вершин",
                   ga.max_waypoints, minimum=1)

    with ui.expansion("Остановка и параметры поиска", value=True).classes("w-full"):
        ui.label("Стагнация означает отсутствие улучшения, а не найденный оптимум.").classes(
            "panel-copy"
        )
        ui.switch("Останавливать при стагнации").disable().tooltip(
            "Переключатель заработает после подключения полного цикла поколений"
        )
        ui.number("Окно стагнации (Q06 VERIFY)", value=ga.stagnation_window).props(
            "outlined dense disable"
        ).classes("setting-field")
        ui.number("Сила случайных весов (Q06 VERIFY)", value=ga.random_weight_strength).props(
            "outlined dense disable"
        ).classes("setting-field")
        ui.label("Рабочие значения исходного чата: 10 поколений и множитель 0,5–2; проверка продолжается.").classes("panel-copy")

    def reset() -> None:
        pending = PendingChange("ga_reset", None, replace(state.config, ga=GAConfig()))

        def update_widgets() -> None:
            defaults = state.config.ga
            for key, widget in widgets.items():
                value = getattr(defaults, PERCENT_FIELDS.get(key, key))
                widget.set_value(value * 100 if key in PERCENT_FIELDS else value)
            mutation_select.set_value(defaults.mutation_method)
            error_label.set_text("")

        submit(pending, lambda: None, update_widgets)

    ui.button("Сбросить параметры ГА", on_click=reset, icon="restart_alt").props(
        "outline"
    ).classes("w-full")
