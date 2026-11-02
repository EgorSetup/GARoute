"""Состояние интерфейса, которое позже свяжет виджеты с расчётом."""

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Callable

from nicegui import ui
from nicegui.elements.number import Number

from algorithm.models import AppConfig, GAConfig, ValidationReport
from config import approved_penalty_defaults, default_config, penalty_catalog, validate_config


@dataclass(slots=True)
class UIState:
    """Хранить настройки и готовность графа отдельно от элементов NiceGUI."""

    config: AppConfig = field(default_factory=default_config)
    graph_loaded: bool = False
    custom_weights: tuple[float, float, float] | None = None


@dataclass(frozen=True, slots=True)
class PendingChange:
    """Предложение изменения до подтверждения общих условий расчёта."""

    key: str
    value: object
    candidate: AppConfig | None
    errors: tuple[str, ...] = ()


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
    state.config = pending.candidate
    if pending.key == "weights" and state.config.fitness.preset == "custom":
        state.custom_weights = state.config.fitness.weights
    return True


def validate_scales(values: tuple[object, object, object]) -> ValidationReport:
    """Проверить три ручных эталона без автоподстановки чисел."""

    if len(values) != 3:
        return ValidationReport(errors=("Нужны три эталона",))
    names = ("Tref (с)", "Lref (м)", "Cref (баллы)")
    errors = tuple(
        f"{name}: требуется положительное конечное число"
        for name, value in zip(names, values)
        if (number := _numeric_input(value)) is None or number <= 0
    )
    return ValidationReport(errors=errors)


def propose_fitness_change(state: UIState, key: str, value: object) -> PendingChange:
    """Подготовить изменение Fitness, сохранив неизвестные параметры пустыми."""

    fitness = state.config.fitness
    if key == "preset":
        if value not in ("fast", "short", "easy", "custom"):
            return PendingChange(key, value, None, ("Неизвестный пресет",))
        fixed = {"fast": (1.0, 0.0, 0.0), "short": (0.0, 1.0, 0.0)}
        weights = fixed.get(value, state.custom_weights if value == "custom" else None)
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
        updated = replace(
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


def reset_approved_penalties(state: UIState) -> bool:
    """Сбросить только к утверждённым коэффициентам; пока их нет, ничего не менять."""

    defaults = approved_penalty_defaults()
    if defaults is None:
        return False
    fitness = replace(state.config.fitness, penalties=defaults)
    candidate = replace(state.config, fitness=fitness)
    if validate_config(candidate).errors:
        return False
    state.config = candidate
    return True


def build_penalty_controls(state: UIState) -> None:
    """Показать 15 штрафов и неизвестность коэффициентов и признаков."""

    with ui.expansion("Штрафы сложности · P01–P15", value=False).classes("w-full"):
        ui.label("Состав утверждён; числовые коэффициенты и часть детекторов ожидают Q04-C.").classes(
            "panel-copy"
        )
        ui.label("Неизвестный признак не означает нулевую сложность маршрута.").classes(
            "setting-error"
        )
        values = state.config.fitness.penalties
        for index, item in enumerate(penalty_catalog()):
            value = values[index] if values is not None and index < len(values) else None
            with ui.card().classes("w-full penalty-card"):
                ui.label(f"{item.id} · {item.name}").classes("penalty-title")
                ui.label(item.evidence_note).classes("panel-copy")
                if item.feature_available is False:
                    ui.label("Признак недоступен в графе уровня дорог.").classes("setting-error")
                else:
                    ui.label("Признак требует формализации и проверки.").classes("panel-copy")
                ui.number("Штраф, баллы", value=value).props("outlined dense disable").classes(
                    "setting-field"
                )
                if value is None:
                    ui.label("Коэффициент не утверждён.").classes("setting-error")
        reset = ui.button("Сбросить к утверждённым", on_click=lambda: reset_approved_penalties(state))
        reset.props("outline").classes("w-full")
        if approved_penalty_defaults() is None:
            reset.disable().tooltip("Утверждённых коэффициентов пока нет (Q04-C)")


def build_fitness_controls(state: UIState) -> None:
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
                if fitness.preset in ("fast", "short"):
                    for name, value in zip(("Время T", "Длина L", "Сложность C"), fitness.weights):
                        ui.label(f"{name}: {value * 100:g}%").classes("panel-copy")
                        ui.slider(min=0, max=100, value=value * 100).disable().classes("w-full")
                elif fitness.preset == "easy":
                    ui.label("Веса Лёгкого ещё не утверждены (Q04).").classes("panel-copy")
                    for name in ("Время T", "Длина L", "Сложность C"):
                        ui.number(name + ", %", value=None).props("outlined dense disable").classes(
                            "setting-field"
                        )
                elif fitness.preset == "custom":
                    draft = [v * 100 for v in fitness.weights] if fitness.weights else [None] * 3

                    def change_weight(index: int, value: object) -> None:
                        draft[index] = value
                        if any(v is None for v in draft):
                            error_label.set_text("Заполните все три веса; сумма должна быть 100%")
                            return
                        pending = propose_fitness_change(state, "weights", tuple(draft))
                        error_label.set_text("; ".join(pending.errors))
                        apply_setting_change(state, pending)

                    for index, name in enumerate(("Время T", "Длина L", "Сложность C")):
                        ui.number(
                            name + ", %", value=draft[index], min=0, max=100, step=1,
                            on_change=lambda event, i=index: change_weight(i, event.value),
                        ).props("outlined dense").classes("setting-field")
                else:
                    ui.label("Выберите пресет; веса пока не заданы.").classes("panel-copy")

        def select_preset(value: object) -> None:
            pending = propose_fitness_change(state, "preset", value)
            error_label.set_text("; ".join(pending.errors))
            if apply_setting_change(state, pending):
                render_weights()

        ui.select(
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


def build_normalization_controls(state: UIState) -> None:
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
                    for key, label in fields:
                        value = getattr(fitness, key)
                        ui.label(f"{label}: {value:g}" if value is not None else
                                 f"{label}: будет вычислено").classes("panel-copy")
                elif fitness.normalization_mode == "manual":
                    ui.label("Введите все три положительных эталона.").classes("panel-copy")

                    def change_reference(key: str, value: object) -> None:
                        pending = propose_fitness_change(state, key, value)
                        error_label.set_text("; ".join(pending.errors))
                        apply_setting_change(state, pending)

                    for key, label in fields:
                        ui.number(
                            label, value=getattr(fitness, key), min=0, step=1,
                            on_change=lambda event, field=key: change_reference(field, event.value),
                        ).props("outlined dense").classes("setting-field")
                else:
                    ui.label("Выберите режим нормализации.").classes("panel-copy")

        def select_mode(value: object) -> None:
            pending = propose_fitness_change(state, "normalization_mode", value)
            error_label.set_text("; ".join(pending.errors))
            if apply_setting_change(state, pending):
                render_references()

        ui.select(
            {"auto": "По начальной популяции", "manual": "Вручную"},
            label="Режим нормализации", value=state.config.fitness.normalization_mode,
            on_change=lambda event: select_mode(event.value),
        ).props("outlined dense").classes("setting-field")
        details = ui.column().classes("w-full")
        render_references()


def build_ga_controls(state: UIState, on_change: Callable[[PendingChange], None]) -> None:
    """Построить поля ГА; виджеты лишь предлагают изменения состояния."""

    widgets: dict[str, Number] = {}
    error_label = ui.label("").classes("setting-error")

    def change(key: str, value: object) -> None:
        pending = propose_setting_change(state, key, value)
        error_label.set_text("; ".join(pending.errors))
        if not pending.errors:
            on_change(pending)

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
        ui.select({"A": "A — заменить участок", "B": "B — через промежуточную точку"},
                  label="Метод мутации", value=ga.mutation_method,
                  on_change=lambda event: change("mutation_method", event.value)
                  ).props("outlined dense").classes("setting-field")
        add_number("tournament_size", "Размер турнира", ga.tournament_size,
                   minimum=1)
        add_number("initial_length_factor", "Предел начальной длины, ×L_min",
                   ga.initial_length_factor, minimum=1, step=0.1)
        add_number("min_waypoints", "Минимум промежуточных вершин",
                   ga.min_waypoints, minimum=1)
        add_number("max_waypoints", "Максимум промежуточных вершин",
                   ga.max_waypoints, minimum=1)

    with ui.expansion("Остановка и открытые параметры", value=True).classes("w-full"):
        ui.label("Стагнация означает отсутствие улучшения, а не найденный оптимум.").classes(
            "panel-copy"
        )
        ui.switch("Останавливать при стагнации").disable().tooltip(
            "Окно поколений ещё не согласовано (Q06)"
        )
        ui.number("Окно стагнации (Q06)", value=None).props(
            "outlined dense disable"
        ).classes("setting-field")
        ui.number("Сила случайных весов (Q06)", value=None).props(
            "outlined dense disable"
        ).classes("setting-field")
        ui.label("Эти значения не подставляются автоматически.").classes("panel-copy")

    def reset() -> None:
        reset_ga_defaults(state)
        defaults = state.config.ga
        for key, widget in widgets.items():
            if key in PERCENT_FIELDS:
                value = getattr(defaults, PERCENT_FIELDS[key]) * 100
            else:
                value = getattr(defaults, key)
            widget.set_value(value)
        error_label.set_text("")

    ui.button("Сбросить параметры ГА", on_click=reset, icon="restart_alt").props(
        "outline"
    ).classes("w-full")
