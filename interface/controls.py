"""Состояние интерфейса, которое позже свяжет виджеты с расчётом."""

from dataclasses import dataclass, field, replace
from math import isfinite
from typing import Callable

from nicegui import ui
from nicegui.elements.number import Number

from algorithm.models import AppConfig, GAConfig
from config import default_config, validate_config


@dataclass(slots=True)
class UIState:
    """Хранить настройки и готовность графа отдельно от элементов NiceGUI."""

    config: AppConfig = field(default_factory=default_config)
    graph_loaded: bool = False


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
        number = float(value)
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
    return True


def reset_ga_defaults(state: UIState) -> None:
    """Сбросить лишь принятые настройки ГА, сохранив будущие настройки Fitness."""

    state.config = replace(state.config, ga=GAConfig())


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
