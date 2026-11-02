"""Пустой каркас результатов и воспроизведения без вымышленных запусков."""

from nicegui import ui

from algorithm.genetic.initialization import PopulationResult
from algorithm.evaluation.penalties import PenaltyBreakdown, calculate_complexity
from config import penalty_catalog
from interface.controls import UIState
from visualization.charts import render_empty_charts


def _disabled_playback_button(label: str, icon: str) -> None:
    button = ui.button(label, icon=icon).props("outline dense").disable()
    button.tooltip("История поколений пока отсутствует")


def population_status_text(result: PopulationResult) -> str:
    """Объяснить P0 и отказ генератора без выдачи её за эволюцию."""

    reasons = ", ".join(f"{reason}: {count}" for reason, count in result.rejections[:4])
    suffix = f" Причины отказов: {reasons}." if reasons else ""
    if result.error is not None:
        return (f"P0 не сформирована: {result.error}; принято {result.accepted_count}, "
                f"попыток {result.attempts}.{suffix}")
    return (f"P0: {len(result.population)} маршрутов, попыток {result.attempts}. "
            f"Это проверка генератора, не эволюция.{suffix}")


def complexity_summary(breakdown: PenaltyBreakdown) -> tuple[str, ...]:
    """Сформировать объяснимый отчёт по наблюдаемой C без ложного нуля для unknown."""

    names = {item.id: item.name for item in penalty_catalog()}
    lines = [f"C_observed = {calculate_complexity(breakdown):.2f} балла · Q04 VERIFY"]
    for line in breakdown.lines:
        if line.confirmed:
            sources = ", ".join(event.object_id or event.source for event in line.confirmed[:3])
            extra = f" и ещё {len(line.confirmed) - 3}" if len(line.confirmed) > 3 else ""
            lines.append(
                f"{line.category} {names[line.category]}: {line.quantity:.2f} × "
                f"{line.coefficient:g} = {line.subtotal:.2f}; источники: {sources}{extra}"
            )
    uncertain = breakdown.uncertain
    if uncertain:
        counts = ", ".join(
            f"{line.category}: {len(line.uncertain)}" for line in breakdown.lines if line.uncertain
        )
        lines.append(f"Неполные/недоступные признаки ({len(uncertain)}): {counts}. "
                     "C_observed не является полной сложностью маршрута.")
    return tuple(lines)


def build_playback_controls(state: UIState) -> None:
    """Разместить все команды перемотки, заблокировав их без истории."""

    with ui.element("section").classes("results-section"):
        ui.label("Воспроизведение поколений").classes("panel-heading")
        if not state.playback.has_history:
            ui.label("Истории поколений нет. Перемотка станет доступна после расчёта.").classes(
                "panel-copy"
            )
        with ui.element("div").classes("playback-actions"):
            _disabled_playback_button("В начало", "first_page")
            _disabled_playback_button("Шаг назад", "skip_previous")
            _disabled_playback_button("Воспроизвести", "play_arrow")
            _disabled_playback_button("Пауза", "pause")
            _disabled_playback_button("Шаг вперёд", "skip_next")
            _disabled_playback_button("В конец", "last_page")
        ui.label("Поколение: нет данных").classes("panel-copy")
        ui.slider(min=0, max=max(1, state.playback.generation_count - 1), value=0).disable().classes(
            "w-full"
        )
        ui.select(
            {0.5: "0,5×", 1.0: "1×", 2.0: "2×"}, label="Скорость воспроизведения",
            value=None,
        ).props("outlined dense disable").classes("setting-field")


def build_results_panel(state: UIState) -> None:
    """Разместить будущие карточки, историю и четыре пустых графика."""

    ui.label("Результаты").classes("panel-heading")
    with ui.element("section").classes("results-section"):
        ui.label("Сложность начальной популяции P0").classes("panel-heading")
        route_select = ui.select({}, label="Маршрут P0", value=None).props("outlined dense").classes("setting-field")
        detail = ui.column().classes("w-full")

        def render_complexity() -> None:
            detail.clear()
            breakdowns = state.initial_complexities
            route_select.options = {index: f"Маршрут {index + 1}" for index in range(len(breakdowns))}
            if not breakdowns:
                route_select.set_value(None)
                route_select.disable()
                with detail:
                    ui.label("P0 ещё не рассчитана; сложность C не показана.").classes("panel-copy")
                return
            route_select.enable()
            selected = route_select.value if route_select.value in range(len(breakdowns)) else 0
            route_select.set_value(selected)
            with detail:
                ui.label("Предварительная оценка P0, не результат эволюции и не Fitness J.").classes("panel-copy")
                for line in complexity_summary(breakdowns[selected]):
                    ui.label(line).classes("panel-copy")

        route_select.on_value_change(lambda _: render_complexity())
        state.refresh_population_results = render_complexity
        render_complexity()
    if state.change_demo_mode:
        ui.label("ДЕМО: искусственная история только для проверки подтверждения.").classes(
            "setting-error"
        )
    with ui.element("section").classes("empty-block"):
        ui.label("Эволюция ещё не выполнялась").classes("panel-heading")
        ui.label("После запуска здесь появятся лучшие маршруты и их метрики.").classes(
            "panel-copy"
        )
    with ui.element("section").classes("results-section"):
        ui.label("Карточки лучших маршрутов").classes("panel-heading")
        ui.label("Пока нет запусков для сравнения: L, T, C и J не вычислены.").classes(
            "panel-copy"
        )
    build_playback_controls(state)
    with ui.element("section").classes("results-section"):
        ui.label("Графики поколений").classes("panel-heading")
        render_empty_charts()
