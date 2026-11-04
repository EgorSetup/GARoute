"""Пустой каркас результатов и воспроизведения без вымышленных запусков."""

from time import monotonic

from nicegui import ui

from algorithm.genetic.initialization import PopulationResult
from algorithm.genetic.evolution import RunResult
from algorithm.evaluation.penalties import PenaltyBreakdown, calculate_complexity
from algorithm.evaluation.fitness import evaluate_cost, resolve_preset
from algorithm.evaluation.normalization import NormalizationScales, validate_manual_scales
from algorithm.models import RouteMetrics
from config import penalty_catalog
from interface.controls import RoutePreview, UIState
from interface.run_state import SavedRun, select_run, set_run_visibility
from visualization.charts import render_charts
from visualization.animation import (
    ORIGIN_COLORS, advance, pause, play, seek, set_index, set_playback_speed,
)


def generation_failure_detail(outcome: RunResult) -> str | None:
    """Показать причину неполного поколения без выдачи его за результат."""

    step = outcome.failed_generation
    if step is None:
        return None
    reasons = ", ".join(f"{name}: {count}" for name, count in step.rejections) or "не указаны"
    return (f"Попыток: {step.attempts}; принято {step.accepted_count}; "
            f"причины отказов: {reasons}. Попробуйте другой seed или параметры ГА.")


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


def render_route_metrics(metrics: RoutePreview, breakdown: PenaltyBreakdown,
                         cost: float | None, scales: NormalizationScales | None) -> tuple[str, ...]:
    """Строки карточки P0 с единицами и явными пробелами обязательных данных."""

    lines = [f"L = {metrics.length_m:.2f} м" if metrics.length_m is not None else "L недоступна"]
    lines.append(f"T = {metrics.time_s:.2f} с" if metrics.time_s is not None else
                 f"T недоступно: {metrics.error or 'нет данных'}")
    lines.append(f"C_observed = {calculate_complexity(breakdown):.2f} балла")
    if scales is not None:
        lines.append(f"Эталоны: Tref={scales.time_s:.2f} с, "
                     f"Lref={scales.length_m:.2f} м, Cref={scales.complexity:.2f} балла")
    lines.append(f"J = {cost:.4f} (безразмерная стоимость; меньше — лучше)" if cost is not None
                 else "J недоступна")
    return tuple(lines)


def build_playback_controls(state: UIState) -> None:
    """Показать сохранённые поколения без повторного вызова расчёта."""

    with ui.element("section").classes("results-section"):
        ui.label("Воспроизведение поколений").classes("panel-heading")
        with ui.row().classes("w-full gap-1"):
            for origin, title in (("elite", "Элита"), ("crossover", "Потомок"),
                                  ("mutation", "Мутант"), ("retired", "Выбыл"),
                                  ("best", "Лучший")):
                ui.label(title).style(
                    f"background:{ORIGIN_COLORS[origin]};color:white;padding:2px 7px;"
                    "border-radius:4px;font-size:11px;"
                )
        summary = ui.label("Истории поколений нет. Перемотка станет доступна после расчёта.").classes(
            "panel-copy"
        )
        last_tick = [monotonic()]
        rendered: list[tuple[int, int] | None] = [None]

        def move(index: int) -> None:
            if not state.generation_history:
                return
            set_index(state.playback, state.generation_history, index)
            refresh()

        def start() -> None:
            play(state.playback)
            last_tick[0] = monotonic()
            refresh()

        def stop() -> None:
            pause(state.playback)
            refresh()

        def step(delta: int) -> None:
            if state.generation_history:
                move(max(0, min(len(state.generation_history) - 1,
                                state.playback.generation_index + delta)))

        def header_action(action: str) -> None:
            if action == "play":
                start()
            elif action == "back":
                step(-1)
            elif action == "next":
                step(1)

        state.playback_action = header_action

        with ui.element("div").classes("playback-actions"):
            first_button = ui.button("В начало", on_click=lambda: move(0), icon="first_page").props("outline dense")
            back_button = ui.button("Шаг назад", on_click=lambda: step(-1), icon="skip_previous").props("outline dense")
            play_button = ui.button("Воспроизвести", on_click=start, icon="play_arrow").props("outline dense")
            pause_button = ui.button("Пауза", on_click=stop, icon="pause").props("outline dense")
            next_button = ui.button("Шаг вперёд", on_click=lambda: step(1), icon="skip_next").props("outline dense")
            last_button = ui.button("В конец", on_click=lambda: move(len(state.generation_history) - 1),
                                    icon="last_page").props("outline dense")
        frame_label = ui.label("Поколение: нет данных").classes("panel-copy")
        slider = ui.slider(
            min=0, max=1, value=0,
            on_change=lambda e: move(int(e.value)) if e.value is not None
            and int(e.value) != state.playback.generation_index else None,
        ).disable().classes(
            "w-full"
        )

        def change_speed(e) -> None:
            try:
                set_playback_speed(state.playback, e.value)
            except ValueError:
                speed_input.set_value(state.playback.speed)

        speed_input = ui.number(
            "Скорость, ×", value=state.playback.speed, step=0.5,
            on_change=change_speed,
        ).props("outlined dense").classes("setting-field")

        def refresh() -> None:
            history = state.generation_history
            count = len(history)
            state.playback.generation_count = count
            playback_timer.active = state.playback.playing and count > 1
            if not count:
                summary.set_text("Истории поколений нет. Перемотка станет доступна после расчёта.")
                frame_label.set_text("Поколение: нет данных")
                for button in (first_button, back_button, play_button, pause_button,
                               next_button, last_button):
                    button.disable()
                slider.disable()
                if rendered[0] is not None and state.show_generation_frame is not None:
                    state.show_generation_frame(None)
                rendered[0] = None
                if state.refresh_playback_header is not None:
                    state.refresh_playback_header()
                return
            index = state.playback.generation_index
            frame = seek(history, index)
            snapshot = frame.snapshot
            best = snapshot.population[snapshot.best_index]
            summary.set_text(
                f"Поколение {snapshot.generation} · особей {len(snapshot.population)} · "
                f"лучший J={best.j:.4f} · принято {snapshot.stats.accepted}, "
                f"отклонено {snapshot.stats.rejected_candidates}"
            )
            frame_label.set_text(f"Поколение: {index} из {count - 1}")
            slider.props(f"max={max(1, count - 1)}")
            if slider.value != index:
                slider.set_value(index)
            slider.enable()
            for button in (first_button, back_button, play_button, pause_button,
                           next_button, last_button):
                button.enable()
            if index == 0:
                first_button.disable()
                back_button.disable()
            if index == count - 1:
                next_button.disable()
                last_button.disable()
            if state.playback.playing:
                play_button.disable()
            else:
                pause_button.disable()
            if count <= 1:
                play_button.disable()
            key = (state.selected_run_id or id(history), index)
            if key != rendered[0]:
                if state.show_generation_frame is not None:
                    state.show_generation_frame(frame)
                rendered[0] = key
            if state.refresh_playback_header is not None:
                state.refresh_playback_header()

        def on_tick() -> None:
            now = monotonic()
            elapsed = now - last_tick[0]
            last_tick[0] = now
            if advance(state.playback, elapsed):
                refresh()

        state.refresh_playback = refresh
        playback_timer = ui.timer(0.1, on_tick, active=False)
        refresh()


def build_results_panel(state: UIState) -> None:
    """Показать P0, независимые запуски, выбранную историю и четыре графика."""

    ui.label("Результаты").classes("panel-heading")
    with ui.element("section").classes("results-section"):
        ui.label("Метрики начальной популяции P0").classes("panel-heading")
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
                ui.label("Предварительная оценка P0, не результат эволюции.").classes("panel-copy")
                if state.initial_population_preset == "easy":
                    ui.label("P0 «Лёгкого»: пригодные маршруты общей базы сохранены, остальные места дозаполнены; автоэталоны серии рассчитаны по общей базе. Q07 VERIFY.").classes("panel-copy")
                if state.initial_population_preset not in (None, state.config.fitness.preset):
                    ui.label("Показана P0 предыдущего пресета. Для нового пресета нажмите «Показать P0»; расчёт обычного пресета использует общую базовую P0.").classes("setting-error")
                if selected < len(state.initial_previews):
                    preview = state.initial_previews[selected]
                    fitness = state.config.fitness
                    scales = None
                    if fitness.normalization_mode == "auto":
                        scales = state.initial_scales
                    elif fitness.normalization_mode == "manual":
                        values = (fitness.time_reference_s, fitness.length_reference_m,
                                  fitness.complexity_reference)
                        if all(value is not None for value in values):
                            candidate = NormalizationScales(*values)
                            if not validate_manual_scales(candidate).errors:
                                scales = candidate
                    cost = None
                    reason = None
                    if preview.time_s is None:
                        reason = preview.error or "неизвестно время маршрута"
                    elif state.easy_references is None:
                        reason = state.easy_reference_error or "обязательные скорости области не проверены"
                    elif scales is None:
                        reason = state.initial_scale_error or "выберите режим и задайте эталоны"
                    elif fitness.preset is None:
                        reason = "выберите пресет"
                    elif fitness.preset == "easy" and state.initial_population_preset != "easy":
                        reason = "базовая P0 ещё не отфильтрована для «Лёгкого»"
                    else:
                        try:
                            metrics = RouteMetrics(preview.length_m, preview.time_s,
                                                   breakdowns[selected].total)
                            cost = evaluate_cost(metrics, resolve_preset(fitness.preset, fitness.weights), scales)
                        except ValueError as error:
                            reason = str(error)
                    for line in render_route_metrics(preview, breakdowns[selected], cost, scales):
                        ui.label(line).classes("panel-copy")
                    if reason is not None:
                        ui.label(f"J недоступна: {reason}").classes("setting-error")
                for line in complexity_summary(breakdowns[selected])[1:]:
                    ui.label(line).classes("panel-copy")

        route_select.on_value_change(lambda _: render_complexity())
        state.refresh_population_results = render_complexity
        render_complexity()
    if state.change_demo_mode:
        ui.label("ДЕМО: искусственная история только для проверки подтверждения.").classes(
            "setting-error"
        )
    run_panel = ui.column().classes("empty-block w-full")

    def render_run() -> None:
        run_panel.clear()
        with run_panel:
            outcome = state.latest_run
            if state.run_status == "running":
                ui.label("Выполняется эволюция…").classes("panel-heading")
            elif isinstance(outcome, RunResult) and outcome.status == "completed":
                ui.label("Эволюция завершена").classes("panel-heading")
                ui.label(f"Поколений: {len(outcome.history)} (P0 + {outcome.transitions} переходов); "
                         f"остановка: {outcome.stop_reason}").classes("panel-copy")
                if outcome.seed_used is not None:
                    ui.label(f"Seed для повтора: {outcome.seed_used}").classes("panel-copy")
                if outcome.best is not None and outcome.best.metrics is not None:
                    metrics = outcome.best.metrics
                    ui.label(f"Лучший: J={outcome.best.j:.4f}, L={metrics.length_m:.2f} м, "
                             f"T={metrics.time_s:.2f} с, C={metrics.complexity:.2f} балла").classes("panel-copy")
            elif isinstance(outcome, RunResult):
                ui.label("Расчёт не завершён").classes("panel-heading")
                ui.label(f"{outcome.stop_reason}: {outcome.error or 'нет подробностей'}").classes("setting-error")
                detail = generation_failure_detail(outcome)
                if detail is not None:
                    ui.label(detail).classes("setting-error")
                if outcome.history:
                    ui.label(f"Сохранено полных поколений для диагностики: {len(outcome.history)}").classes("panel-copy")
            else:
                ui.label("Эволюция ещё не выполнялась").classes("panel-heading")
                ui.label("После запуска здесь появятся лучшие маршруты и их метрики.").classes("panel-copy")

    previous_refresh = state.refresh_run
    state.refresh_run = lambda: (previous_refresh() if previous_refresh else None, render_run())
    render_run()
    with ui.element("section").classes("results-section"):
        ui.label("Карточки лучших маршрутов").classes("panel-heading")
        ui.label("Для разных пресетов сравнивайте L/T/C при общих условиях. J имеет разные веса и не образует общего рейтинга.").classes("panel-copy")
        cards = ui.column().classes("w-full")

        def render_saved_runs() -> None:
            cards.clear()
            with cards:
                if not state.saved_runs:
                    ui.label("Пока нет завершённых запусков для сравнения.").classes("panel-copy")
                    return
                names = {"fast": "Быстрый", "short": "Экономичный",
                         "easy": "Лёгкий", "custom": "Пользовательский"}
                for index, record in enumerate(state.saved_runs, start=1):
                    if not isinstance(record, SavedRun):
                        continue
                    with ui.card().classes("w-full penalty-card").style(
                        f"border-left:5px solid {record.color}"
                    ):
                        preset = names.get(record.config.fitness.preset, "Неизвестный")
                        ui.label(f"Запуск {index} · {preset} · {record.run_id[:8]}").classes("penalty-title")
                        best = record.outcome.best
                        if best is not None and best.metrics is not None:
                            metrics = best.metrics
                            ui.label(f"L={metrics.length_m:.2f} м · T={metrics.time_s:.2f} с · "
                                     f"C_observed={metrics.complexity:.2f} балла · "
                                     f"J={best.j:.4f} (только внутри пресета)").classes("panel-copy")
                        ui.label(f"Seed: {record.seed_used}; остановка: {record.outcome.stop_reason}; "
                                 f"поколений: {len(record.outcome.history)}").classes("panel-copy")
                        if record.config.fitness.modeled_speeds_enabled:
                            configured = ", ".join(
                                f"{name}={speed:g} км/ч" for name, speed in record.config.fitness.modeled_speeds_kph
                            )
                            usage = ", ".join(
                                f"{name}: {count} рёбер" for name, count in record.modeled_speed_usage
                            ) or "0 рёбер"
                            ui.label(
                                f"Учебная модель скорости: {configured}. Применение в лучшем маршруте: {usage}. "
                                "У замещаемых рёбер исходный maxspeed отсутствует; T зависит от введённых значений."
                            ).classes("setting-error")
                        if record.config.fitness.weights is not None:
                            weights = record.config.fitness.weights
                            ui.label(f"Веса T/L/C: {weights[0]*100:g}/{weights[1]*100:g}/{weights[2]*100:g}%").classes("panel-copy")
                        if record.outcome.scales is not None:
                            scales = record.outcome.scales
                            ui.label(f"Эталоны: T={scales.time_s:.2f} с, "
                                     f"L={scales.length_m:.2f} м, C={scales.complexity:.2f}").classes("panel-copy")
                        if record.uncertain_penalties:
                            ui.label(f"Неподтверждённых штрафных признаков: {record.uncertain_penalties}; "
                                     "C_observed не включает их как нули.").classes("setting-error")
                        with ui.row().classes("items-center"):
                            ui.button("Выбрать историю", on_click=lambda key=record.run_id: select_run(state, key)).props("flat dense")
                            ui.switch("Показать в сравнении", value=record.visible,
                                      on_change=lambda e, key=record.run_id: set_run_visibility(state, key, e.value))
                        if state.selected_run_id == record.run_id:
                            ui.label("Выбрана история этого запуска").classes("panel-copy")

        state.refresh_runs = render_saved_runs
        render_saved_runs()
    build_playback_controls(state)
    with ui.element("section").classes("results-section"):
        ui.label("Графики поколений").classes("panel-heading")
        state.refresh_charts = render_charts(lambda: state.generation_history)
