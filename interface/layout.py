"""Основная страница GARoute и неблокирующий запуск расчёта."""

import asyncio
from dataclasses import dataclass

from nicegui import ui
from nicegui.elements.tabs import TabPanels, Tabs
from nicegui.elements.input import Input

from interface.controls import (
    UIState, PendingChange, build_fitness_controls, build_ga_controls,
    build_normalization_controls, build_penalty_controls, cancel_change,
    confirm_change, request_condition_change,
)
from interface.map_view import TRAFFIC_COLORS, MapHandle, create_map, set_map_mode
from interface.results import build_results_panel
from interface.run_state import apply_run_completion, start_run
from algorithm.genetic.evolution import RunResult, run_evolution
from algorithm.genetic.initialization import PopulationResult


@dataclass(slots=True)
class ViewHandles:
    """Ссылки на части страницы для следующих интерфейсных задач."""

    map_view: MapHandle
    tabs: Tabs
    panels: TabPanels


STYLE = """
<style>
  :root {
    --ink: #173047;
    --muted: #607789;
    --line: #d8e2e8;
    --paper: #ffffff;
    --wash: #eef4f6;
    --deep: #123147;
    --accent: #167e91;
  }
  body { background: var(--wash); color: var(--ink); }
  .nicegui-content { padding: 0 !important; }
  .app-shell { width: 100%; min-height: 100vh; }
  .topbar { min-height: 78px; padding: 15px 24px; background: var(--deep);
    color: white; display: flex; align-items: center; justify-content: space-between;
    gap: 20px; flex-wrap: wrap; }
  .brand { display: flex; align-items: baseline; gap: 14px; }
  .brand-name { font: 700 27px/1.1 system-ui, sans-serif; letter-spacing: -.03em; }
  .brand-subtitle { color: #bcd0da; font: 400 13px/1.3 system-ui, sans-serif; }
  .top-actions { display: flex; align-items: center; gap: 7px; }
  .top-actions .q-btn { min-height: 35px; }
  .workspace { display: grid; grid-template-columns: 248px minmax(0, 1fr) 360px;
    gap: 12px; padding: 12px; height: calc(100vh - 78px); min-height: 620px; }
  .side-panel { min-width: 0; background: var(--paper); border: 1px solid var(--line);
    border-radius: 12px; padding: 20px; overflow-y: auto; }
  .panel-heading { font: 700 18px/1.3 system-ui, sans-serif; color: var(--ink); }
  .panel-copy { color: var(--muted); font: 400 13px/1.5 system-ui, sans-serif; }
  .panel-section { padding-bottom: 18px; margin-bottom: 18px;
    border-bottom: 1px solid var(--line); }
  .panel-section:last-child { border-bottom: 0; }
  .left-panel .q-field { width: 100%; }
  .map-stage { position: relative; min-width: 0; min-height: 580px;
    border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
    background: #dce9ec; }
  .map-canvas { width: 100%; height: 100%; min-height: 580px; }
  .map-message { position: absolute; z-index: 500; top: 18px; right: 18px;
    max-width: 300px; padding: 15px 17px; background: rgba(255,255,255,.96);
    border: 1px solid var(--line); border-radius: 9px; pointer-events: none;
    box-shadow: 0 5px 24px rgba(18,49,71,.09); }
  .map-message-title { font: 700 15px/1.3 system-ui, sans-serif; }
  .map-message-copy { color: var(--muted); font: 400 12px/1.4 system-ui, sans-serif; }
  .map-message .q-btn { pointer-events: auto; margin-top: 9px; }
  .map-demo-status { position: absolute; z-index: 500; bottom: 25px; left: 18px;
    max-width: 80%; padding: 8px 10px; background: white; border: 1px solid #ce3f35;
    border-radius: 6px; font: 600 12px/1.4 system-ui, sans-serif; pointer-events: none; }
  .right-panel { padding: 0; display: flex; flex-direction: column; min-height: 0; }
  .right-panel .q-tabs { border-bottom: 1px solid var(--line); }
  .right-panel .q-tab-panels { flex: 1; min-height: 0; overflow-y: auto; }
  .tab-content { padding: 20px; }
  .setting-field { width: 100%; margin-top: 12px; }
  .setting-error { color: #a73835; font: 500 12px/1.4 system-ui, sans-serif; }
  .penalty-card { margin-top: 10px; box-shadow: none; border: 1px solid var(--line); }
  .penalty-title { font: 650 13px/1.3 system-ui, sans-serif; }
  .right-panel .q-expansion-item { border-bottom: 1px solid var(--line); }
  .empty-block { margin-top: 18px; padding: 18px; border: 1px dashed #b9cbd4;
    border-radius: 9px; background: #f7fafb; }
  .results-section { margin-top: 22px; padding-top: 18px; border-top: 1px solid var(--line); }
  .playback-actions { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
  .playback-actions .q-btn { min-height: 30px; }
  .chart-list { margin-top: 12px; }
  .chart-placeholder { box-shadow: none; border: 1px dashed #b9cbd4;
    background: #f7fafb; min-height: 92px; }
  .chart-enlarged { width: min(90vw, 850px); min-height: 320px; }
  @media (max-width: 1120px) {
    .workspace { grid-template-columns: 220px minmax(0, 1fr); height: auto; }
    .right-panel { grid-column: 1 / -1; min-height: 300px; }
  }
  @media (max-width: 720px) {
    .topbar { padding: 16px; }
    .brand { display: block; }
    .brand-subtitle { margin-top: 5px; }
    .workspace { display: flex; flex-direction: column; padding: 8px; min-height: 0; }
    .map-stage, .map-canvas { min-height: 420px; }
    .map-stage { order: -1; }
    .side-panel { overflow: visible; }
  }
</style>
"""


def _disabled_button(label: str, reason: str, icon: str | None = None) -> None:
    """Показать недоступную команду и причину блокировки."""

    button = ui.button(label, icon=icon).props("outline dense").disable()
    button.tooltip(reason)


def _build_header(state: UIState, open_results) -> None:
    with ui.element("header").classes("topbar"):
        with ui.element("div").classes("brand"):
            ui.label("GARoute").classes("brand-name")
            ui.label("Генетический поиск маршрута").classes("brand-subtitle")
        with ui.element("div").classes("top-actions"):
            async def calculate() -> None:
                handle, error = start_run(state)
                if handle is None:
                    state.run_status = "failed"
                    state.run_error = error
                    refresh()
                    return
                try:
                    outcome = await asyncio.to_thread(run_evolution, handle.run_input)
                except Exception as error:
                    outcome = RunResult("failed", "worker_error", (), None, None, str(error))
                if apply_run_completion(state, handle.run_id, outcome) and outcome.status == "completed":
                    open_results()

            button = ui.button("Рассчитать", on_click=calculate, icon="route").props("outline dense")
            status = ui.label("").classes("panel-copy")

            def refresh() -> None:
                ready = (isinstance(state.initial_population, PopulationResult)
                         and state.initial_population.error is None
                         and not state.initial_population_running)
                if ready and state.active_run_id is None:
                    button.enable()
                else:
                    button.disable()
                if state.run_status == "running":
                    status.set_text("Идёт расчёт…")
                elif state.run_error:
                    status.set_text(f"Расчёт недоступен: {state.run_error}")
                elif state.run_status == "completed":
                    status.set_text("Расчёт завершён")
                else:
                    status.set_text("")

            state.refresh_run = refresh
            refresh()
            play_top = ui.button(
                "Воспроизвести", on_click=lambda: state.playback_action("play")
                if state.playback_action is not None else None, icon="play_arrow",
            ).props("outline dense").disable()
            back_top = ui.button(
                "Шаг назад", on_click=lambda: state.playback_action("back")
                if state.playback_action is not None else None, icon="skip_previous",
            ).props("outline dense").disable()
            next_top = ui.button(
                "Шаг вперёд", on_click=lambda: state.playback_action("next")
                if state.playback_action is not None else None, icon="skip_next",
            ).props("outline dense").disable()

            def refresh_playback_header() -> None:
                playback = state.playback
                count = playback.generation_count
                if count > 1 and not playback.playing:
                    play_top.enable()
                else:
                    play_top.disable()
                if count and playback.generation_index > 0:
                    back_top.enable()
                else:
                    back_top.disable()
                if count and playback.generation_index < count - 1:
                    next_top.enable()
                else:
                    next_top.disable()

            state.refresh_playback_header = refresh_playback_header


def _build_left_panel(state: UIState) -> tuple[Input, Input]:
    with ui.element("aside").classes("side-panel left-panel"):
        with ui.element("section").classes("panel-section"):
            ui.label("Маршрут").classes("panel-heading")
            ui.label("Выберите начало и конец после загрузки графа.").classes("panel-copy")
            endpoint_a = ui.input("Точка A").props("outlined dense disable")
            endpoint_b = ui.input("Точка B").props("outlined dense disable")
            if not state.demo_mode:
                ui.select(
                    {"view": "Просмотр", "A": "Выбрать A", "B": "Выбрать B",
                     "traffic": "Редактировать пробки"},
                    label="Режим карты", value="view",
                    on_change=lambda e: set_map_mode(state, e.value),
                ).props("outlined dense").classes("setting-field")
        with ui.element("section").classes("panel-section"):
            ui.label("Дорожные условия").classes("panel-heading")
            ui.label("Режим «Редактировать пробки»: ЛКМ +1, ПКМ −1. Уровень 11 закрывает проезд.").classes(
                "panel-copy"
            )
            reset_button = ui.button(
                "Сбросить пробки", on_click=lambda: state.request_traffic_reset()
                if state.request_traffic_reset is not None else None,
                icon="restart_alt",
            ).props("outline dense")
            reset_button.disable()

            def refresh_reset() -> None:
                if state.segment_levels and state.traffic_segments is not None:
                    reset_button.enable()
                else:
                    reset_button.disable()

            state.refresh_traffic_controls = refresh_reset
        with ui.element("section").classes("panel-section"):
            ui.label("Легенда").classes("panel-heading")
            ui.label("Пробка: 1 — свободно, 2–10 — замедление, 11 — закрыто. Цвет линии показывает уровень визуального сегмента; расчётный уровень всего ребра определяется длиной покрытия.").classes("panel-copy")
            with ui.row().classes("w-full gap-1"):
                for level, color in enumerate(TRAFFIC_COLORS, start=1):
                    ui.label(str(level)).style(
                        f"background:{color};color:white;min-width:20px;text-align:center;"
                        "border-radius:3px;font-size:11px;"
                    )
            ui.label("Маршруты появятся после расчёта.").classes("panel-copy")
        if state.demo_mode:
            with ui.element("section").classes("panel-section"):
                ui.label("ДЕМО: события карты").classes("panel-heading")
                ui.label("Тестовые клики не задают реальные A/B или пробки.").classes("panel-copy")
                mode_label = ui.label("Режим: просмотр").classes("panel-copy")

                def change_mode(value: str) -> None:
                    if set_map_mode(state, value):
                        mode_label.set_text(f"Режим: {value}")

                ui.select(
                    {"view": "Просмотр", "A": "Выбрать A", "B": "Выбрать B",
                     "traffic": "Редактировать пробку"},
                    label="Режим карты", value="view",
                    on_change=lambda e: change_mode(e.value),
                ).props("outlined dense").classes("setting-field")

    return endpoint_a, endpoint_b


def _build_right_panel(state: UIState) -> tuple[Tabs, TabPanels, object]:
    with ui.element("aside").classes("side-panel right-panel"):
        pending_ui: dict[str, object] = {}
        with ui.dialog().props("persistent") as change_dialog, ui.card():
            ui.label("Изменение условий расчёта").classes("panel-heading")
            ui.label(
                "Вы собираетесь изменить условия расчёта. Все результаты предыдущих "
                "запусков будут безвозвратно удалены. Продолжить?"
            ).classes("panel-copy")

            def cancel() -> None:
                cancel_change(state)
                restore = pending_ui.pop("restore", None)
                pending_ui.pop("committed", None)
                change_dialog.close()
                if restore:
                    restore()

            def confirm() -> None:
                confirm_change(state)
                committed = pending_ui.pop("committed", None)
                pending_ui.pop("restore", None)
                change_dialog.close()
                if committed:
                    committed()

            with ui.row():
                ui.button("Отмена", on_click=cancel).props("flat")
                ui.button("Изменить и удалить", on_click=confirm).props("color=negative")

        def submit(pending: PendingChange, restore, committed) -> None:
            result = request_condition_change(state, pending)
            if result.errors:
                restore()
            elif state.pending_change is not None:
                pending_ui["restore"] = restore
                pending_ui["committed"] = committed
                change_dialog.open()
            else:
                committed()

        with ui.tabs().classes("w-full") as tabs:
            settings = ui.tab("Настройки")
            results = ui.tab("Результаты")
        with ui.tab_panels(tabs, value=settings, keep_alive=True).classes("w-full") as panels:
            with ui.tab_panel(settings).classes("tab-content"):
                ui.label("Настройки расчёта").classes("panel-heading")
                build_ga_controls(state, submit)
                build_fitness_controls(state, submit)
                build_normalization_controls(state, submit)
                build_penalty_controls(state, submit)
            with ui.tab_panel(results).classes("tab-content"):
                build_results_panel(state)
    return tabs, panels, results


def build_layout(state: UIState) -> ViewHandles:
    """Собрать одну страницу с картой и панелями, сохраняя карту при смене вкладки."""

    ui.add_head_html(STYLE)
    tabs_holder = {}
    with ui.element("div").classes("app-shell"):
        _build_header(state, lambda: tabs_holder["tabs"].set_value(tabs_holder["results"]))
        with ui.element("main").classes("workspace"):
            endpoint_inputs = _build_left_panel(state)
            map_view = create_map(state, endpoint_inputs=endpoint_inputs)
            tabs, panels, results = _build_right_panel(state)
            tabs_holder["tabs"] = tabs
            tabs_holder["results"] = results
    return ViewHandles(map_view=map_view, tabs=tabs, panels=panels)
