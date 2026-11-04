"""Основная страница GARoute и неблокирующий запуск расчёта."""

import asyncio
from dataclasses import dataclass

from nicegui import ui
from nicegui.elements.tabs import TabPanels, Tabs
from nicegui.elements.label import Label

from interface.controls import (
    UIState, PendingChange, build_fitness_controls, build_ga_controls,
    build_normalization_controls, build_penalty_controls, cancel_change,
    confirm_change, request_condition_change,
)
from interface.map_view import MapHandle, create_map, set_map_mode
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
  .topbar { min-height: 44px; padding: 4px 12px; background: var(--deep);
    color: white; display: flex; align-items: center; justify-content: space-between;
    gap: 20px; flex-wrap: wrap; }
  .top-actions { display: flex; align-items: center; gap: 7px; }
  .top-actions .q-btn { min-height: 35px; }
  .workspace { display: grid; grid-template-columns: 205px minmax(0, 1fr) 340px;
    gap: 8px; padding: 8px; height: calc(100vh - 44px); min-height: 560px; }
  .side-panel { min-width: 0; background: var(--paper); border: 1px solid var(--line);
    border-radius: 9px; padding: 12px; overflow-y: auto; }
  .panel-heading { font: 700 18px/1.3 system-ui, sans-serif; color: var(--ink); }
  .panel-copy { color: var(--muted); font: 400 13px/1.5 system-ui, sans-serif; }
  .panel-section { padding-bottom: 10px; margin-bottom: 10px;
    border-bottom: 1px solid var(--line); }
  .panel-section:last-child { border-bottom: 0; }
  .mode-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px; }
  .mode-grid .q-btn { min-width: 0; padding: 4px 3px; font-size: 11px; }
  .mode-active { outline: 2px solid #167e91; outline-offset: -2px; }
  .mode-set { background: #228f55 !important; color: white !important; }
  .map-stage { position: relative; min-width: 0; min-height: 500px;
    border: 1px solid var(--line); border-radius: 12px; overflow: hidden;
    background: #dce9ec; }
  .map-canvas { width: 100%; height: 100%; min-height: 500px; }
  .map-demo-status { position: absolute; z-index: 500; bottom: 25px; left: 18px;
    max-width: 80%; padding: 8px 10px; background: white; border: 1px solid #ce3f35;
    border-radius: 6px; font: 600 12px/1.4 system-ui, sans-serif; pointer-events: none; }
  .right-panel { padding: 0; display: flex; flex-direction: column; min-height: 0; }
  .right-panel .q-tabs { border-bottom: 1px solid var(--line); }
  .right-panel .q-tab-panels { flex: 1; min-height: 0; overflow-y: auto; }
  .tab-content { padding: 10px; }
  .setting-field { width: 100%; margin-top: 7px; }
  .setting-error { color: #a73835; font: 500 12px/1.4 system-ui, sans-serif; }
  .penalty-card { margin-top: 10px; box-shadow: none; border: 1px solid var(--line); }
  .penalty-title { font: 650 13px/1.3 system-ui, sans-serif; }
  .right-panel .q-expansion-item { border-bottom: 1px solid var(--line); }
  .empty-block { margin-top: 10px; padding: 10px; border: 1px dashed #b9cbd4;
    border-radius: 9px; background: #f7fafb; }
  .results-section { margin-top: 10px; padding-top: 8px; border-top: 1px solid var(--line); }
  .result-overview { font: 650 15px/1.35 system-ui, sans-serif; margin-top: 8px; }
  .playback-actions { display: flex; flex-wrap: wrap; gap: 4px; margin: 6px 0; }
  .playback-actions .q-btn { min-height: 30px; }
  .chart-list { margin-top: 6px; gap: 6px; }
  .chart-placeholder { box-shadow: none; border: 1px dashed #b9cbd4;
    background: #f7fafb; min-height: 70px; padding: 4px; }
  .chart-enlarged { width: min(90vw, 850px); min-height: 320px; }
  @media (max-width: 1120px) {
    .workspace { grid-template-columns: 200px minmax(0, 1fr); height: auto; }
    .right-panel { grid-column: 1 / -1; min-height: 300px; }
  }
  @media (max-width: 720px) {
    .topbar { padding: 6px; }
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
                    if state.run_error == "attempts_exhausted":
                        status.set_text("Не удалось заполнить поколение за ограниченное число попыток")
                    else:
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
            def refresh_playback_header() -> None:
                playback = state.playback
                count = playback.generation_count
                if count > 1 and not playback.playing:
                    play_top.enable()
                else:
                    play_top.disable()

            state.refresh_playback_header = refresh_playback_header


def _build_left_panel(state: UIState) -> tuple[Label, Label]:
    with ui.element("aside").classes("side-panel left-panel"):
        with ui.element("section").classes("panel-section"):
            with ui.element("div").classes("mode-grid"):
                buttons = {
                    mode: ui.button(label, on_click=lambda m=mode: set_map_mode(state, m)).props("outline dense")
                    for mode, label in (("view", "Просмотр"), ("A", "Точка A"),
                                        ("B", "Точка B"), ("traffic", "Пробки"))
                }
            p0_button = ui.button(
                "Создать P0", on_click=lambda: state.create_p0() if state.create_p0 else None,
            ).props("outline dense").classes("w-full")
            status_label = ui.label("Загрузка дорог…").classes("panel-copy")
            population_label = ui.label("").classes("panel-copy")
            retry_button = ui.button(
                "Повторить загрузку", on_click=lambda: state.load_roads() if state.load_roads else None,
            ).props("flat dense")
            retry_button.set_visibility(False)

            def refresh_map_controls() -> None:
                for mode, button in buttons.items():
                    button.classes(add="mode-active" if state.map_mode == mode else None,
                                   remove="mode-active" if state.map_mode != mode else None)
                    selected = mode == "A" and state.endpoint_a is not None or mode == "B" and state.endpoint_b is not None
                    button.classes(add="mode-set" if selected else None,
                                   remove="mode-set" if not selected else None)
                    if selected:
                        button.props(add="color=positive unelevated", remove="outline")
                    else:
                        button.props(add="outline", remove="color=positive unelevated")
                    if mode in ("A", "B"):
                        button.set_text(("✓ " if selected else "") + ("Точка A" if mode == "A" else "Точка B"))
                if (state.graph_loaded and state.endpoint_a is not None and state.endpoint_b is not None
                        and not state.initial_population_running and state.active_run_id is None):
                    p0_button.enable()
                else:
                    p0_button.disable()
                retry_button.set_visibility(state.graph_load_failed)

            state.refresh_map_controls = refresh_map_controls
            refresh_map_controls()
        with ui.element("section").classes("panel-section"):
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
    return status_label, population_label


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
            status_labels = _build_left_panel(state)
            map_view = create_map(state, status_label=status_labels[0], population_label=status_labels[1])
            tabs, panels, results = _build_right_panel(state)
            tabs_holder["tabs"] = tabs
            tabs_holder["results"] = results
            tabs.on_value_change(
                lambda event: state.refresh_charts()
                if event.value == results and state.refresh_charts is not None else None,
            )
    return ViewHandles(map_view=map_view, tabs=tabs, panels=panels)
