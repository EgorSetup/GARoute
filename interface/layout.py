"""Основная страница GARoute без вычислений генетического алгоритма."""

from dataclasses import dataclass

from nicegui import ui
from nicegui.elements.tabs import TabPanels, Tabs

from interface.controls import (
    UIState, apply_setting_change, build_fitness_controls, build_ga_controls,
    build_normalization_controls,
)
from interface.map_view import MapHandle, create_map


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
  .right-panel { padding: 0; display: flex; flex-direction: column; min-height: 0; }
  .right-panel .q-tabs { border-bottom: 1px solid var(--line); }
  .right-panel .q-tab-panels { flex: 1; min-height: 0; overflow-y: auto; }
  .tab-content { padding: 20px; }
  .setting-field { width: 100%; margin-top: 12px; }
  .setting-error { color: #a73835; font: 500 12px/1.4 system-ui, sans-serif; }
  .right-panel .q-expansion-item { border-bottom: 1px solid var(--line); }
  .empty-block { margin-top: 18px; padding: 18px; border: 1px dashed #b9cbd4;
    border-radius: 9px; background: #f7fafb; }
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


def _build_header() -> None:
    with ui.element("header").classes("topbar"):
        with ui.element("div").classes("brand"):
            ui.label("GARoute").classes("brand-name")
            ui.label("Генетический поиск маршрута").classes("brand-subtitle")
        with ui.element("div").classes("top-actions"):
            _disabled_button("Рассчитать", "Сначала нужен дорожный граф и точки A/B", "route")
            _disabled_button("Воспроизвести", "Расчёты ещё не выполнены", "play_arrow")
            _disabled_button("Шаг назад", "Расчёты ещё не выполнены", "skip_previous")
            _disabled_button("Шаг вперёд", "Расчёты ещё не выполнены", "skip_next")


def _build_left_panel() -> None:
    with ui.element("aside").classes("side-panel left-panel"):
        with ui.element("section").classes("panel-section"):
            ui.label("Маршрут").classes("panel-heading")
            ui.label("Выберите начало и конец после загрузки графа.").classes("panel-copy")
            ui.input("Точка A").props("outlined dense disable")
            ui.input("Точка B").props("outlined dense disable")
        with ui.element("section").classes("panel-section"):
            ui.label("Дорожные условия").classes("panel-heading")
            ui.label("Пробки и перекрытия станут доступны после загрузки дорог.").classes(
                "panel-copy"
            )
            _disabled_button("Изменить пробку", "Граф ещё не загружен", "traffic")
            _disabled_button("Сбросить изменения", "Изменений пока нет", "restart_alt")
        with ui.element("section").classes("panel-section"):
            ui.label("Легенда").classes("panel-heading")
            ui.label("Маршруты появятся после расчёта.").classes("panel-copy")


def _build_right_panel(state: UIState) -> tuple[Tabs, TabPanels]:
    with ui.element("aside").classes("side-panel right-panel"):
        with ui.tabs().classes("w-full") as tabs:
            settings = ui.tab("Настройки")
            results = ui.tab("Результаты")
        with ui.tab_panels(tabs, value=settings, keep_alive=True).classes("w-full") as panels:
            with ui.tab_panel(settings).classes("tab-content"):
                ui.label("Настройки расчёта").classes("panel-heading")
                build_ga_controls(state, lambda pending: apply_setting_change(state, pending))
                build_fitness_controls(state)
                build_normalization_controls(state)
            with ui.tab_panel(results).classes("tab-content"):
                ui.label("Результаты").classes("panel-heading")
                with ui.element("div").classes("empty-block"):
                    ui.label("Расчёты ещё не выполнены").classes("panel-heading")
                    ui.label("После запуска здесь появятся маршруты и графики.").classes(
                        "panel-copy"
                    )
    return tabs, panels


def build_layout(state: UIState) -> ViewHandles:
    """Собрать одну страницу с картой и панелями, сохраняя карту при смене вкладки."""

    ui.add_head_html(STYLE)
    with ui.element("div").classes("app-shell"):
        _build_header()
        with ui.element("main").classes("workspace"):
            _build_left_panel()
            map_view = create_map(state)
            tabs, panels = _build_right_panel(state)
    return ViewHandles(map_view=map_view, tabs=tabs, panels=panels)
