"""Карта Leaflet и проверяемые режимы её событий."""

from dataclasses import dataclass

from nicegui import ui
from nicegui.elements.label import Label
from nicegui.elements.leaflet.leaflet import Leaflet

from interface.controls import UIState


MAP_MODES = ("view", "A", "B", "traffic")


def set_map_mode(state: UIState, mode: str) -> bool:
    """Включить ровно один режим карты."""

    if mode not in MAP_MODES:
        return False
    state.map_mode = mode
    return True


def handle_map_click(state: UIState, coords: tuple[float, float]) -> bool:
    """В деморежиме проверить выбор A/B; без графа обычный выбор недоступен."""

    if not state.demo_mode or state.map_mode not in ("A", "B"):
        return False
    if state.map_mode == "A":
        state.demo_point_a = coords
    else:
        state.demo_point_b = coords
    state.demo_map_clicks += 1
    return True


def handle_segment_click(state: UIState, segment_id: str, action: str) -> bool:
    """Менять только демонстрационный сегмент и только в режиме пробок."""

    if not state.demo_mode or state.map_mode != "traffic" or segment_id != "demo":
        return False
    if action not in ("increase", "decrease"):
        return False
    step = 1 if action == "increase" else -1
    state.demo_segment_level = max(1, min(11, state.demo_segment_level + step))
    state.demo_segment_clicks += 1
    return True


@dataclass(slots=True)
class MapHandle:
    """Элементы карты, которые понадобятся при подключении дорожного графа."""

    element: Leaflet
    status_label: Label


def create_map(state: UIState) -> MapHandle:
    """Показать карту без маршрутов и без выдуманной целевой области."""

    with ui.element("div").classes("map-stage"):
        # Центр мира временный: точная область карты остаётся OPEN Q01.
        map_element = ui.leaflet(center=(0.0, 0.0), zoom=2).classes("map-canvas")
        with ui.element("div").classes("map-message"):
            status = "ДЕМО: проверка событий, не дорожный граф" if state.demo_mode else (
                "Граф готов" if state.graph_loaded else "Граф ещё не загружен"
            )
            status_label = ui.label(status).classes("map-message-title")
            if state.demo_mode:
                ui.label("Красная линия — тестовый сегмент; данные не участвуют в расчёте.").classes(
                    "map-message-copy"
                )
            elif not state.graph_loaded:
                ui.label("Дорожная сеть появится после выбора области.").classes(
                    "map-message-copy"
                )

        if state.demo_mode:
            segment = map_element.generic_layer(
                name="polyline",
                args=[[[0, -35], [0, 35]],
                      {"color": "#ce3f35", "weight": 16, "bubblingMouseEvents": False}],
            )
            event_label = ui.label("Режим: просмотр · уровень 1 · событий сегмента 0")
            event_label.classes("map-demo-status")

            def refresh() -> None:
                event_label.set_text(
                    f"Режим: {state.map_mode} · уровень {state.demo_segment_level} · "
                    f"событий сегмента {state.demo_segment_clicks} · "
                    f"кликов карты {state.demo_map_clicks} · "
                    f"A={state.demo_point_a} · B={state.demo_point_b}"
                )

            def on_segment(e) -> None:
                if handle_segment_click(state, e.args["segment_id"], e.args["action"]):
                    refresh()

            ui.on("garoute-demo-segment", on_segment)

            def on_map(e) -> None:
                position = e.args.get("latlng")
                if position and handle_map_click(state, (position["lat"], position["lng"])):
                    refresh()

            map_element.on("map-click", on_map)

            def attach_segment_events() -> None:
                for event, action in (("click", "increase"), ("contextmenu", "decrease")):
                    callback = (
                        "function(e) { L.DomEvent.stop(e.originalEvent); "
                        f"emitEvent('garoute-demo-segment', {{segment_id:'demo', action:'{action}'}}); }}"
                    )
                    segment.run_method(":on", f"'{event}'", callback)

            map_element.on("init", attach_segment_events)

    return MapHandle(element=map_element, status_label=status_label)
