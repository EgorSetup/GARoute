"""Карта Leaflet и честное сообщение о ещё не загруженном графе."""

from dataclasses import dataclass

from nicegui import ui
from nicegui.elements.label import Label
from nicegui.elements.leaflet.leaflet import Leaflet

from interface.controls import UIState


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
            status = "Граф готов" if state.graph_loaded else "Граф ещё не загружен"
            status_label = ui.label(status).classes("map-message-title")
            if not state.graph_loaded:
                ui.label("Дорожная сеть появится после выбора области.").classes(
                    "map-message-copy"
                )

    return MapHandle(element=map_element, status_label=status_label)
