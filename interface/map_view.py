"""Карта Leaflet и проверяемые режимы её событий."""

import asyncio
from dataclasses import dataclass
from math import isfinite

import networkx as nx
from pyproj import Geod

from nicegui import ui
from nicegui.elements.input import Input
from nicegui.elements.label import Label
from nicegui.elements.leaflet.leaflet import Leaflet

from config import GAROUTE_AREA
from data.osm_loader import OSMLoadError, load_road_network
from data.restrictions import apply_access_to_graph, build_forbidden_turns
from data.road_graph import NormalizedRoadData, build_road_data
from interface.controls import (
    UIState, cancel_change, confirm_change, request_condition_change, select_endpoint,
)


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


def show_load_error(handle: MapHandle, error: OSMLoadError) -> None:
    """Показать тип и причину сбоя загрузки без подмены его пустой картой."""

    prefix = {"network": "Ошибка сети", "format": "Ошибка данных OSM", "empty": "Пустая область"}
    handle.status_label.set_text(f"{prefix.get(error.kind, 'Ошибка загрузки')}: {error.detail}")


def snap_to_node(coords: tuple[float, float], graph: nx.MultiDiGraph, endpoint: str) -> object | None:
    """Найти ближайшую доступную вершину к клику Leaflet (lat, lon)."""

    lat, lon = coords
    if endpoint not in ("A", "B") or not all(isfinite(value) for value in coords):
        return None
    geod = Geod(ellps="WGS84")
    best: tuple[float, object] | None = None
    for node, attrs in graph.nodes(data=True):
        edges = graph.out_edges(node, keys=True) if endpoint == "A" else graph.in_edges(node, keys=True)
        if not any(graph[u][v][key].get("car_access") is True for u, v, key in edges):
            continue
        distance = geod.inv(lon, lat, float(attrs["x"]), float(attrs["y"]))[2]
        if best is None or distance < best[0]:
            best = (distance, node)
    return None if best is None else best[1]


def _prepare_roads():
    """Подготовить геометрию и запреты в рабочем потоке вне UI."""

    raw = load_road_network(GAROUTE_AREA)
    try:
        normalized = build_road_data(raw)
        graph, access_report = apply_access_to_graph(normalized)
        turn_rules = build_forbidden_turns(normalized.raw_relations, normalized)
    except ValueError as error:
        raise OSMLoadError("format", str(error)) from error
    return raw, normalized, graph, access_report, turn_rules


def _road_geojson(roads: NormalizedRoadData, graph: nx.MultiDiGraph, accessible: bool) -> dict:
    """Собрать один слой Leaflet из всех рёбер нужного статуса."""

    features = [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": edge.geometry_lonlat}}
        for edge_id, edge in roads.edges.items()
        if (graph[edge_id[0]][edge_id[1]][edge_id[2]]["car_access"] is True) == accessible
    ]
    return {"type": "FeatureCollection", "features": features}


def create_map(state: UIState, *, endpoint_inputs: tuple[Input, Input] | None = None) -> MapHandle:
    """Показать область, реальные дороги и привязанные к графу точки A/B."""

    with ui.element("div").classes("map-stage"):
        center = ((GAROUTE_AREA.north + GAROUTE_AREA.south) / 2,
                  (GAROUTE_AREA.east + GAROUTE_AREA.west) / 2)
        map_element = ui.leaflet(center=center, zoom=13).classes("map-canvas")
        markers: dict[str, object] = {}

        def show_pair_status() -> None:
            if state.endpoint_a is not None and state.endpoint_b is not None:
                status_label.set_text("A/B выбраны и направленно достижимы. Расчёт пока недоступен: модель оценки ещё не подключена.")
            elif state.endpoint_a is not None:
                status_label.set_text("Точка A выбрана; теперь выберите B")
            elif state.endpoint_b is not None:
                status_label.set_text("Точка B выбрана; теперь выберите A")

        def refresh_marker(endpoint: str) -> None:
            node = state.endpoint_a if endpoint == "A" else state.endpoint_b
            if node is None or state.routing_graph is None:
                return
            attrs = state.routing_graph.nodes[node]
            lat, lon = float(attrs["y"]), float(attrs["x"])
            marker = markers.get(endpoint)
            if marker is None:
                markers[endpoint] = map_element.marker(latlng=(lat, lon), options={"title": endpoint})
            else:
                marker.move(lat, lon)
            if endpoint_inputs is not None:
                (endpoint_inputs[0] if endpoint == "A" else endpoint_inputs[1]).set_value(str(node))

        if not state.demo_mode:
            with ui.dialog().props("persistent") as endpoint_dialog, ui.card():
                ui.label("Изменение точек маршрута").classes("panel-heading")
                ui.label("Изменение существующей пары удалит результаты. Продолжить?")

                def cancel_endpoint() -> None:
                    cancel_change(state)
                    endpoint_dialog.close()

                def confirm_endpoint() -> None:
                    key = state.pending_change.key if state.pending_change is not None else None
                    confirm_change(state)
                    endpoint_dialog.close()
                    if key in ("endpoint_a", "endpoint_b"):
                        refresh_marker("A" if key == "endpoint_a" else "B")
                        show_pair_status()

                with ui.row():
                    ui.button("Отмена", on_click=cancel_endpoint).props("flat")
                    ui.button("Изменить и удалить", on_click=confirm_endpoint).props("color=negative")
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
                hint_label = ui.label("Дорожная сеть появится после загрузки согласованной области.").classes(
                    "map-message-copy"
                )
                async def load_area() -> None:
                    button.disable()
                    status_label.set_text("Загружаю автомобильную сеть OSM…")
                    try:
                        road_data, normalized, graph, access_report, turn_rules = await asyncio.to_thread(_prepare_roads)
                    except OSMLoadError as error:
                        show_load_error(MapHandle(map_element, status_label), error)
                        button.enable()
                        return
                    state.road_data = road_data
                    state.normalized_roads = normalized
                    state.routing_graph = graph
                    state.turn_rules = turn_rules
                    state.graph_loaded = True
                    for accessible, color in ((True, "#167e91"), (False, "#9aaab2")):
                        layer = _road_geojson(normalized, graph, accessible)
                        if layer["features"]:
                            map_element.generic_layer(name="geoJSON", args=[layer, {
                                "style": {"color": color, "weight": 3 if accessible else 4,
                                          "opacity": 0.85, "interactive": False},
                            }])
                    status_label.set_text(
                        f"Граф загружен: {road_data.metadata['nodes']} узлов, "
                        f"{road_data.metadata['edges']} рёбер. Выберите A и B. "
                        f"Неподдержанных отношений: {len(turn_rules.issues)}."
                    )
                    hint_label.set_text("Режим A/B выбирается слева; бирюзовый — доступная дорога, серый — запрет.")
                    button.set_visibility(False)

                button = ui.button("Загрузить дороги", on_click=load_area).props("outline dense")

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

        else:
            def on_real_map(e) -> None:
                if not state.graph_loaded or state.routing_graph is None or state.map_mode not in ("A", "B"):
                    return
                position = e.args.get("latlng")
                if not position:
                    return
                coords = (float(position["lat"]), float(position["lng"]))
                if not (GAROUTE_AREA.south <= coords[0] <= GAROUTE_AREA.north
                        and GAROUTE_AREA.west <= coords[1] <= GAROUTE_AREA.east):
                    status_label.set_text("Выберите точку внутри загруженной области")
                    return
                endpoint = state.map_mode
                node = snap_to_node(coords, state.routing_graph, endpoint)
                if node is None:
                    status_label.set_text("В области нет доступной автомобильной вершины")
                    return
                proposal = select_endpoint(state, endpoint, node)
                result = request_condition_change(state, proposal)
                if result.errors:
                    status_label.set_text(result.errors[0])
                elif state.pending_change is not None:
                    endpoint_dialog.open()
                else:
                    refresh_marker(endpoint)
                    show_pair_status()

            map_element.on("map-click", on_real_map)

    return MapHandle(element=map_element, status_label=status_label)
