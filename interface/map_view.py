"""Карта Leaflet и проверяемые режимы её событий."""

import asyncio
import json
from dataclasses import dataclass
from math import isfinite
from random import Random
from typing import Sequence

import networkx as nx
from pyproj import Geod

from nicegui import ui
from nicegui.elements.input import Input
from nicegui.elements.label import Label
from nicegui.elements.leaflet.leaflet import Leaflet

from algorithm.genetic.initialization import (
    PopulationResult, choose_seed, generate_initial_population, production_weight_strategy,
)
from algorithm.evaluation.penalties import PenaltyPolicy, calculate_penalties, extract_penalty_events
from algorithm.evaluation.fitness import (
    TimeModel, calculate_length, calculate_travel_time, compute_easy_references,
)
from algorithm.evaluation.normalization import compute_scales
from algorithm.models import Route, RouteContext, RouteMetrics
from config import GAROUTE_AREA
from data.osm_loader import OSMLoadError, load_road_network
from data.restrictions import apply_access_to_graph, build_forbidden_turns
from data.road_graph import NormalizedRoadData, build_road_data
from data.traffic import assign_edge_levels, build_visual_segments
from interface.controls import (
    RoutePreview, UIState, cancel_change, confirm_change, request_condition_change,
    propose_traffic_edit, reset_traffic, select_endpoint,
)
from interface.results import population_status_text
from interface.run_state import SavedRun
from visualization.animation import Frame, frame_layers


MAP_MODES = ("view", "A", "B", "traffic")
TRAFFIC_COLORS = (
    "#167e91", "#4590a3", "#6ba6a4", "#8db599", "#b2bd81",
    "#d4ba68", "#e2a44e", "#df8640", "#d66536", "#bf4b38", "#252a36",
)


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


def _prepare_roads(segment_length: float):
    """Подготовить геометрию и запреты в рабочем потоке вне UI."""

    raw = load_road_network(GAROUTE_AREA)
    try:
        normalized = build_road_data(raw)
        graph, access_report = apply_access_to_graph(normalized)
        turn_rules = build_forbidden_turns(normalized.raw_relations, normalized)
        segments = build_visual_segments(normalized, segment_length)
        levels = assign_edge_levels(segments, {})
    except ValueError as error:
        raise OSMLoadError("format", str(error)) from error
    return raw, normalized, graph, access_report, turn_rules, segments, levels


def _road_geojson(roads: NormalizedRoadData, graph: nx.MultiDiGraph, accessible: bool) -> dict:
    """Собрать один слой Leaflet из всех рёбер нужного статуса."""

    features = [
        {"type": "Feature", "geometry": {"type": "LineString", "coordinates": edge.geometry_lonlat}}
        for edge_id, edge in roads.edges.items()
        if (graph[edge_id[0]][edge_id[1]][edge_id[2]]["car_access"] is True) == accessible
    ]
    return {"type": "FeatureCollection", "features": features}


def route_geometry(route: Route, roads: NormalizedRoadData) -> list[list[float]]:
    """Собрать полилинию особи по исходной направленной геометрии рёбер."""

    coordinates: list[list[float]] = []
    for edge_id in route.edges:
        edge = roads.edges[edge_id]
        points = [[lon, lat] for lon, lat in edge.geometry_lonlat]
        coordinates.extend(points if not coordinates else points[1:])
    return coordinates


def render_population(map_element: Leaflet, population: Sequence[Route],
                      roads: NormalizedRoadData, existing_layers: Sequence[object] = ()) -> list[object]:
    """Заменить, а не накопить, линии P0 по направленным геометриям."""

    for layer in existing_layers:
        map_element.remove_layer(layer)
    palette = ("#d35c23", "#8b5bb1", "#c54875", "#499f5a", "#d1a02c")
    layers: list[object] = []
    for index, route in enumerate(population):
        feature = {"type": "Feature", "geometry": {
            "type": "LineString", "coordinates": route_geometry(route, roads),
        }}
        layer = map_element.generic_layer(name="geoJSON", args=[feature, {
            "style": {"color": palette[index % len(palette)], "weight": 5,
                      "opacity": 0.75, "interactive": False},
        }])
        layers.append(layer)
    return layers


def clear_run_layers(map_element: Leaflet, layers: list[object]) -> None:
    """Удалить предыдущий кадр до показа следующего."""

    for layer in layers:
        map_element.remove_layer(layer)
    layers.clear()


def render_frame(map_element: Leaflet, frame: Frame, roads: NormalizedRoadData,
                 existing_layers: list[object]) -> list[object]:
    """Показать фактические маршруты кадра и их сохранённое происхождение."""

    clear_run_layers(map_element, existing_layers)
    for styled in frame_layers(frame):
        feature = {"type": "Feature", "geometry": {
            "type": "LineString", "coordinates": route_geometry(styled.route, roads),
        }}
        options = {"color": styled.color, "weight": styled.weight,
                   "opacity": styled.opacity, "interactive": False}
        if styled.dashed:
            options["dashArray"] = "6 5"
        existing_layers.append(map_element.generic_layer(
            name="geoJSON", args=[feature, {"style": options}],
        ))
    return existing_layers


def create_map(state: UIState, *, endpoint_inputs: tuple[Input, Input] | None = None) -> MapHandle:
    """Показать область, реальные дороги и привязанные к графу точки A/B."""

    with ui.element("div").classes("map-stage"):
        center = ((GAROUTE_AREA.north + GAROUTE_AREA.south) / 2,
                  (GAROUTE_AREA.east + GAROUTE_AREA.west) / 2)
        map_element = ui.leaflet(center=center, zoom=13).classes("map-canvas")
        markers: dict[str, object] = {}
        population_layers: list[object] = []
        playback_layers: list[object] = []
        run_layers: list[object] = []
        segment_layers: dict[str, object] = {}
        rendered_levels: dict[str, int] = {}

        def refresh_traffic() -> None:
            for segment_id, layer in segment_layers.items():
                level = state.segment_levels.get(segment_id, 1)
                if level != rendered_levels[segment_id]:
                    layer.run_method("setStyle", {"color": TRAFFIC_COLORS[level - 1]})
                    rendered_levels[segment_id] = level
            if state.refresh_traffic_controls is not None:
                state.refresh_traffic_controls()

        state.refresh_traffic = refresh_traffic

        def refresh_run_overlays() -> None:
            clear_run_layers(map_element, run_layers)
            if state.normalized_roads is None:
                return
            for record in state.saved_runs:
                if not isinstance(record, SavedRun) or not record.visible or record.outcome.best is None:
                    continue
                feature = {"type": "Feature", "geometry": {
                    "type": "LineString",
                    "coordinates": route_geometry(record.outcome.best.route, state.normalized_roads),
                }}
                run_layers.append(map_element.generic_layer(
                    name="geoJSON", args=[feature, {"style": {
                        "color": record.color, "weight": 4, "opacity": 0.6,
                        "interactive": False,
                    }}],
                ))

        state.refresh_run_overlays = refresh_run_overlays

        def show_generation_frame(frame: Frame | None) -> None:
            if frame is None or state.normalized_roads is None:
                clear_run_layers(map_element, playback_layers)
                return
            for layer in population_layers:
                map_element.remove_layer(layer)
            population_layers.clear()
            render_frame(map_element, frame, state.normalized_roads, playback_layers)

        state.show_generation_frame = show_generation_frame

        def clear_population() -> None:
            for layer in population_layers:
                map_element.remove_layer(layer)
            population_layers.clear()
            state.initial_population = None
            state.initial_seed_used = None
            state.initial_population_preset = None
            state.initial_complexities = ()
            state.initial_previews = ()
            state.initial_scales = None
            state.initial_scale_error = None
            state.easy_references = None
            state.easy_reference_error = None
            if state.refresh_population_results is not None:
                state.refresh_population_results()
            if state.refresh_normalization is not None:
                state.refresh_normalization()
            if state.refresh_easy is not None:
                state.refresh_easy()
            if not state.demo_mode:
                population_label.set_text("P0 ещё не сформирована")
            if state.refresh_run is not None:
                state.refresh_run()

        if not state.demo_mode:
            state.clear_population_layers = clear_population

            with ui.dialog().props("persistent") as traffic_dialog, ui.card():
                ui.label("Изменение пробки").classes("panel-heading")
                ui.label("Все результаты предыдущих запусков будут удалены. Продолжить?")

                def cancel_traffic() -> None:
                    cancel_change(state)
                    traffic_dialog.close()
                    status_label.set_text("Изменение пробки отменено")

                def confirm_traffic() -> None:
                    confirm_change(state)
                    traffic_dialog.close()
                    status_label.set_text("Дорожные условия изменены; прежние результаты удалены")

                with ui.row():
                    ui.button("Отмена", on_click=cancel_traffic).props("flat")
                    ui.button("Изменить и удалить", on_click=confirm_traffic).props("color=negative")

            def submit_traffic(proposal) -> None:
                result = request_condition_change(state, proposal)
                if result.errors:
                    status_label.set_text(result.errors[0])
                elif state.pending_change is not None:
                    traffic_dialog.open()
                else:
                    status_label.set_text("Дорожные условия изменены")

            def request_reset() -> None:
                if state.segment_levels:
                    submit_traffic(reset_traffic(state))

            state.request_traffic_reset = request_reset

            def on_traffic_segment(e) -> None:
                if state.map_mode != "traffic":
                    return
                segment_id = e.args.get("segment_id")
                action = e.args.get("action")
                if action not in ("increase", "decrease"):
                    return
                delta = 1 if action == "increase" else -1
                previous = state.segment_levels.get(segment_id, 1)
                submit_traffic(propose_traffic_edit(state, segment_id, delta))
                if state.pending_change is None and previous != state.segment_levels.get(segment_id, 1):
                    effective = {state.traffic_levels.get(interval.edge_id, 1)
                                 for segment in state.traffic_segments.segments
                                 if segment.segment_id == segment_id for interval in segment.coverage}
                    status_label.set_text(
                        f"Сегмент {segment_id}: уровень {state.segment_levels.get(segment_id, 1)}; "
                        f"связанные рёбра: {', '.join(map(str, sorted(effective)))}"
                    )

            ui.on("garoute-traffic-segment", on_traffic_segment)

        def show_pair_status() -> None:
            if state.endpoint_a is not None and state.endpoint_b is not None:
                status_label.set_text("A/B выбраны и направленно достижимы.")
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
            if state.endpoint_a is not None and state.endpoint_b is not None:
                p0_button.enable()

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
                hint_label = ui.label("Дорожная сеть появится после загрузки области.").classes(
                    "map-message-copy"
                )
                async def load_area() -> None:
                    if state.active_run_id is not None:
                        return
                    button.disable()
                    status_label.set_text("Загружаю автомобильную сеть OSM…")
                    try:
                        (road_data, normalized, graph, access_report, turn_rules,
                         segments, levels) = await asyncio.to_thread(
                            _prepare_roads, state.config.ga.segment_length_m,
                        )
                    except OSMLoadError as error:
                        show_load_error(MapHandle(map_element, status_label), error)
                        button.enable()
                        return
                    state.road_data = road_data
                    state.normalized_roads = normalized
                    state.routing_graph = graph
                    state.turn_rules = turn_rules
                    state.traffic_segments = segments
                    state.traffic_levels = levels
                    state.graph_version += 1
                    state.graph_loaded = True
                    if state.refresh_modeled_speeds is not None:
                        state.refresh_modeled_speeds()
                    for accessible, color in ((True, "#167e91"), (False, "#9aaab2")):
                        layer = _road_geojson(normalized, graph, accessible)
                        if layer["features"]:
                            map_element.generic_layer(name="geoJSON", args=[layer, {
                                "style": {"color": color, "weight": 3 if accessible else 4,
                                          "opacity": 0.85, "interactive": False},
                            }])
                    for visual in segments.segments:
                        layer = map_element.generic_layer(name="polyline", args=[
                            [[lat, lon] for lon, lat in visual.geometry_lonlat],
                            {"color": TRAFFIC_COLORS[0], "weight": 7, "opacity": 0.8,
                             "bubblingMouseEvents": False},
                        ])
                        segment_layers[visual.segment_id] = layer
                        rendered_levels[visual.segment_id] = 1
                        for event, action in (("click", "increase"), ("contextmenu", "decrease")):
                            callback = (
                                "function(e) { L.DomEvent.stop(e.originalEvent); "
                                "emitEvent('garoute-traffic-segment', {segment_id:"
                                f"{json.dumps(visual.segment_id)}, action:'{action}'}}); }}"
                            )
                            layer.run_method(":on", f"'{event}'", callback)
                    status_label.set_text(
                        f"Граф загружен: {road_data.metadata['nodes']} узлов, "
                        f"{road_data.metadata['edges']} рёбер. Выберите A и B. "
                        f"Неподдержанных отношений: {len(turn_rules.issues)}."
                    )
                    hint_label.set_text("Выберите A и B слева.")
                    button.set_visibility(False)

                button = ui.button("Загрузить дороги", on_click=load_area).props("outline dense")
                population_label = ui.label("P0 ещё не сформирована").classes("map-message-copy")

                async def show_initial_population() -> None:
                    if state.active_run_id is not None or state.initial_population_running:
                        return
                    if (state.routing_graph is None or state.normalized_roads is None
                            or state.endpoint_a is None or state.endpoint_b is None):
                        return
                    p0_button.disable()
                    state.initial_population_running = True
                    if state.refresh_run is not None:
                        state.refresh_run()

                    def finish() -> None:
                        state.initial_population_running = False
                        p0_button.enable()
                        if state.refresh_run is not None:
                            state.refresh_run()

                    population_label.set_text("Формирую P0…")
                    graph = state.routing_graph
                    roads = state.normalized_roads
                    start, end = state.endpoint_a, state.endpoint_b
                    app_config = state.config
                    config = app_config.ga
                    seed_used = state.base_seed_used if state.base_seed_used is not None else choose_seed(config.seed)
                    context = RouteContext(graph, start, end, state.turn_rules)
                    time_model = TimeModel(
                        roads.edges, dict(state.traffic_levels),
                        modeled_speeds_enabled=app_config.fitness.modeled_speeds_enabled,
                        modeled_speeds_kph=dict(app_config.fitness.modeled_speeds_kph),
                    )
                    base = state.base_population
                    if not isinstance(base, PopulationResult) or base.error is not None:
                        try:
                            base = await asyncio.to_thread(
                                generate_initial_population, context, config, Random(seed_used),
                                production_weight_strategy(config.random_weight_strength),
                            )
                        except (ValueError, OverflowError) as error:
                            population_label.set_text(f"Базовая P0 не сформирована: {error}")
                            finish()
                            return
                        if base.error is not None:
                            population_label.set_text(population_status_text(base))
                            finish()
                            return
                        if (state.endpoint_a, state.endpoint_b, state.config) != (start, end, app_config):
                            finish()
                            return
                        state.base_population = base
                        state.base_seed_used = seed_used
                    result = base
                    if (state.endpoint_a, state.endpoint_b, state.config) != (start, end, app_config):
                        finish()
                        return
                    breakdowns = ()
                    previews = ()
                    complexity_error = None
                    scales = None
                    scale_error = None
                    easy_refs = None
                    easy_error = None
                    if result.error is None and app_config.fitness.penalties is not None:
                        def evaluate_p0():
                            policy = PenaltyPolicy(roads.edges)
                            evaluated = tuple(calculate_penalties(
                                extract_penalty_events(route, context, policy),
                                app_config.fitness.penalties,
                            ) for route in result.population)
                            try:
                                references = compute_easy_references(context, time_model)
                                reference_error = None
                            except (ValueError, KeyError, OverflowError) as error:
                                references = None
                                reference_error = f"L_min/T_min недоступны: {error}"
                            route_previews = []
                            for route in result.population:
                                try:
                                    length = calculate_length(route, context, time_model)
                                except (ValueError, KeyError, OverflowError) as error:
                                    route_previews.append(RoutePreview(None, None, str(error)))
                                    continue
                                try:
                                    time = calculate_travel_time(route, context, time_model)
                                    route_previews.append(RoutePreview(length, time))
                                except (ValueError, KeyError, OverflowError) as error:
                                    route_previews.append(RoutePreview(length, None, str(error)))
                            previews_snapshot = tuple(route_previews)
                            if any(item.time_s is None for item in previews_snapshot):
                                first_error = next(item.error for item in previews_snapshot if item.time_s is None)
                                return (evaluated, previews_snapshot, None,
                                        f"Автоэталоны не вычислены: {first_error}",
                                        references, reference_error)
                            metrics = tuple(RouteMetrics(item.length_m, item.time_s, breakdown.total)
                                            for item, breakdown in zip(previews_snapshot, evaluated))
                            try:
                                scales_snapshot = compute_scales(metrics)
                                return evaluated, previews_snapshot, scales_snapshot, None, references, reference_error
                            except (ValueError, OverflowError) as error:
                                return (evaluated, previews_snapshot, None,
                                        f"Автоэталоны не вычислены: {error}",
                                        references, reference_error)
                        try:
                            breakdowns, previews, scales, scale_error, easy_refs, easy_error = await asyncio.to_thread(evaluate_p0)
                        except (ValueError, KeyError, OverflowError) as error:
                            complexity_error = str(error)
                    base_scales = state.base_scales
                    if base_scales is None and result is base:
                        base_scales = scales
                    elif base_scales is None and app_config.fitness.penalties is not None:
                        def evaluate_base_scales():
                            policy = PenaltyPolicy(roads.edges)
                            metrics = tuple(RouteMetrics(
                                calculate_length(route, context, time_model),
                                calculate_travel_time(route, context, time_model),
                                calculate_complexity(calculate_penalties(
                                    extract_penalty_events(route, context, policy),
                                    app_config.fitness.penalties,
                                )),
                            ) for route in base.population)
                            return compute_scales(metrics)
                        try:
                            base_scales = await asyncio.to_thread(evaluate_base_scales)
                        except (ValueError, KeyError, OverflowError):
                            pass
                    if (state.endpoint_a, state.endpoint_b, state.config) != (start, end, app_config):
                        finish()
                        return
                    clear_population()
                    state.initial_population = result
                    state.base_scales = base_scales
                    state.initial_seed_used = seed_used
                    state.initial_population_preset = None
                    state.initial_complexities = breakdowns
                    state.initial_previews = previews
                    state.initial_scales = base_scales
                    state.initial_scale_error = scale_error
                    state.easy_references = easy_refs
                    state.easy_reference_error = easy_error
                    if result.error is None:
                        population_layers.extend(render_population(map_element, result.population, roads))
                    status = population_status_text(result)
                    status += f" Seed: {seed_used}."
                    if complexity_error is not None:
                        status += f" C не вычислена: {complexity_error}."
                    population_label.set_text(status)
                    if state.refresh_population_results is not None:
                        state.refresh_population_results()
                    if state.refresh_normalization is not None:
                        state.refresh_normalization()
                    if state.refresh_easy is not None:
                        state.refresh_easy()
                    if state.refresh_run is not None:
                        state.refresh_run()
                    finish()

                p0_button = ui.button("Показать P0", on_click=show_initial_population).props("outline dense")
                p0_button.disable()

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
