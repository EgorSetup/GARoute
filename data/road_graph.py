"""Связующие пути с учётом входящего ребра и запретов перехода."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from math import isfinite
from random import Random
import re
from typing import Callable, Hashable, Mapping

import networkx as nx
from pyproj import Geod
from shapely.geometry import LineString

from algorithm.models import EdgeId, RoadEdge, Route, RouteContext
from data.osm_loader import RoadData as RawRoadData
from data.restrictions import is_turn_allowed


EdgeWeight = Callable[[EdgeId, Mapping[str, object]], float]
_SPEED_PATTERN = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(km/h|kmh|kph|mph)?\s*$", re.IGNORECASE)
_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True, slots=True)
class NormalizedRoadData:
    """Исходный граф и рёбра с явной диагностикой неизвестных скоростей."""

    graph: nx.MultiDiGraph
    edges: Mapping[EdgeId, RoadEdge]
    way_to_edges: Mapping[int, tuple[EdgeId, ...]]
    unknown_speed_edges: tuple[EdgeId, ...]
    raw_relations: tuple[dict, ...]
    metadata: dict


def _speed_kph(value: object) -> float | None:
    """Разбирать только однозначное число с явной или стандартной единицей."""

    if isinstance(value, bool) or isinstance(value, (list, tuple, set)):
        return None
    if type(value) in (int, float):
        speed = float(value)
    elif isinstance(value, str):
        match = _SPEED_PATTERN.fullmatch(value)
        if match is None:
            return None
        speed = float(match.group(1))
        if (match.group(2) or "").lower() == "mph":
            speed *= 1.609344
    else:
        return None
    return speed if isfinite(speed) and speed > 0 else None


def _node_lonlat(graph: nx.MultiDiGraph, node: Hashable) -> tuple[float, float]:
    attrs = graph.nodes[node]
    try:
        point = (float(attrs["x"]), float(attrs["y"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"node {node!r} has no valid lon/lat") from error
    if not all(isfinite(value) for value in point):
        raise ValueError(f"node {node!r} has no finite lon/lat")
    return point


def _geometry(graph: nx.MultiDiGraph, edge_id: EdgeId, attrs: Mapping[str, object]) -> tuple[tuple[tuple[float, float], ...], str]:
    u, v, _ = edge_id
    start, end = _node_lonlat(graph, u), _node_lonlat(graph, v)
    raw = attrs.get("geometry")
    if raw is None:
        return (start, end), "node_coordinates"
    if not isinstance(raw, LineString):
        raise ValueError(f"edge {edge_id!r} has unsupported geometry")
    coords = tuple((float(x), float(y)) for x, y in raw.coords)
    if len(coords) < 2 or any(not all(isfinite(value) for value in pair) for pair in coords):
        raise ValueError(f"edge {edge_id!r} has invalid geometry")
    near = lambda a, b: abs(a[0] - b[0]) <= 1e-6 and abs(a[1] - b[1]) <= 1e-6
    if near(coords[0], start) and near(coords[-1], end):
        return coords, "osm_geometry"
    if near(coords[-1], start) and near(coords[0], end):
        return tuple(reversed(coords)), "osm_geometry_reversed"
    raise ValueError(f"edge {edge_id!r} geometry does not meet its nodes")


def _way_ids(value: object) -> tuple[int, ...]:
    values = value if isinstance(value, (list, tuple)) else (value,)
    if any(type(item) not in (int, str) or not str(item).isdigit() for item in values):
        return ()
    return tuple(dict.fromkeys(int(item) for item in values))


def _selected_speed(attrs: Mapping[str, object]) -> tuple[object | None, str | None]:
    reversed_way = attrs.get("reversed")
    directional = "maxspeed:backward" if reversed_way is True else "maxspeed:forward"
    if directional in attrs and reversed_way in (True, False):
        return attrs[directional], directional
    if "maxspeed" in attrs:
        return attrs["maxspeed"], "maxspeed"
    return None, None


def build_road_data(raw: RawRoadData, *, urban_context: bool = False) -> NormalizedRoadData:
    """Нормализовать все полные EdgeId без изменения исходного графа."""

    graph = raw.graph
    if not isinstance(graph, nx.MultiDiGraph):
        raise ValueError("road data requires a directed MultiDiGraph")
    edges: dict[EdgeId, RoadEdge] = {}
    way_edges: dict[int, list[EdgeId]] = {}
    unknown: list[EdgeId] = []
    for u, v, key, attrs in graph.edges(keys=True, data=True):
        edge_id = (u, v, key)
        geometry, geometry_source = _geometry(graph, edge_id, attrs)
        raw_length = attrs.get("length")
        if raw_length is None:
            length_m = abs(_GEOD.line_length(*zip(*geometry)))
            length_source = "geodesic_geometry"
        else:
            try:
                length_m = float(raw_length)
            except (TypeError, ValueError) as error:
                raise ValueError(f"edge {edge_id!r} has invalid length") from error
            length_source = "osmnx_length"
        if not isfinite(length_m) or length_m <= 0:
            raise ValueError(f"edge {edge_id!r} has nonpositive length")
        raw_speed, speed_source = _selected_speed(attrs)
        speed_kph = _speed_kph(raw_speed)
        speed_status = "observed" if speed_kph is not None else "unknown"
        speed_rule = "explicit_numeric" if speed_kph is not None else None
        symbolic = {"RU:urban": 60.0, "RU:living_street": 20.0}
        if type(raw_speed) is str and raw_speed in symbolic:
            speed_kph = symbolic[raw_speed]
            speed_status = "implicit_osm_verify"
            speed_rule = f"{raw_speed}_to_{speed_kph:g}_kph"
        if speed_kph is None and raw_speed is None and urban_context:
            road_class = attrs.get("highway")
            if road_class == "living_street" or attrs.get("living_street") == "yes":
                speed_kph = 20.0
                speed_rule = "RU_urban_living_street_20"
            elif type(road_class) is str and road_class in {
                "residential", "unclassified", "tertiary", "tertiary_link",
                "secondary", "secondary_link", "primary", "primary_link",
            }:
                speed_kph = 60.0
                speed_rule = "RU_urban_general_60"
            if speed_kph is not None:
                speed_status = "assumed_rule"
                speed_source = "PDD_10.2+OSM_highway"
        if speed_kph is None:
            unknown.append(edge_id)
        edges[edge_id] = RoadEdge(
            edge_id=edge_id, geometry_lonlat=geometry, geometry_source=geometry_source,
            length_m=length_m, length_source=length_source,
            osm_way_ids=_way_ids(attrs.get("osmid")), raw_attributes=dict(attrs),
            speed_kph=speed_kph, speed_status=speed_status,
            speed_source=speed_source, speed_rule=speed_rule, raw_speed=raw_speed,
        )
        for way_id in edges[edge_id].osm_way_ids:
            way_edges.setdefault(way_id, []).append(edge_id)
    return NormalizedRoadData(
        graph, edges, {way: tuple(ids) for way, ids in way_edges.items()},
        tuple(unknown), raw.raw_relations, raw.metadata,
    )


@dataclass(frozen=True, slots=True)
class ReachabilityResult:
    """Факт достижимости без выдачи готовой особи в P0."""

    reachable: bool
    reason: str | None = None


def _traversable(edge: EdgeId, context: RouteContext) -> bool:
    u, v, key = edge
    return (
        context.graph.has_edge(u, v, key)
        and context.graph[u][v][key].get("car_access") is True
        and edge not in context.closed_edges
    )


def _search(
    start: Hashable,
    end: Hashable,
    context: RouteContext,
    rng: Random | None,
    incoming: EdgeId | None,
    outgoing: EdgeId | None,
    weight: EdgeWeight | None,
    complete: bool,
) -> Route | None:
    if start not in context.graph or end not in context.graph or start == end:
        return None
    if incoming is not None and (incoming[1] != start or not _traversable(incoming, context)):
        return None
    if outgoing is not None and (outgoing[0] != end or not _traversable(outgoing, context)):
        return None

    order = count()
    queue = [(0.0, next(order), start, incoming, (), frozenset({start}))]
    labels: dict[tuple[Hashable, EdgeId | None], list[tuple[float, frozenset[Hashable]]]] = {
        (start, incoming): [(0.0, frozenset({start}))],
    }
    best_cost: dict[tuple[Hashable, EdgeId | None], float] = {(start, incoming): 0.0}

    def enqueue(cost: float, neighbor: Hashable, edge_id: EdgeId,
                path: tuple[EdgeId, ...], visited: frozenset[Hashable]) -> None:
        """Отсекать только путь, доминируемый по стоимости и посещённым вершинам."""

        state = (neighbor, edge_id)
        if not complete:
            if cost >= best_cost.get(state, float("inf")):
                return
            best_cost[state] = cost
            heappush(queue, (cost, next(order), neighbor, edge_id, path, visited))
            return
        previous_labels = labels.setdefault(state, [])
        if any(old_cost <= cost and old_visited <= visited for old_cost, old_visited in previous_labels):
            return
        previous_labels[:] = [
            (old_cost, old_visited) for old_cost, old_visited in previous_labels
            if not (cost <= old_cost and visited <= old_visited)
        ]
        previous_labels.append((cost, visited))
        heappush(queue, (cost, next(order), neighbor, edge_id, path, visited))

    while queue:
        cost, _, node, previous, path, visited = heappop(queue)
        if complete and (cost, visited) not in labels.get((node, previous), ()):
            continue
        if not complete and cost > best_cost.get((node, previous), float("inf")):
            continue
        if node == end:
            return Route(path)
        edges = list(context.graph.out_edges(node, keys=True))
        if rng is not None:
            rng.shuffle(edges)
        else:
            edges.sort(key=lambda edge: tuple(map(str, edge)))
        for edge in edges:
            _, neighbor, key = edge
            edge_id = (node, neighbor, key)
            if neighbor in visited:
                continue
            if not _traversable(edge_id, context):
                continue
            if previous is not None and not is_turn_allowed(previous, edge_id, context.turn_rules):
                continue
            attributes = context.graph[node][neighbor][key]
            edge_cost = 1.0 if weight is None else float(weight(edge_id, attributes))
            if not isfinite(edge_cost) or edge_cost <= 0:
                raise ValueError("edge weight must be positive and finite")
            if neighbor == end:
                if outgoing is None or is_turn_allowed(edge_id, outgoing, context.turn_rules):
                    enqueue(cost + edge_cost, neighbor, edge_id, path + (edge_id,), visited | {neighbor})
                continue
            enqueue(cost + edge_cost, neighbor, edge_id, path + (edge_id,), visited | {neighbor})
    return None


def find_connector(
    start: Hashable,
    end: Hashable,
    context: RouteContext,
    rng: Random,
    incoming: EdgeId | None = None,
    outgoing: EdgeId | None = None,
    weight: EdgeWeight | None = None,
    complete: bool = False,
) -> Route | None:
    """Искать простой подмаршрут; быстрый отказ не доказывает недостижимость.

    complete=True сохраняет альтернативы с разными посещёнными вершинами
    для точных эталонов и проверки достижимости.
    """

    return _search(start, end, context, rng, incoming, outgoing, weight, complete)


def check_reachability(a: Hashable, b: Hashable, context: RouteContext) -> ReachabilityResult:
    """Проверить доступность без записи найденного пути в популяцию."""

    if a == b:
        return ReachabilityResult(False, "same_endpoints")
    route = _search(a, b, context, None, None, None, None, True)
    return ReachabilityResult(route is not None, None if route is not None else "unreachable")
