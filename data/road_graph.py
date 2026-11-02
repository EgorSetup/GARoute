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


def build_road_data(raw: RawRoadData) -> NormalizedRoadData:
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
        if raw_speed == "RU:urban":
            speed_kph = 60.0
            speed_status = "implicit_osm_verify"
            speed_rule = "RU:urban_to_60_kph"
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
) -> Route | None:
    if start not in context.graph or end not in context.graph or start == end:
        return None
    if incoming is not None and (incoming[1] != start or not _traversable(incoming, context)):
        return None
    if outgoing is not None and (outgoing[0] != end or not _traversable(outgoing, context)):
        return None

    order = count()
    queue = [(0.0, next(order), start, incoming, (), frozenset({start}))]
    while queue:
        cost, _, node, previous, path, visited = heappop(queue)
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
                    heappush(queue, (cost + edge_cost, next(order), neighbor, edge_id, path + (edge_id,), visited | {neighbor}))
                continue
            heappush(queue, (cost + edge_cost, next(order), neighbor, edge_id, path + (edge_id,), visited | {neighbor}))
    return None


def find_connector(
    start: Hashable,
    end: Hashable,
    context: RouteContext,
    rng: Random,
    incoming: EdgeId | None = None,
    outgoing: EdgeId | None = None,
    weight: EdgeWeight | None = None,
) -> Route | None:
    """Найти один допустимый простой подмаршрут или явно вернуть None."""

    return _search(start, end, context, rng, incoming, outgoing, weight)


def check_reachability(a: Hashable, b: Hashable, context: RouteContext) -> ReachabilityResult:
    """Проверить доступность без записи найденного пути в популяцию."""

    if a == b:
        return ReachabilityResult(False, "same_endpoints")
    route = _search(a, b, context, None, None, None, None)
    return ReachabilityResult(route is not None, None if route is not None else "unreachable")
