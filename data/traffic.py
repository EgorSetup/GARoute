"""Замедление Q02 и визуальные дорожные сегменты с происхождением OSM."""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import networkx as nx
from pyproj import Geod

from algorithm.models import EdgeId, RoadEdge
from config import TRAFFIC_SPEED_FACTORS
from data.road_graph import NormalizedRoadData
from data.restrictions import is_edge_accessible


_GEOD = Geod(ellps="WGS84")


@dataclass(frozen=True, slots=True)
class CoveredInterval:
    """Покрытие направленного ребра в метрах от его начала."""

    edge_id: EdgeId
    start_m: float
    end_m: float


@dataclass(frozen=True, slots=True)
class VisualSegment:
    """Одна редактируемая часть направленной цепочки OSM way."""

    segment_id: str
    way_id: int
    name: str
    geometry_lonlat: tuple[tuple[float, float], ...]
    length_m: float
    coverage: tuple[CoveredInterval, ...]


@dataclass(frozen=True, slots=True)
class SegmentationIssue:
    """Ребро остаётся недоступным для правки до исправления источника."""

    edge_id: EdgeId
    reason: str


@dataclass(frozen=True, slots=True)
class SegmentCollection:
    """Полные сегменты и явные ошибки покрытия доступных рёбер."""

    segments: tuple[VisualSegment, ...]
    issues: tuple[SegmentationIssue, ...]


def _edge_order(edge_id: EdgeId) -> str:
    return repr(edge_id)


def _geometry_lengths(edge: RoadEdge) -> tuple[float, ...]:
    """Накопленные геодезические длины вершин полилинии."""

    points = edge.geometry_lonlat
    if not isinstance(points, (tuple, list)) or len(points) < 2 or any(
        not isinstance(point, (tuple, list)) or len(point) != 2
        or any(type(value) not in (int, float) for value in point)
        or not all(isfinite(value) for value in point)
        or not -180 <= point[0] <= 180 or not -90 <= point[1] <= 90
        for point in points
    ):
        raise ValueError("invalid_geometry")
    cumulative = [0.0]
    for first, second in zip(points, points[1:]):
        _, _, distance = _GEOD.inv(*first, *second)
        if not isfinite(distance) or distance < 0:
            raise ValueError("invalid_geometry")
        cumulative.append(cumulative[-1] + distance)
    if cumulative[-1] <= 0:
        raise ValueError("invalid_geometry")
    return tuple(cumulative)


def _point_at(points: tuple[tuple[float, float], ...], distances: tuple[float, ...],
              target: float) -> tuple[float, float]:
    if target <= 0:
        return points[0]
    if target >= distances[-1]:
        return points[-1]
    for index in range(len(points) - 1):
        if distances[index + 1] < target or distances[index + 1] == distances[index]:
            continue
        first, second = points[index], points[index + 1]
        azimuth, _, length = _GEOD.inv(*first, *second)
        lon, lat, _ = _GEOD.fwd(*first, azimuth, target - distances[index])
        return lon, lat
    raise ValueError("invalid_geometry")


def _slice_geometry(edge: RoadEdge, distances: tuple[float, ...],
                    start_m: float, end_m: float) -> tuple[tuple[float, float], ...]:
    """Вырезать геометрию по долям нормализованной длины ребра."""

    start = distances[-1] * start_m / edge.length_m
    end = distances[-1] * end_m / edge.length_m
    points = [_point_at(edge.geometry_lonlat, distances, start)]
    points.extend(point for point, distance in zip(edge.geometry_lonlat[1:-1], distances[1:-1])
                  if start < distance < end)
    points.append(_point_at(edge.geometry_lonlat, distances, end))
    return tuple(point for index, point in enumerate(points)
                 if index == 0 or point != points[index - 1])


def _chains(edges: tuple[EdgeId, ...]) -> tuple[tuple[EdgeId, ...], ...]:
    """Продлевать way только при единственном неразворотном продолжении."""

    incoming: dict[object, list[EdgeId]] = defaultdict(list)
    outgoing: dict[object, list[EdgeId]] = defaultdict(list)
    for edge_id in edges:
        outgoing[edge_id[0]].append(edge_id)
        incoming[edge_id[1]].append(edge_id)
    successor: dict[EdgeId, EdgeId] = {}
    for edge_id in edges:
        candidates = [next_edge for next_edge in outgoing[edge_id[1]]
                      if next_edge[1] != edge_id[0]]
        if len(candidates) != 1:
            continue
        next_edge = candidates[0]
        predecessors = [previous for previous in incoming[next_edge[0]]
                        if previous[0] != next_edge[1]]
        if predecessors == [edge_id]:
            successor[edge_id] = next_edge
    has_predecessor = set(successor.values())
    ordered = sorted((edge for edge in edges if edge not in has_predecessor), key=_edge_order)
    ordered.extend(sorted((edge for edge in edges if edge in has_predecessor), key=_edge_order))
    seen: set[EdgeId] = set()
    chains = []
    for first in ordered:
        if first in seen:
            continue
        chain = []
        edge = first
        while edge not in seen:
            chain.append(edge)
            seen.add(edge)
            if edge not in successor:
                break
            edge = successor[edge]
        chains.append(tuple(chain))
    return tuple(chains)


def build_visual_segments(
    road_data: NormalizedRoadData, segment_length: float,
    street_policy: str = "osm_way",
) -> SegmentCollection:
    """Нарезать доступные направленные way, сохранив интервалы исходных рёбер."""

    if (type(segment_length) not in (int, float) or not isfinite(segment_length)
            or segment_length <= 0):
        raise ValueError("segment length must be positive and finite")
    if street_policy != "osm_way":
        raise ValueError("unsupported street policy")
    if not isinstance(road_data.graph, nx.MultiDiGraph):
        raise ValueError("road data requires MultiDiGraph")
    by_way: dict[int, list[EdgeId]] = defaultdict(list)
    geometry_lengths: dict[EdgeId, tuple[float, ...]] = {}
    issues: list[SegmentationIssue] = []
    target_edges = [edge_id for edge_id in road_data.graph.edges(keys=True)
                    if is_edge_accessible(edge_id, road_data)]
    for edge_id in target_edges:
        edge = road_data.edges.get(edge_id)
        if edge is None:
            issues.append(SegmentationIssue(edge_id, "missing_normalized_edge"))
            continue
        if len(edge.osm_way_ids) != 1:
            issues.append(SegmentationIssue(edge_id, "ambiguous_way_id"))
            continue
        if type(edge.length_m) not in (int, float) or not isfinite(edge.length_m) or edge.length_m <= 0:
            issues.append(SegmentationIssue(edge_id, "invalid_length"))
            continue
        try:
            geometry_lengths[edge_id] = _geometry_lengths(edge)
        except ValueError as error:
            issues.append(SegmentationIssue(edge_id, str(error)))
            continue
        by_way[edge.osm_way_ids[0]].append(edge_id)

    segments: list[VisualSegment] = []
    for way_id in sorted(by_way):
        for chain_index, chain in enumerate(_chains(tuple(by_way[way_id]))):
            part_index = 0
            coverage: list[CoveredInterval] = []
            geometry: list[tuple[float, float]] = []
            accumulated = 0.0
            name = next((raw for edge_id in chain
                         if isinstance(raw := road_data.edges[edge_id].raw_attributes.get("name"), str)
                         and raw.strip()), f"way {way_id}")

            def flush() -> None:
                nonlocal part_index, accumulated
                if not coverage:
                    return
                segments.append(VisualSegment(
                    f"{way_id}:{chain_index}:{part_index}", way_id, name,
                    tuple(geometry), accumulated, tuple(coverage),
                ))
                part_index += 1
                coverage.clear()
                geometry.clear()
                accumulated = 0.0

            for edge_id in chain:
                edge = road_data.edges[edge_id]
                start = 0.0
                while start < edge.length_m:
                    remaining = segment_length - accumulated
                    end = min(edge.length_m, start + remaining)
                    if end <= start:
                        raise ValueError(f"segment length cannot advance on {edge_id!r}")
                    piece = _slice_geometry(edge, geometry_lengths[edge_id], start, end)
                    if not geometry:
                        geometry.extend(piece)
                    else:
                        geometry.extend(piece[1:] if geometry[-1] == piece[0] else piece)
                    coverage.append(CoveredInterval(edge_id, start, end))
                    accumulated += end - start
                    start = end
                    if accumulated >= segment_length:
                        flush()
            flush()

    covered: dict[EdgeId, list[CoveredInterval]] = defaultdict(list)
    for segment in segments:
        for interval in segment.coverage:
            covered[interval.edge_id].append(interval)
    issue_edges = {issue.edge_id for issue in issues}
    for edge_id in target_edges:
        if edge_id in issue_edges:
            continue
        intervals = sorted(covered[edge_id], key=lambda item: item.start_m)
        previous = 0.0
        for interval in intervals:
            if interval.start_m != previous or interval.end_m <= interval.start_m:
                issues.append(SegmentationIssue(edge_id, "invalid_coverage"))
                break
            previous = interval.end_m
        else:
            if previous != road_data.edges[edge_id].length_m:
                issues.append(SegmentationIssue(edge_id, "missing_coverage"))
    return SegmentCollection(tuple(segments), tuple(issues))


def speed_factor(level: int, factors: Mapping[int, float] = TRAFFIC_SPEED_FACTORS) -> float:
    """Вернуть k(n) для 1–10; закрытие 11 обрабатывать до деления."""

    if type(level) is not int or not 1 <= level <= 11:
        raise ValueError(f"invalid traffic level: {level!r}")
    if level == 11:
        raise ValueError("traffic level 11 closes the edge")
    factor = factors.get(level)
    if type(factor) not in (int, float) or not isfinite(factor) or factor <= 0 or factor > 1:
        raise ValueError(f"invalid speed factor for level {level}")
    return float(factor)
