"""Общий контракт для запретов пар направленных рёбер."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, TYPE_CHECKING

import networkx as nx

from algorithm.models import EdgeId

if TYPE_CHECKING:
    from data.road_graph import NormalizedRoadData


@dataclass(frozen=True, slots=True)
class TurnRules:
    """Запрещённые переходы; импорт OSM-отношений появится в T02.04."""

    forbidden: frozenset[tuple[EdgeId, EdgeId]] = frozenset()


def is_turn_allowed(in_edge: EdgeId, out_edge: EdgeId, rules: TurnRules) -> bool:
    """Проверить именно заданную пару полных EdgeId."""

    return (in_edge, out_edge) not in rules.forbidden


@dataclass(frozen=True, slots=True)
class EdgeAccess:
    """Статус проезда обычного автомобиля по направленному ребру."""

    status: str  # allowed, denied, unsupported, closed, missing
    reason: str

    @property
    def accessible(self) -> bool:
        return self.status == "allowed"


@dataclass(frozen=True, slots=True)
class RestrictionReport:
    """Проверка рёбер без обещания полного учёта временных правил и ПДД."""

    allowed: int
    denied: tuple[tuple[EdgeId, str], ...]
    unsupported: tuple[tuple[EdgeId, str], ...]
    closed: tuple[EdgeId, ...]


_ACCESS_KEYS = ("access", "vehicle", "motor_vehicle", "motorcar")
_ALLOW = frozenset({"yes", "permissive", "designated", "official"})
_DENY = frozenset({"no", "private"})
_RESTRICTED = frozenset({"destination", "delivery", "customers", "agricultural", "forestry", "military"})


def edge_access(
    edge_id: EdgeId,
    road_data: NormalizedRoadData,
    traffic: Mapping[EdgeId, int] | None = None,
) -> EdgeAccess:
    """Проверить граф, перекрытие и поддержанные теги без неявных разрешений."""

    if edge_id not in road_data.edges or not road_data.graph.has_edge(*edge_id):
        return EdgeAccess("missing", "direction_not_in_graph")
    level = (traffic or {}).get(edge_id, 1)
    if type(level) is not int or not 1 <= level <= 11:
        return EdgeAccess("unsupported", "invalid_traffic_level")
    if level == 11:
        return EdgeAccess("closed", "traffic_level_11")

    attrs = road_data.edges[edge_id].raw_attributes
    if any(f"{key}:conditional" in attrs for key in (*_ACCESS_KEYS, "oneway")):
        return EdgeAccess("unsupported", "conditional_access_or_direction")
    reversed_way = attrs.get("reversed")
    if reversed_way not in (True, False) and any(
        f"{key}:{direction}" in attrs
        for key in _ACCESS_KEYS for direction in ("forward", "backward")
    ):
        return EdgeAccess("unsupported", "directional_access_without_orientation")

    value = None
    chosen_key = None
    direction = "backward" if reversed_way is True else "forward"
    for key in _ACCESS_KEYS:
        if key in attrs:
            value, chosen_key = attrs[key], key
        directional_key = f"{key}:{direction}"
        if reversed_way in (True, False) and directional_key in attrs:
            value, chosen_key = attrs[directional_key], directional_key
    if value is None:
        return EdgeAccess("allowed", "osmnx_drive_default")
    if not isinstance(value, str):
        return EdgeAccess("unsupported", f"unrecognized_{chosen_key}")
    normalized = value.strip().lower()
    if normalized in _ALLOW:
        return EdgeAccess("allowed", f"{chosen_key}={normalized}")
    if normalized in _DENY:
        return EdgeAccess("denied", f"{chosen_key}={normalized}")
    if normalized in _RESTRICTED:
        return EdgeAccess("unsupported", f"restricted_use_{chosen_key}={normalized}")
    return EdgeAccess("unsupported", f"unrecognized_{chosen_key}={normalized}")


def is_edge_accessible(
    edge_id: EdgeId,
    road_data: NormalizedRoadData,
    traffic: Mapping[EdgeId, int] | None = None,
) -> bool:
    """Вернуть True только для доказанно доступного ребра."""

    return edge_access(edge_id, road_data, traffic).accessible


def apply_access_to_graph(
    road_data: NormalizedRoadData,
    traffic: Mapping[EdgeId, int] | None = None,
) -> tuple[nx.MultiDiGraph, RestrictionReport]:
    """Пометить копию графа для существующего валидатора и вернуть диагностику."""

    graph = road_data.graph.copy()
    denied: list[tuple[EdgeId, str]] = []
    unsupported: list[tuple[EdgeId, str]] = []
    closed: list[EdgeId] = []
    allowed = 0
    for edge_id in road_data.edges:
        status = edge_access(edge_id, road_data, traffic)
        graph[edge_id[0]][edge_id[1]][edge_id[2]]["car_access"] = status.accessible
        if status.status == "allowed":
            allowed += 1
        elif status.status == "denied":
            denied.append((edge_id, status.reason))
        elif status.status == "unsupported":
            unsupported.append((edge_id, status.reason))
        elif status.status == "closed":
            closed.append(edge_id)
    return graph, RestrictionReport(allowed, tuple(denied), tuple(unsupported), tuple(closed))
