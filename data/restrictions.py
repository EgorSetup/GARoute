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
    applied_relation_ids: tuple[int, ...] = ()
    issues: tuple[RestrictionIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class RestrictionIssue:
    """Причина, по которой отношение OSM не перенесено в таблицу."""

    relation_id: int
    reason: str


_SIMPLE_NO = frozenset({"no_left_turn", "no_right_turn", "no_u_turn", "no_straight_on"})
_SIMPLE_ONLY = frozenset({"only_left_turn", "only_right_turn", "only_u_turn", "only_straight_on"})


def is_turn_allowed(in_edge: EdgeId, out_edge: EdgeId, rules: TurnRules) -> bool:
    """Проверить именно заданную пару полных EdgeId."""

    return (in_edge, out_edge) not in rules.forbidden


def build_forbidden_turns(
    relations: tuple[dict, ...], road_data: NormalizedRoadData,
) -> TurnRules:
    """Перенести только однозначные no/only via-node на точные пары EdgeId."""

    forbidden: set[tuple[EdgeId, EdgeId]] = set()
    applied: list[int] = []
    issues: list[RestrictionIssue] = []
    for relation in relations:
        relation_id = relation.get("id")
        if type(relation_id) is not int:
            raise ValueError("restriction relation has no numeric ID")
        tags = relation.get("tags")
        members = relation.get("members")
        if not isinstance(tags, dict) or not isinstance(members, list):
            issues.append(RestrictionIssue(relation_id, "invalid_relation_format"))
            continue
        if tags.get("type") != "restriction":
            issues.append(RestrictionIssue(relation_id, "not_restriction"))
            continue
        restriction = tags.get("restriction")
        if any("conditional" in key for key in tags):
            issues.append(RestrictionIssue(relation_id, "conditional_unsupported"))
            continue
        if "except" in tags or any(key.startswith("restriction:") for key in tags):
            issues.append(RestrictionIssue(relation_id, "vehicle_exception_unsupported"))
            continue
        if not isinstance(restriction, str) or restriction not in _SIMPLE_NO | _SIMPLE_ONLY:
            issues.append(RestrictionIssue(relation_id, "restriction_type_unsupported"))
            continue
        by_role: dict[str, list[dict]] = {"from": [], "via": [], "to": []}
        for member in members:
            if not isinstance(member, dict) or member.get("role") not in by_role:
                continue
            by_role[member["role"]].append(member)
        if any(len(by_role[role]) != 1 for role in by_role):
            issues.append(RestrictionIssue(relation_id, "member_count_unsupported"))
            continue
        from_member, via_member, to_member = (by_role[role][0] for role in ("from", "via", "to"))
        if via_member.get("type") != "node":
            issues.append(RestrictionIssue(relation_id, "via_way_unsupported"))
            continue
        if from_member.get("type") != "way" or to_member.get("type") != "way":
            issues.append(RestrictionIssue(relation_id, "member_type_unsupported"))
            continue
        from_way, via, to_way = (member.get("ref") for member in (from_member, via_member, to_member))
        if any(type(value) is not int for value in (from_way, via, to_way)):
            issues.append(RestrictionIssue(relation_id, "invalid_member_id"))
            continue
        incoming = [edge for edge in road_data.way_to_edges.get(from_way, ()) if edge[1] == via]
        outgoing = [edge for edge in road_data.way_to_edges.get(to_way, ()) if edge[0] == via]
        if len(incoming) != 1 or len(outgoing) != 1:
            reason = "missing_graph_way_or_direction" if not incoming or not outgoing else "ambiguous_way_mapping"
            issues.append(RestrictionIssue(relation_id, reason))
            continue
        in_edge, to_edge = incoming[0], outgoing[0]
        if restriction in _SIMPLE_NO:
            forbidden.add((in_edge, to_edge))
        else:
            for edge in road_data.graph.out_edges(via, keys=True):
                if edge != to_edge:
                    forbidden.add((in_edge, edge))
        applied.append(relation_id)
    return TurnRules(frozenset(forbidden), tuple(applied), tuple(issues))


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
