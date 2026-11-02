"""Наблюдаемые события сложности P01–P15 без подстановки отсутствующих фактов."""

from dataclasses import dataclass
from math import isfinite
import re
from typing import Mapping

from pyproj import Geod

from algorithm.genetic.validation import validate_route
from algorithm.models import EdgeId, RoadEdge, Route, RouteContext


PENALTY_IDS = tuple(f"P{index:02d}" for index in range(1, 16))
PRESENT, ABSENT, UNKNOWN, UNSUPPORTED = "PRESENT", "ABSENT", "UNKNOWN", "UNSUPPORTED"
_GEOD = Geod(ellps="WGS84")
_METRES = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(?:m)?\s*$", re.IGNORECASE)
_INTEGER = re.compile(r"^\s*([1-9]\d*)\s*$")
_BUMP_TYPES = frozenset({"bump", "hump", "table", "cushion"})


@dataclass(frozen=True, slots=True)
class PenaltyPolicy:
    """Рабочие пороги Q04 VERIFY и нормализованные рёбра области."""

    roads: Mapping[EdgeId, RoadEdge]
    turn_min_degrees: float = 30.0
    uturn_min_degrees: float = 150.0
    multilane_min: int = 3
    narrow_width_m: float = 3.5
    distance_unit_m: float = 100.0


@dataclass(frozen=True, slots=True)
class PenaltyEvent:
    """Факт, отсутствие, неопределённость или отсутствие детектора."""

    category: str
    status: str
    object_id: str | None
    quantity: float
    source: str
    detail: str = ""


@dataclass(frozen=True, slots=True)
class PenaltyEvents:
    """Реестр с отдельной неопределённостью; баллы добавит T04.03."""

    observations: tuple[PenaltyEvent, ...]

    @property
    def confirmed(self) -> tuple[PenaltyEvent, ...]:
        return tuple(item for item in self.observations if item.status == PRESENT)

    @property
    def uncertain(self) -> tuple[PenaltyEvent, ...]:
        return tuple(item for item in self.observations if item.status in (UNKNOWN, UNSUPPORTED))


def _way_id(edge: RoadEdge) -> str:
    if edge.osm_way_ids:
        return "osm:way:" + ",".join(map(str, edge.osm_way_ids))
    return f"graph:edge:{edge.edge_id!r}"


def _number(value: object) -> float | None:
    if type(value) in (int, float):
        try:
            parsed = float(value)
        except OverflowError:
            return None
    elif isinstance(value, str):
        match = _METRES.fullmatch(value)
        if match is None:
            return None
        parsed = float(match.group(1))
    else:
        return None
    return parsed if isfinite(parsed) and parsed > 0 else None


def _lanes(edge: RoadEdge) -> int | None:
    attrs = edge.raw_attributes
    reverse = attrs.get("reversed")
    direction = "backward" if reverse is True else "forward"
    key = f"lanes:{direction}"
    if reverse in (True, False) and key in attrs:
        value = attrs[key]
    elif attrs.get("oneway") is True:
        value = attrs.get("lanes")
    else:
        return None
    if type(value) is int and value > 0:
        return value
    if isinstance(value, str):
        match = _INTEGER.fullmatch(value)
        return int(match.group(1)) if match else None
    return None


def _direction_status(attrs: Mapping[str, object], edge: RoadEdge, category: str) -> str:
    keys = {"P07": ("traffic_signals:direction", "direction"),
            "P08": ("stop:direction", "direction"),
            "P09": ("give_way:direction", "direction")}[category]
    raw = next((attrs[key] for key in keys if key in attrs), None)
    if raw == "both":
        return PRESENT
    reversed_way = edge.raw_attributes.get("reversed")
    if raw not in ("forward", "backward") or reversed_way not in (True, False):
        return UNKNOWN
    actual = "backward" if reversed_way is True else "forward"
    return PRESENT if raw == actual else ABSENT


def _at_grade(node_attrs: Mapping[str, object], incoming: RoadEdge) -> bool:
    """Явный узел пересечения не должен противоречить мосту/тоннелю дороги."""

    attrs = incoming.raw_attributes
    if attrs.get("bridge") not in (None, "no", False) or attrs.get("tunnel") not in (None, "no", False):
        return False
    if attrs.get("layer") not in (None, "0", 0):
        return False
    if node_attrs.get("bridge") not in (None, "no", False):
        return False
    return True


def _turn_category(incoming: RoadEdge, outgoing: RoadEdge,
                   policy: PenaltyPolicy) -> str | None:
    a = incoming.geometry_lonlat
    b = outgoing.geometry_lonlat
    if len(a) < 2 or len(b) < 2:
        return None
    if a[-2] == a[-1] or b[0] == b[1]:
        return None
    azimuth_in = _GEOD.inv(*a[-2], *a[-1])[0]
    azimuth_out = _GEOD.inv(*b[0], *b[1])[0]
    if not isfinite(azimuth_in) or not isfinite(azimuth_out):
        return None
    delta = (azimuth_out - azimuth_in + 180) % 360 - 180
    if abs(delta) >= policy.uturn_min_degrees:
        return "P03"
    if abs(delta) >= policy.turn_min_degrees:
        return "P01" if delta > 0 else "P02"
    return ""


def extract_penalty_events(route: Route, context: RouteContext, policy: PenaltyPolicy) -> PenaltyEvents:
    """Извлечь только подтверждённые факты; неполноту оставить отдельной."""

    report = validate_route(route, context)
    if not report.valid:
        raise ValueError(f"invalid route: {', '.join(report.codes)}")
    if not (
        all(type(value) in (int, float) and isfinite(value) for value in (
            policy.turn_min_degrees, policy.uturn_min_degrees,
            policy.narrow_width_m, policy.distance_unit_m,
        ))
        and 0 < policy.turn_min_degrees < policy.uturn_min_degrees <= 180
        and type(policy.multilane_min) is int and policy.multilane_min >= 1
        and policy.narrow_width_m > 0 and policy.distance_unit_m > 0
    ):
        raise ValueError("invalid penalty policy")
    roads = [policy.roads[edge_id] for edge_id in route.edges]
    if any(not isfinite(edge.length_m) or edge.length_m <= 0 for edge in roads):
        raise ValueError("road length must be positive and finite")
    observations: list[PenaltyEvent] = []
    seen: set[tuple[str, str | None]] = set()

    def add(category: str, status: str, object_id: str | None, quantity: float,
            source: str, detail: str = "") -> None:
        if category not in PENALTY_IDS or status not in (PRESENT, ABSENT, UNKNOWN, UNSUPPORTED):
            raise ValueError("invalid penalty observation")
        key = (category, object_id)
        if object_id is not None and key in seen:
            return
        if object_id is not None:
            seen.add(key)
        observations.append(PenaltyEvent(category, status, object_id, quantity, source, detail))

    for index, (edge_id, edge) in enumerate(zip(route.edges, roads)):
        u, v, _ = edge_id
        attrs = edge.raw_attributes
        edge_object = f"{_way_id(edge)}:{edge_id!r}"

        if attrs.get("junction") == "roundabout" and (
            index == 0 or roads[index - 1].raw_attributes.get("junction") != "roundabout"
        ):
            add("P06", PRESENT, f"osm:roundabout:{u}:{_way_id(edge)}", 1,
                "junction=roundabout", "один непрерывный проезд")

        lane_count = _lanes(edge)
        if lane_count is None:
            add("P12", UNKNOWN, edge_object, 0, "lanes:forward/backward",
                "число полос в направлении движения неизвестно")
        elif lane_count >= policy.multilane_min:
            add("P12", PRESENT, edge_object, edge.length_m / policy.distance_unit_m,
                "lanes:forward/backward or one-way lanes", f"{lane_count} полос")

        if attrs.get("highway") in ("residential", "living_street"):
            width = _number(attrs.get("width"))
            if width is None:
                add("P13", UNKNOWN, edge_object, 0, "width",
                    "ширина жилой улицы неизвестна")
            elif width < policy.narrow_width_m:
                add("P13", PRESENT, edge_object, edge.length_m / policy.distance_unit_m,
                    "highway+width", f"ширина {width:g} м")

        if attrs.get("hazard") == "school_zone" and (
            index == 0 or roads[index - 1].raw_attributes.get("hazard") != "school_zone"
        ):
            add("P14", PRESENT, f"osm:school_zone:{u}:{_way_id(edge)}", 1,
                "hazard=school_zone", "один вход в отмеченную зону")

        node_attrs = context.graph.nodes[v]
        node_object = f"osm:node:{v}"
        highway = node_attrs.get("highway")
        category = {"traffic_signals": "P07", "stop": "P08", "give_way": "P09"}.get(highway)
        if category is not None:
            status = _direction_status(node_attrs, edge, category)
            add(category, status, node_object, 1 if status == PRESENT else 0,
                f"node.highway={highway}; direction", "направление не подтверждено" if status == UNKNOWN else "")

        railway = node_attrs.get("railway")
        rail_category = {"tram_level_crossing": "P10", "level_crossing": "P11"}.get(railway)
        if rail_category is not None:
            status = PRESENT if _at_grade(node_attrs, edge) else UNKNOWN
            add(rail_category, status, node_object, 1 if status == PRESENT else 0,
                f"node.railway={railway}", "конфликт с bridge/tunnel/layer" if status == UNKNOWN else "")

        if node_attrs.get("traffic_calming") in _BUMP_TYPES:
            add("P15", PRESENT, node_object, 1, "node.traffic_calming")

        if index:
            previous = roads[index - 1]
            junction = u
            neighbors = set(context.graph.predecessors(junction)) | set(context.graph.successors(junction))
            if (len(neighbors) >= 3
                    and previous.raw_attributes.get("junction") != "roundabout"
                    and attrs.get("junction") != "roundabout"):
                if not previous.osm_way_ids or not edge.osm_way_ids:
                    for turn in ("P01", "P02", "P03"):
                        add(turn, UNKNOWN, f"osm:node:{junction}:{turn}", 0,
                            "directed geometry + OSM way", "неизвестно происхождение дороги")
                elif set(previous.osm_way_ids).isdisjoint(edge.osm_way_ids):
                    turn = _turn_category(previous, edge, policy)
                    if turn is None:
                        for category_id in ("P01", "P02", "P03"):
                            add(category_id, UNKNOWN, f"osm:node:{junction}:{category_id}", 0,
                                "directed geometry", "нет корректной касательной")
                    elif turn:
                        add(turn, PRESENT, f"osm:node:{junction}", 1,
                            "directed geometry + distinct OSM ways")

    add("P04", UNSUPPORTED, None, 0, "road-level graph",
        "фактическая смена полосы не наблюдается")
    for category in PENALTY_IDS:
        if not any(item.category == category for item in observations):
            status = ABSENT if category in ("P01", "P02", "P03") else UNKNOWN
            add(category, status, None, 0, "route/OSM coverage",
                "нет подтверждённого события" if status == ABSENT else "недостаточно данных для отрицания")
    return PenaltyEvents(tuple(observations))
