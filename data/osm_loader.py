"""Загрузка исходной автомобильной сети OSM и отношений ограничений."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from typing import Iterator
from xml.etree import ElementTree

import networkx as nx
import osmnx as ox
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
import requests

from config import AreaConfig, OVERPASS_URL


DEFAULT_CACHE_DIR = Path(__file__).resolve().parent / "osm_cache"
OSM_MAP_URL = "https://api.openstreetmap.org/api/0.6/map"
_LOAD_LOCK = RLock()


class OSMLoadError(RuntimeError):
    """Ошибка сети, формата или пустой области с понятным типом для UI."""

    def __init__(self, kind: str, detail: str):
        self.kind = kind
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class RoadData:
    """Сырой граф, полные OSM-отношения и происхождение загрузки."""

    graph: nx.MultiDiGraph
    raw_relations: tuple[dict, ...]
    metadata: dict
    cache_hit: bool


def _request_description(area: AreaConfig) -> dict:
    return {
        "bbox_wsen": list(area.bbox),
        "network_type": area.network_type,
        "simplify": False,
        "retain_all": True,
        "osmnx_version": ox.__version__,
        "overpass_url": OVERPASS_URL,
        "relations_source": OSM_MAP_URL,
        "relations_tiles_per_axis": 2,
    }


def _cache_paths(cache_dir: Path, request: dict) -> tuple[Path, Path, Path]:
    key = sha256(json.dumps(request, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    stem = cache_dir / f"drive_{key}"
    return stem.with_suffix(".graphml"), stem.with_suffix(".relations.json"), stem.with_suffix(".metadata.json")


def _hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_graph(graph: object, area: AreaConfig) -> nx.MultiDiGraph:
    if not isinstance(graph, nx.MultiDiGraph):
        raise OSMLoadError("format", "OSM не вернул ориентированный мультиграф")
    if graph.number_of_nodes() == 0 or graph.number_of_edges() == 0:
        raise OSMLoadError("empty", "В выбранной области нет автомобильных дорог")
    for _, attrs in graph.nodes(data=True):
        try:
            x, y = float(attrs["x"]), float(attrs["y"])
        except (KeyError, TypeError, ValueError) as error:
            raise OSMLoadError("format", "У вершины OSM нет корректных координат") from error
        if not (area.west - 0.001 <= x <= area.east + 0.001
                and area.south - 0.001 <= y <= area.north + 0.001):
            raise OSMLoadError("format", "Вершина OSM расположена вне запрошенной области")
    return graph


def _graph_summary(graph: nx.MultiDiGraph) -> dict:
    weak = [len(nodes) for nodes in nx.weakly_connected_components(graph)]
    strong = [len(nodes) for nodes in nx.strongly_connected_components(graph)]
    xs = [float(attrs["x"]) for _, attrs in graph.nodes(data=True)]
    ys = [float(attrs["y"]) for _, attrs in graph.nodes(data=True)]
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "weak_components": len(weak),
        "largest_weak_component": max(weak),
        "strong_components": len(strong),
        "largest_strong_component": max(strong),
        "graph_bbox_wsen": [min(xs), min(ys), max(xs), max(ys)],
    }


@contextmanager
def _osmnx_cache(cache_dir: Path) -> Iterator[None]:
    old_folder, old_use_cache = ox.settings.cache_folder, ox.settings.use_cache
    old_overpass_url = ox.settings.overpass_url
    old_tags = list(ox.settings.useful_tags_way)
    try:
        ox.settings.cache_folder = cache_dir / "http"
        ox.settings.use_cache = True
        ox.settings.overpass_url = OVERPASS_URL
        ox.settings.useful_tags_way = list(dict.fromkeys(old_tags + [
            "surface", "smoothness", "maxspeed:forward", "maxspeed:backward",
            "lanes:forward", "lanes:backward", "turn:lanes",
        ]))
        yield
    finally:
        ox.settings.cache_folder, ox.settings.use_cache = old_folder, old_use_cache
        ox.settings.overpass_url = old_overpass_url
        ox.settings.useful_tags_way = old_tags


def _fetch_graph(area: AreaConfig, cache_dir: Path) -> nx.MultiDiGraph:
    try:
        with _osmnx_cache(cache_dir):
            graph = ox.graph_from_bbox(
                area.bbox, network_type=area.network_type,
                simplify=False, retain_all=True,
            )
    except InsufficientResponseError as error:
        raise OSMLoadError("empty", "OSM не нашёл автомобильных дорог в области") from error
    except (requests.RequestException, ResponseStatusCodeError, TimeoutError, ConnectionError) as error:
        raise OSMLoadError("network", f"Не удалось загрузить дорожный граф OSM: {error}") from error
    return _validate_graph(graph, area)


def _fetch_relations(area: AreaConfig) -> tuple[dict, ...]:
    """Получить исходные отношения через OSM API небольшими плитками."""

    middle_lon = (area.west + area.east) / 2
    middle_lat = (area.south + area.north) / 2
    boxes = (
        (area.west, area.south, middle_lon, middle_lat),
        (middle_lon, area.south, area.east, middle_lat),
        (area.west, middle_lat, middle_lon, area.north),
        (middle_lon, middle_lat, area.east, area.north),
    )
    relations: dict[int, dict] = {}
    with requests.Session() as session:
        for box in boxes:
            try:
                response = session.get(
                    OSM_MAP_URL,
                    params={"bbox": ",".join(map(str, box))},
                    timeout=90,
                    headers={"User-Agent": "GARoute/0.1 (local educational application)"},
                )
                response.raise_for_status()
            except requests.RequestException as error:
                raise OSMLoadError("network", f"Не удалось загрузить ограничения OSM: {error}") from error
            try:
                root = ElementTree.fromstring(response.content)
                if root.tag != "osm":
                    raise ValueError("not an OSM document")
                for element in root.iter("relation"):
                    tags = {tag.attrib["k"]: tag.attrib["v"] for tag in element.iter("tag")}
                    if tags.get("type") != "restriction":
                        continue
                    relation_id = int(element.attrib["id"])
                    members = [
                        {"type": member.attrib["type"], "ref": int(member.attrib["ref"]),
                         "role": member.attrib.get("role", "")}
                        for member in element.iter("member")
                    ]
                    relations[relation_id] = {
                        "type": "relation", "id": relation_id,
                        "members": members, "tags": tags,
                        "version": element.attrib.get("version"),
                        "timestamp": element.attrib.get("timestamp"),
                    }
            except (ElementTree.ParseError, KeyError, ValueError) as error:
                raise OSMLoadError("format", "Ответ OSM с ограничениями имеет неверный формат") from error
    return tuple(relations[key] for key in sorted(relations))


def _read_cache(area: AreaConfig, request: dict, paths: tuple[Path, Path, Path]) -> RoadData | None:
    graph_path, relations_path, metadata_path = paths
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (metadata["request"] != request or not graph_path.exists() or not relations_path.exists()
                or metadata["graph_sha256"] != _hash_file(graph_path)
                or metadata["relations_sha256"] != _hash_file(relations_path)):
            raise ValueError("cache manifest mismatch")
        graph = _validate_graph(ox.load_graphml(filepath=graph_path), area)
        relations = json.loads(relations_path.read_text(encoding="utf-8"))
        if not isinstance(relations, list):
            raise ValueError("invalid cached relations")
        return RoadData(graph, tuple(relations), metadata, True)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise OSMLoadError("format", "Локальный кэш OSM повреждён; требуется явное обновление") from error


def _load_road_network(
    area: AreaConfig, *, cache_dir: Path = DEFAULT_CACHE_DIR, refresh: bool = False,
) -> RoadData:
    """Выполнить загрузку при удерживаемой блокировке общего состояния OSMnx."""

    cache_dir = Path(cache_dir)
    request = _request_description(area)
    paths = _cache_paths(cache_dir, request)
    if not refresh:
        cached = _read_cache(area, request, paths)
        if cached is not None:
            return cached

    graph = _fetch_graph(area, cache_dir)
    relations = _fetch_relations(area)
    metadata = {
        "request": request,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenStreetMap: drive graph via OSMnx/Overpass; restriction relations via OSM API v0.6/map",
        "relation_count": len(relations),
        **_graph_summary(graph),
    }
    cache_dir.mkdir(parents=True, exist_ok=True)
    graph_path, relations_path, metadata_path = paths
    graph_tmp = graph_path.with_suffix(".graphml.tmp")
    relations_tmp = relations_path.with_suffix(".json.tmp")
    metadata_tmp = metadata_path.with_suffix(".json.tmp")
    try:
        ox.save_graphml(graph, filepath=graph_tmp)
        relations_tmp.write_text(json.dumps(relations, ensure_ascii=False), encoding="utf-8")
        metadata["graph_sha256"] = _hash_file(graph_tmp)
        metadata["relations_sha256"] = _hash_file(relations_tmp)
        metadata_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(graph_tmp, graph_path)
        os.replace(relations_tmp, relations_path)
        os.replace(metadata_tmp, metadata_path)
    except (OSError, TypeError, ValueError) as error:
        raise OSMLoadError("format", f"Не удалось сохранить кэш OSM: {error}") from error
    return RoadData(graph, relations, metadata, False)


def load_road_network(
    area: AreaConfig, *, cache_dir: Path = DEFAULT_CACHE_DIR, refresh: bool = False,
) -> RoadData:
    """Загрузить drive-граф и отношения, не допуская конкурирующей записи кэша."""

    with _LOAD_LOCK:
        return _load_road_network(area, cache_dir=cache_dir, refresh=refresh)
