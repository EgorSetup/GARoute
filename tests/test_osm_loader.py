"""Границы, кэш и честные ошибки загрузки OSM без сетевых запросов."""

import networkx as nx
from pyproj import Geod
import pytest
import requests

from config import AreaConfig, GAROUTE_AREA
from data import osm_loader
from interface.map_view import MapHandle, show_load_error


def sample_graph() -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="epsg:4326")
    graph.add_node(1, x=30.38, y=60.01)
    graph.add_node(2, x=30.39, y=60.01)
    graph.add_edge(1, 2, key=0, osmid=42, length=100.0, highway="residential")
    graph.add_edge(2, 1, key=0, osmid=42, length=100.0, highway="residential")
    return graph


def test_area_expansion_is_explicit() -> None:
    area = GAROUTE_AREA
    assert area.bbox == (30.355, 59.995, 30.415, 60.025)
    assert area.network_type == "drive"
    expanded = area.expanded()
    assert area.bbox == (30.355, 59.995, 30.415, 60.025)
    assert expanded.west < area.west and expanded.east > area.east
    assert expanded.south < area.south and expanded.north > area.north
    geod = Geod(ellps="WGS84")
    assert geod.inv(area.west, area.north, expanded.west, area.north)[2] == pytest.approx(1000, abs=3)
    with pytest.raises(ValueError, match="WGS84"):
        AreaConfig(59.0, 60.0, 30.4, 30.3)


def test_cache_metadata(monkeypatch, tmp_path) -> None:
    calls = {"graph": 0, "relations": 0}
    def graph(area, cache_dir):
        calls["graph"] += 1
        return sample_graph()
    def relations(area):
        calls["relations"] += 1
        return ({"type": "relation", "id": 7, "members": [],
                 "tags": {"type": "restriction", "restriction": "no_left_turn"}},)
    monkeypatch.setattr(osm_loader, "_fetch_graph", graph)
    monkeypatch.setattr(osm_loader, "_fetch_relations", relations)
    first = osm_loader.load_road_network(GAROUTE_AREA, cache_dir=tmp_path)
    assert not first.cache_hit and first.metadata["nodes"] == 2
    assert first.metadata["edges"] == 2 and first.metadata["relation_count"] == 1
    assert first.metadata["request"]["bbox_wsen"] == list(GAROUTE_AREA.bbox)
    assert first.metadata["request"]["simplify"] is False
    assert first.metadata["fetched_at_utc"] and first.metadata["source"]
    second = osm_loader.load_road_network(GAROUTE_AREA, cache_dir=tmp_path)
    assert second.cache_hit and second.graph.number_of_edges() == 2
    assert second.raw_relations[0]["id"] == 7
    assert calls == {"graph": 1, "relations": 1}


def test_network_error(monkeypatch, tmp_path) -> None:
    def unavailable(*args, **kwargs):
        raise requests.Timeout("timed out")
    monkeypatch.setattr(osm_loader.ox, "graph_from_bbox", unavailable)
    with pytest.raises(osm_loader.OSMLoadError) as error:
        osm_loader.load_road_network(GAROUTE_AREA, cache_dir=tmp_path)
    assert error.value.kind == "network"


def test_empty_area(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(osm_loader.ox, "graph_from_bbox", lambda *args, **kwargs: nx.MultiDiGraph())
    with pytest.raises(osm_loader.OSMLoadError) as error:
        osm_loader.load_road_network(GAROUTE_AREA, cache_dir=tmp_path)
    assert error.value.kind == "empty"


def test_relation_format_error(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(osm_loader, "_fetch_graph", lambda area, cache: sample_graph())
    class Response:
        content = b"not XML"
        def raise_for_status(self):
            pass
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(osm_loader.requests, "Session", Session)
    with pytest.raises(osm_loader.OSMLoadError) as error:
        osm_loader.load_road_network(GAROUTE_AREA, cache_dir=tmp_path)
    assert error.value.kind == "format"


def test_raw_relations_are_deduplicated_across_tiles(monkeypatch) -> None:
    class Response:
        content = (b'<osm><relation id="7" version="2" timestamp="2026-01-01T00:00:00Z">'
                   b'<member type="way" ref="123" role="from"/>'
                   b'<member type="node" ref="456" role="via"/>'
                   b'<tag k="type" v="restriction"/>'
                   b'<tag k="restriction" v="no_left_turn"/>'
                   b'</relation></osm>')
        def raise_for_status(self):
            pass
    class Session:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get(self, *args, **kwargs):
            return Response()
    monkeypatch.setattr(osm_loader.requests, "Session", Session)
    relations = osm_loader._fetch_relations(GAROUTE_AREA)
    assert len(relations) == 1
    assert relations[0]["members"][0] == {"type": "way", "ref": 123, "role": "from"}
    assert relations[0]["tags"]["restriction"] == "no_left_turn"


def test_error_visible_in_map_status() -> None:
    class Label:
        text = ""
        def set_text(self, text):
            self.text = text
    label = Label()
    show_load_error(MapHandle(None, label), osm_loader.OSMLoadError("network", "нет соединения"))
    assert label.text == "Ошибка сети: нет соединения"
