"""Карта показывает происхождение только реально сохранённых маршрутов."""

import networkx as nx

from algorithm.genetic.evolution import GenerationSnapshot, OperationStats
from algorithm.models import GenerationEvent, Individual, RoadEdge, Route, RouteMetrics
from data.road_graph import NormalizedRoadData
from interface.map_view import clear_run_layers, render_frame
from visualization.animation import Frame, ORIGIN_COLORS, frame_layers


FIRST = (1, 2, 0)
SECOND = (2, 3, 0)


def roads() -> NormalizedRoadData:
    graph = nx.MultiDiGraph()
    edges = {}
    for edge_id, start, end in (
        (FIRST, (30.37, 60.01), (30.38, 60.01)),
        (SECOND, (30.38, 60.01), (30.39, 60.01)),
    ):
        graph.add_edge(*edge_id)
        edges[edge_id] = RoadEdge(
            edge_id, (start, end), "fixture", 100, "fixture", (101,), {},
            36, "observed", "maxspeed", None, "36",
        )
    return NormalizedRoadData(graph, edges, {}, (), (), {})


def frame() -> Frame:
    stats = OperationStats(0, 0, 0, 0, 0, 2, 1, 1, (("invalid", 1),))
    first = Individual(Route((FIRST,)), 1, RouteMetrics(100, 10, 0))
    second = Individual(Route((SECOND,)), 2, RouteMetrics(100, 10, 0))
    previous = GenerationSnapshot((first, second), 0, stats, (), 0)
    events = (
        GenerationEvent("admission", (1,), 0, True, True, (), "mutation"),
        GenerationEvent("elite", (0,), 1, False, True, (), "elite"),
        GenerationEvent("retired", (1,), origin="retired"),
        GenerationEvent("admission", (0,), accepted=False, reasons=("invalid",)),
    )
    current = GenerationSnapshot((first, second), 0, stats, events, 1)
    return Frame(1, current, previous)


class FakeMap:
    def __init__(self) -> None:
        self.created = []
        self.removed = []

    def generic_layer(self, *, name, args):
        layer = (name, args)
        self.created.append(layer)
        return layer

    def remove_layer(self, layer) -> None:
        self.removed.append(layer)


def test_best_mutant_style() -> None:
    layers = frame_layers(frame())
    assert [layer.origin for layer in layers] == ["retired", "elite", "mutation", "best"]
    assert layers[0].dashed and layers[0].color == ORIGIN_COLORS["retired"]
    assert layers[-2].route == layers[-1].route
    assert layers[-2].color == ORIGIN_COLORS["mutation"]
    assert layers[-1].color == ORIGIN_COLORS["best"]
    assert len(layers) == 4  # Отбракованный кандидат не имеет геометрии.


def test_layer_reuse() -> None:
    map_element = FakeMap()
    active = []
    current = frame()
    render_frame(map_element, current, roads(), active)
    assert len(active) == 4 and map_element.removed == []
    render_frame(map_element, current, roads(), active)
    assert len(active) == 4 and len(map_element.removed) == 4
    assert len(map_element.created) == 8
    assert map_element.created[0][1][1]["style"]["dashArray"] == "6 5"


def test_clear_layers() -> None:
    map_element = FakeMap()
    active = render_frame(map_element, frame(), roads(), [])
    clear_run_layers(map_element, active)
    assert active == [] and len(map_element.removed) == 4
    clear_run_layers(map_element, active)
    assert len(map_element.removed) == 4
