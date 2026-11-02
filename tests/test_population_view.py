"""Геометрия P0 и честная диагностика её генерации."""

from algorithm.genetic.initialization import PopulationResult
from algorithm.models import RoadEdge, Route
from data.road_graph import NormalizedRoadData
from interface.map_view import render_population, route_geometry
from interface.results import population_status_text


def test_route_geometry_uses_way_shape() -> None:
    ids = ((1, 2, 0), (2, 3, 0))
    shapes = (((30.0, 60.0), (30.01, 60.02), (30.02, 60.0)),
              ((30.02, 60.0), (30.025, 60.015), (30.03, 60.0)))
    edges = {edge_id: RoadEdge(edge_id, shape, "test", 100, "test", (1,), {},
                               60, "observed", "maxspeed", None, 60)
             for edge_id, shape in zip(ids, shapes)}
    roads = NormalizedRoadData(None, edges, {}, (), (), {})
    assert route_geometry(Route(ids), roads) == [
        [30.0, 60.0], [30.01, 60.02], [30.02, 60.0],
        [30.025, 60.015], [30.03, 60.0],
    ]


def test_population_status_distinguishes_failure() -> None:
    failed = PopulationResult((), 2, 12, (("duplicate_limit", 10),), "attempts_exhausted")
    assert "не сформирована" in population_status_text(failed)
    assert "duplicate_limit: 10" in population_status_text(failed)
    ready = PopulationResult((Route(((1, 2, 0),)),), 1, 1, ())
    assert "не эволюция" in population_status_text(ready)


def test_render_replaces_layers() -> None:
    edge_id = (1, 2, 0)
    edge = RoadEdge(edge_id, ((30.0, 60.0), (30.01, 60.02), (30.02, 60.0)),
                    "test", 100, "test", (1,), {}, 60, "observed", "maxspeed", None, 60)
    roads = NormalizedRoadData(None, {edge_id: edge}, {}, (), (), {})

    class FakeMap:
        def __init__(self):
            self.layers = []

        def generic_layer(self, **kwargs):
            self.layers.append(kwargs)
            return kwargs

        def remove_layer(self, layer):
            self.layers.remove(layer)

    test_map = FakeMap()
    first = render_population(test_map, (Route((edge_id,)),), roads)
    second = render_population(test_map, (Route((edge_id,)),), roads, first)
    assert len(first) == len(second) == len(test_map.layers) == 1
    assert test_map.layers[0]["args"][0]["geometry"]["coordinates"][1] == [30.01, 60.02]
