"""Мутация A на искусственном графе с управляемыми весами."""

from random import Random
from dataclasses import replace

from algorithm.genetic.mutation import MutationPolicy, mutate_subroute
from algorithm.genetic.validation import validate_route
from algorithm.models import Individual, Route, RouteContext


class FixedPairRandom(Random):
    def __init__(self, pair):
        super().__init__(0)
        self.pair = pair

    def sample(self, population, k):
        assert k == 2
        assert all(index in population for index in self.pair)
        return list(self.pair)


def favor_three(edge, attributes, rng):
    return 100 if edge[0:2] == (1, 2) else attributes["length"]


def test_subroute_endpoints(route_context: RouteContext) -> None:
    parent = Individual(Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))), 3)
    outcome = mutate_subroute(parent, route_context, FixedPairRandom((0, 2)), MutationPolicy(favor_three))
    assert outcome.route == Route(((1, 3, 0), (3, 5, 0), (5, 6, 0)))
    assert outcome.attempted and outcome.changed
    assert validate_route(outcome.route, route_context).valid
    assert parent.route == Route(((1, 2, 0), (2, 5, 0), (5, 6, 0)))


def test_subroute_boundaries(route_context: RouteContext) -> None:
    parent = Individual(Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))), 3)
    def favor_detour(edge, attributes, rng):
        return 100 if edge == (2, 5, 0) else attributes["length"]
    outcome = mutate_subroute(parent, route_context, FixedPairRandom((1, 2)), MutationPolicy(favor_detour))
    assert outcome.route == parent.route
    assert outcome.reason == "no_change" and not outcome.changed


def test_subroute_no_alternative(route_context: RouteContext) -> None:
    parent = Individual(Route(((1, 4, 0), (4, 6, 0))), 5)
    outcome = mutate_subroute(parent, route_context, FixedPairRandom((1, 2)), MutationPolicy(favor_three))
    assert outcome.route == parent.route
    assert outcome.reason == "no_change" and outcome.attempted


def test_subroute_seed(route_context: RouteContext) -> None:
    parent = Individual(Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))), 3)
    first = mutate_subroute(parent, route_context, Random(42), MutationPolicy(favor_three))
    second = mutate_subroute(parent, route_context, Random(42), MutationPolicy(favor_three))
    assert first == second


def test_subroute_invalid_candidate_falls_back(route_context: RouteContext) -> None:
    graph = route_context.graph.copy()
    graph.add_edge(2, 1, key=0, length=1, car_access=True)
    context = replace(route_context, graph=graph)
    parent = Individual(Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))), 3)
    def avoid_direct(edge, attributes, rng):
        return 100 if edge == (2, 5, 0) else attributes["length"]
    outcome = mutate_subroute(parent, context, FixedPairRandom((1, 2)), MutationPolicy(avoid_direct))
    assert outcome.route == parent.route
    assert outcome.reason == "invalid_route" and outcome.rejected_count == 1
    assert outcome.attempted and not outcome.changed
