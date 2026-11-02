"""Детерминированные проверки операторов на искусственных маршрутах."""

from random import Random
from dataclasses import replace

import pytest

from algorithm.genetic.selection import tournament_select
from algorithm.genetic.crossover import crossover
from algorithm.models import Individual, Route, RouteContext
from data.restrictions import TurnRules


def population() -> tuple[Individual, ...]:
    return (
        Individual(Route(((1, 2, 0),)), 9),
        Individual(Route(((1, 2, 1),)), 2),
        Individual(Route(((1, 3, 0),)), 5),
        Individual(Route(((1, 4, 0),)), 7),
    )


def test_tournament_minimum() -> None:
    members = population()
    assert tournament_select(members, 4, Random(0)) == members[1]
    assert tournament_select(members[:1], 1, Random(0)) == members[0]
    assert members == population()


def test_tournament_seed() -> None:
    members = population()
    first = tournament_select(members, 3, Random(42))
    second = tournament_select(members, 3, Random(42))
    assert first == second
    tied = (Individual(members[0].route, 1), Individual(members[1].route, 1))
    expected = Random(7).sample(list(tied), 2)[0]
    assert tournament_select(tied, 2, Random(7)) == expected


@pytest.mark.parametrize("size", (0, -1, 5, 1.5, True))
def test_tournament_invalid_size(size) -> None:
    with pytest.raises(ValueError, match="tournament size"):
        tournament_select(population(), size, Random(0))


def test_tournament_invalid_cost() -> None:
    members = population()
    invalid = members + (Individual(Route(((1, 5, 0),)), float("nan")),)
    with pytest.raises(ValueError, match="finite"):
        tournament_select(invalid, 3, Random(0))
    with pytest.raises(ValueError, match="tournament size"):
        tournament_select((), 1, Random(0))


def parents_for_splice(context: RouteContext):
    graph = context.graph.copy()
    graph.add_edge(5, 4, key=0, length=1, car_access=True)
    scoped = replace(context, graph=graph)
    a = Individual(Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))), 3)
    b = Individual(Route(((1, 3, 0), (3, 5, 0), (5, 4, 0), (4, 6, 0))), 4)
    return scoped, a, b


def test_common_vertex(route_context: RouteContext) -> None:
    context, a, b = parents_for_splice(route_context)
    outcome = crossover(a, b, context, Random(1))
    assert outcome.route == Route(((1, 2, 0), (2, 5, 0), (5, 4, 0), (4, 6, 0)))
    assert outcome.attempted and outcome.changed and outcome.rejected_count == 0


def test_no_common_internal_vertex(route_context: RouteContext) -> None:
    a = Individual(Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))), 3)
    b = Individual(Route(((1, 4, 0), (4, 6, 0))), 5)
    outcome = crossover(a, b, route_context, Random(0))
    assert outcome.route == a.route
    assert not outcome.attempted and not outcome.changed
    assert outcome.reason == "no_common_internal_vertex"


def test_forbidden_splice(route_context: RouteContext) -> None:
    context, a, b = parents_for_splice(route_context)
    rules = TurnRules(context.turn_rules.forbidden | {((2, 5, 0), (5, 4, 0))})
    outcome = crossover(a, b, replace(context, turn_rules=rules), Random(1))
    assert outcome.route == a.route
    assert outcome.attempted and not outcome.changed and outcome.rejected_count == 1
    assert outcome.reason == "all_splices_invalid"


def test_crossover_cycle(route_context: RouteContext) -> None:
    graph = route_context.graph.copy()
    graph.add_edge(5, 2, key=0, length=1, car_access=True)
    graph.add_edge(2, 6, key=0, length=1, car_access=True)
    rules = TurnRules(route_context.turn_rules.forbidden | {((1, 2, 0), (2, 6, 0))})
    context = replace(route_context, graph=graph, turn_rules=rules)
    a = Individual(Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))), 3)
    b = Individual(Route(((1, 3, 0), (3, 5, 0), (5, 2, 0), (2, 6, 0))), 4)
    outcome = crossover(a, b, context, Random(0))
    assert outcome.route == a.route
    assert outcome.rejected_count == 2 and not outcome.changed


def test_parents_unchanged(route_context: RouteContext) -> None:
    context, a, b = parents_for_splice(route_context)
    before = (a.route.edges, b.route.edges)
    crossover(a, b, context, Random(0))
    assert (a.route.edges, b.route.edges) == before
