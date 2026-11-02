"""Детерминированные проверки операторов на искусственных маршрутах."""

from random import Random

import pytest

from algorithm.genetic.selection import tournament_select
from algorithm.models import Individual, Route


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
