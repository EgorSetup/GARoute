"""Отбор родителей без изменения популяции."""

from math import isfinite
from random import Random
from typing import Sequence

from algorithm.models import Individual


def ranking_key(individual: Individual) -> tuple[bool, float]:
    """Сначала допустимые маршруты, внутри группы — исходная стоимость J."""

    return not individual.eligible, individual.j


def tournament_select(
    population: Sequence[Individual], tournament_size: int, rng: Random,
) -> Individual:
    """Выбрать минимум J; равенство разрешить порядком случайной выборки."""

    if type(tournament_size) is not int or not 1 <= tournament_size <= len(population):
        raise ValueError("tournament size must be between 1 and population size")
    if any(type(individual.j) not in (int, float) or not isfinite(individual.j) for individual in population):
        raise ValueError("individual J must be finite")
    participants = rng.sample(list(population), tournament_size)
    return min(participants, key=ranking_key)
