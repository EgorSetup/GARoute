"""Рабочая модель замедления по уровню пробки Q02 (VERIFY)."""

from math import isfinite
from typing import Mapping

from config import TRAFFIC_SPEED_FACTORS


def speed_factor(level: int, factors: Mapping[int, float] = TRAFFIC_SPEED_FACTORS) -> float:
    """Вернуть k(n) для 1–10; закрытие 11 обрабатывать до деления."""

    if type(level) is not int or not 1 <= level <= 11:
        raise ValueError(f"invalid traffic level: {level!r}")
    if level == 11:
        raise ValueError("traffic level 11 closes the edge")
    factor = factors.get(level)
    if type(factor) not in (int, float) or not isfinite(factor) or factor <= 0 or factor > 1:
        raise ValueError(f"invalid speed factor for level {level}")
    return float(factor)
