"""Общий контракт для запретов пар направленных рёбер."""

from dataclasses import dataclass

from algorithm.models import EdgeId


@dataclass(frozen=True, slots=True)
class TurnRules:
    """Запрещённые переходы; импорт OSM-отношений появится в T02.04."""

    forbidden: frozenset[tuple[EdgeId, EdgeId]] = frozenset()


def is_turn_allowed(in_edge: EdgeId, out_edge: EdgeId, rules: TurnRules) -> bool:
    """Проверить именно заданную пару полных EdgeId."""

    return (in_edge, out_edge) not in rules.forbidden
