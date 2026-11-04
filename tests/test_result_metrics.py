"""Карточка P0 показывает единицы и не подставляет неизвестное время/J."""

from algorithm.evaluation.fitness import evaluate_cost, resolve_preset
from algorithm.evaluation.normalization import NormalizationScales
from algorithm.evaluation.penalties import (
    PRESENT, PenaltyEvent, PenaltyEvents, calculate_penalties,
)
from algorithm.models import RouteMetrics
from config import working_penalty_defaults
from interface.controls import RoutePreview
from interface.results import render_route_metrics


def test_manual_route_cost_and_display():
    events = PenaltyEvents((
        PenaltyEvent("P02", PRESENT, "osm:node:1", 2, "turn"),
        PenaltyEvent("P12", PRESENT, "osm:way:2", 3.5, "lanes+length"),
    ))
    breakdown = calculate_penalties(events, working_penalty_defaults())
    metrics = RouteMetrics(350, 30, breakdown.total)
    scales = NormalizationScales(30, 350, 11.75)
    cost = evaluate_cost(metrics, resolve_preset("custom", (0.2, 0.3, 0.5)), scales)
    assert breakdown.total == 11.75
    assert cost == 1
    lines = render_route_metrics(RoutePreview(350, 30), breakdown, cost, scales)
    assert "L = 350.00 м" in lines
    assert "T = 30.00 с" in lines
    assert "C_observed = 11.75 балла" in lines
    assert "J = 1.0000 (безразмерная стоимость; меньше — лучше)" in lines
    assert any("Tref=30.00 с" in line and "Lref=350.00 м" in line for line in lines)


def test_unknown_time_displays_no_cost():
    breakdown = calculate_penalties(PenaltyEvents(()), working_penalty_defaults())
    lines = render_route_metrics(RoutePreview(350, None, "unknown or invalid speed"),
                                 breakdown, None, None)
    assert "L = 350.00 м" in lines
    assert "T недоступно: unknown or invalid speed" in lines
    assert "J недоступна" in lines
    assert not any(line.startswith("J =") for line in lines)
