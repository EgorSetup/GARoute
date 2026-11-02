"""Полные снимки поколений и счётчики событий без геометрий отказов."""

from dataclasses import replace
from random import Random

from algorithm.genetic.evolution import (
    GenerationContext,
    build_next_generation,
    snapshot_generation,
    summarize_events,
)
from algorithm.genetic.mutation import MutationPolicy
from algorithm.models import GAConfig, GenerationEvent, Individual, Route, RouteContext, RouteMetrics


def population_with_metrics(context: RouteContext) -> list[Individual]:
    routes = (
        Route(((1, 2, 0), (2, 5, 0), (5, 6, 0))),
        Route(((1, 3, 0), (3, 5, 0), (5, 6, 0))),
        Route(((1, 4, 0), (4, 6, 0))),
    )
    answer = []
    for route in routes:
        length = sum(context.graph[u][v][key]["length"] for u, v, key in route.edges)
        answer.append(Individual(route, length, RouteMetrics(length, length * 2, 0)))
    return answer


def accepted_events(n: int) -> list[GenerationEvent]:
    return [GenerationEvent("elite", (i,), i, accepted=True, origin="elite") for i in range(n)]


def test_snapshot_independent(route_context: RouteContext) -> None:
    source = population_with_metrics(route_context)
    mutable_edges = list(source[0].route.edges)
    source[0] = Individual(Route(mutable_edges), 3, RouteMetrics(3, 6, 0))
    events = accepted_events(3)
    snapshot = snapshot_generation(source, summarize_events(events), events)
    mutable_edges.clear()
    source.clear()
    events.clear()
    assert len(snapshot.population) == 3
    assert snapshot.population[0].route.edges == ((1, 2, 0), (2, 5, 0), (5, 6, 0))
    assert len(snapshot.events) == 3


def test_full_population_history(route_context: RouteContext) -> None:
    source = population_with_metrics(route_context)
    events = accepted_events(3)
    snapshot = snapshot_generation(source, summarize_events(events), events)
    assert len(snapshot.population) == 3 and snapshot.best_index == 0
    assert [person.j for person in snapshot.population] == [3, 3, 5]
    assert [(person.metrics.length_m, person.metrics.time_s, person.metrics.complexity)
            for person in snapshot.population] == [(3, 6, 0), (3, 6, 0), (5, 10, 0)]


def test_event_counts(route_context: RouteContext) -> None:
    current = population_with_metrics(route_context) + [population_with_metrics(route_context)[0]]
    config = replace(GAConfig(), population_size=4, elite_count=1, max_copies=2,
                     tournament_size=2, attempts_per_slot=30,
                     crossover_probability=1, mutation_probability=1)
    def evaluate(route):
        length = sum(route_context.graph[u][v][key]["length"] for u, v, key in route.edges)
        return Individual(route, length, RouteMetrics(length, length * 2, 0))
    result = build_next_generation(
        current, GenerationContext(route_context, config,
                                   MutationPolicy(lambda edge, attrs, rng: attrs["length"])),
        Random(5), evaluate,
    )
    assert result.error is None
    stats = summarize_events(result.events)
    assert stats.accepted == len(result.population) == 4
    assert stats.crossover_attempts == result.crossover_attempts
    assert stats.mutation_attempts == result.mutation_attempts
    assert stats.retired == 3
    snapshot = snapshot_generation(result.population, stats, result.events)
    assert len(snapshot.population) == 4
    assert any(event.kind == "admission" and event.origin for event in snapshot.events)


def test_rejected_geometry_not_stored() -> None:
    event = GenerationEvent("admission", (0,), accepted=False,
                            reasons=("forbidden_turn",), origin=None)
    assert not hasattr(event, "route") and not hasattr(event, "geometry")
    assert summarize_events([event]).rejected_candidates == 1


def test_noop_not_success() -> None:
    events = [GenerationEvent("mutation", (0,), changed=False, reasons=("no_change",)),
              GenerationEvent("admission", (0,), 0, accepted=True, origin="copy")]
    stats = summarize_events(events)
    assert stats.mutation_attempts == 1 and stats.mutation_changes == 0
    assert stats.operator_failed == 1 and stats.accepted == 1
    assert stats.rejected_candidates == 0


def test_rejection_multiple_reasons_counted_once() -> None:
    events = [GenerationEvent("admission", (0,), accepted=False,
                              reasons=("cycle", "forbidden_turn", "cycle"))]
    stats = summarize_events(events)
    assert stats.rejected_candidates == 1
    assert dict(stats.rejection_reasons) == {"cycle": 1, "forbidden_turn": 1}
