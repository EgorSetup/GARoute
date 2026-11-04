"""Перемотка использует сохранённые поколения без запуска вычислений."""

from random import Random

import pytest

from algorithm.genetic.evolution import GenerationSnapshot, OperationStats
from algorithm.models import Individual, Route, RouteMetrics
from visualization.animation import (
    PlaybackState, advance, load_history, pause, play, seek, set_index,
    set_playback_speed,
)


def history(count: int = 3) -> tuple[GenerationSnapshot, ...]:
    stats = OperationStats(0, 0, 0, 0, 0, 1, 0, 0, ())
    return tuple(
        GenerationSnapshot(
            (Individual(Route(((index, index + 1, 0),)),
                        float(index + 1), RouteMetrics(100, 10, 0)),),
            0, stats, (), index,
        ) for index in range(count)
    )


def test_seek_bounds() -> None:
    snapshots = history()
    controller = PlaybackState()
    load_history(controller, snapshots)
    assert seek(snapshots, 0).snapshot is snapshots[0]
    assert set_index(controller, snapshots, 2).snapshot is snapshots[2]
    assert controller.generation_index == 2 and not controller.playing
    for index in (-1, 3, 1.5):
        with pytest.raises(IndexError):
            seek(snapshots, index)


def test_pause_resume_and_speed() -> None:
    controller = PlaybackState()
    snapshots = history()
    load_history(controller, snapshots)
    set_playback_speed(controller, 2)
    play(controller)
    assert not advance(controller, 0.24)
    assert advance(controller, 0.26)
    assert controller.generation_index == 1
    pause(controller)
    assert not advance(controller, 100)
    play(controller)
    assert advance(controller, 0.5)
    assert controller.generation_index == 2 and not controller.playing
    play(controller)
    assert controller.generation_index == 0 and controller.playing
    set_index(controller, snapshots, 1)
    assert not controller.playing
    with pytest.raises(ValueError):
        set_playback_speed(controller, 0)


def test_no_algorithm_calls(monkeypatch) -> None:
    from algorithm.evaluation import fitness
    from algorithm.genetic import evolution

    def forbidden(*args, **kwargs):
        raise AssertionError("calculation during playback")

    monkeypatch.setattr(evolution, "run_evolution", forbidden)
    monkeypatch.setattr(fitness, "evaluate_cost", forbidden)
    snapshots = history()
    controller = PlaybackState()
    load_history(controller, snapshots)
    for index in (2, 0, 1):
        assert set_index(controller, snapshots, index).snapshot is snapshots[index]


def test_no_rng_consumption() -> None:
    rng = Random(123)
    before = rng.getstate()
    snapshots = history()
    controller = PlaybackState()
    load_history(controller, snapshots)
    play(controller)
    advance(controller, 1.1)
    seek(snapshots, 2)
    assert rng.getstate() == before


def test_history_replacement() -> None:
    controller = PlaybackState()
    load_history(controller, history())
    play(controller)
    advance(controller, 1.1)
    assert controller.generation_index == 1 and controller.playing
    replacement = history(2)
    load_history(controller, replacement)
    assert controller.generation_count == 2
    assert controller.generation_index == 0 and not controller.playing
    assert not advance(controller, 10)
