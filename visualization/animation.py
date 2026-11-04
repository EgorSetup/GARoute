"""Состояние воспроизведения уже рассчитанной истории."""

from dataclasses import dataclass
from math import floor, isfinite
from typing import Sequence

from algorithm.genetic.evolution import GenerationSnapshot
from algorithm.models import Route


@dataclass(slots=True)
class PlaybackState:
    """Пустое состояние до появления сохранённых поколений."""

    generation_count: int = 0
    generation_index: int = 0
    playing: bool = False
    speed: float = 1.0
    elapsed_s: float = 0.0

    @property
    def has_history(self) -> bool:
        """Разрешать перемотку лишь при реальной истории."""

        return self.generation_count > 0


@dataclass(frozen=True, slots=True)
class Frame:
    """Ссылка на уже рассчитанный неизменяемый снимок поколения."""

    index: int
    snapshot: GenerationSnapshot
    previous: GenerationSnapshot | None = None


ORIGIN_COLORS = {
    "initial": "#3c7892",
    "elite": "#25834e",
    "crossover": "#2563b9",
    "mutation": "#8b43b5",
    "copy": "#67849a",
    "unknown": "#67849a",
    "retired": "#99a4ad",
    "best": "#d6a422",
}


@dataclass(frozen=True, slots=True)
class StyledRoute:
    """Один слой фактической особи с сохранённым происхождением."""

    route: Route
    origin: str
    color: str
    weight: int
    opacity: float
    dashed: bool = False


def frame_layers(frame: Frame) -> tuple[StyledRoute, ...]:
    """Подготовить слои принятых и выбывших особей без геометрий отказов."""

    snapshot = frame.snapshot
    sources = {
        event.child_index: event.origin or "unknown"
        for event in snapshot.events
        if event.accepted is True and event.child_index is not None
    }
    layers: list[StyledRoute] = []
    previous = frame.previous
    if previous is not None:
        for event in snapshot.events:
            if (event.kind == "retired" and event.parents
                    and 0 <= event.parents[0] < len(previous.population)):
                route = previous.population[event.parents[0]].route
                layers.append(StyledRoute(route, "retired", ORIGIN_COLORS["retired"], 3, 0.45, True))
    for index, person in enumerate(snapshot.population):
        if index == snapshot.best_index:
            continue
        origin = sources.get(index, "unknown")
        layers.append(StyledRoute(person.route, origin, ORIGIN_COLORS.get(origin, ORIGIN_COLORS["unknown"]), 5, 0.65))
    best = snapshot.population[snapshot.best_index]
    best_origin = sources.get(snapshot.best_index, "unknown")
    layers.append(StyledRoute(best.route, best_origin,
                              ORIGIN_COLORS.get(best_origin, ORIGIN_COLORS["unknown"]), 10, 0.85))
    layers.append(StyledRoute(best.route, "best", ORIGIN_COLORS["best"], 5, 1.0))
    return tuple(layers)


def load_history(controller: PlaybackState, history: Sequence[GenerationSnapshot]) -> None:
    """Заменить историю, сбросив старый таймер к паузе на P0."""

    controller.generation_count = len(history)
    controller.generation_index = 0
    controller.playing = False
    controller.elapsed_s = 0.0


def seek(history: Sequence[GenerationSnapshot], generation_index: int) -> Frame:
    """Прочитать кадр без вызова алгоритма и генератора случайных чисел."""

    if type(generation_index) is not int or not 0 <= generation_index < len(history):
        raise IndexError("generation index out of range")
    return Frame(generation_index, history[generation_index],
                 history[generation_index - 1] if generation_index > 0 else None)


def set_index(controller: PlaybackState, history: Sequence[GenerationSnapshot],
              generation_index: int) -> Frame:
    frame = seek(history, generation_index)
    controller.generation_count = len(history)
    controller.generation_index = generation_index
    controller.elapsed_s = 0.0
    controller.playing = False
    return frame


def play(controller: PlaybackState) -> None:
    if not controller.has_history:
        return
    if controller.generation_index >= controller.generation_count - 1:
        controller.generation_index = 0
    controller.elapsed_s = 0.0
    controller.playing = controller.generation_count > 1


def pause(controller: PlaybackState) -> None:
    controller.playing = False
    controller.elapsed_s = 0.0


def set_playback_speed(controller: PlaybackState, speed: float) -> None:
    if type(speed) not in (int, float) or not isfinite(speed) or speed <= 0:
        raise ValueError("playback speed must be positive and finite")
    controller.speed = float(speed)


def advance(controller: PlaybackState, elapsed_s: float) -> bool:
    """Продвинуть показ на целые поколения; возвратить факт смены кадра."""

    if type(elapsed_s) not in (int, float) or not isfinite(elapsed_s) or elapsed_s < 0:
        raise ValueError("elapsed time must be finite and nonnegative")
    if not controller.playing:
        return False
    controller.elapsed_s += elapsed_s * controller.speed
    steps = floor(controller.elapsed_s)
    if not steps:
        return False
    previous = controller.generation_index
    controller.generation_index = min(controller.generation_count - 1, previous + steps)
    controller.elapsed_s -= steps
    if controller.generation_index == controller.generation_count - 1:
        pause(controller)
    return controller.generation_index != previous
