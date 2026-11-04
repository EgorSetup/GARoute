"""Состояние воспроизведения уже рассчитанной истории."""

from dataclasses import dataclass
from math import floor, isfinite
from typing import Sequence

from algorithm.genetic.evolution import GenerationSnapshot


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
    return Frame(generation_index, history[generation_index])


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
