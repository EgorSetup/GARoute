"""Состояние воспроизведения уже рассчитанной истории."""

from dataclasses import dataclass


@dataclass(slots=True)
class PlaybackState:
    """Пустое состояние до появления сохранённых поколений."""

    generation_count: int = 0
    generation_index: int = 0
    playing: bool = False
    speed: float = 1.0

    @property
    def has_history(self) -> bool:
        """Разрешать перемотку лишь при реальной истории."""

        return self.generation_count > 0
