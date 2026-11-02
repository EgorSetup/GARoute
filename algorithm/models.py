"""Общие структуры расчёта, не зависящие от интерфейса."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GAConfig:
    """Настройки генетического алгоритма с утверждёнными стартовыми значениями."""

    population_size: int = 10
    elite_count: int = 2
    crossover_probability: float = 0.8
    mutation_probability: float = 0.3
    max_generations: int = 50
    max_copies: int = 2
    attempts_per_slot: int = 100
    tournament_size: int = 3
    initial_length_factor: float = 3.0
    min_waypoints: int = 1
    max_waypoints: int = 4
    segment_length_m: float = 500.0
    mutation_method: str = "A"
    seed: int | None = None
    random_weight_strength: float | None = None  # OPEN Q06-C
    stagnation_window: int | None = None  # OPEN Q06-C


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Параметры стоимости; неутверждённые числа остаются пустыми."""

    preset: str | None = None
    weights: tuple[float, float, float] | None = None  # T, L, C
    penalties: tuple[float, ...] | None = None  # P01–P15, OPEN Q04-C
    normalization_mode: str | None = None
    time_reference_s: float | None = None
    length_reference_m: float | None = None
    complexity_reference: float | None = None
    easy_length_delta: float | None = None  # OPEN Q05-C
    easy_time_delta: float | None = None  # OPEN Q05-C


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Снимок настроек расчёта для проверки перед запуском."""

    ga: GAConfig
    fitness: FitnessConfig


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Ошибки чисел и открытые решения, препятствующие расчёту."""

    errors: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Вернуть True, если введённые значения структурно допустимы."""

        return not self.errors

    @property
    def ready_for_run(self) -> bool:
        """Запретить расчёт при ошибках или неразрешённых параметрах."""

        return self.is_valid and not self.unresolved
