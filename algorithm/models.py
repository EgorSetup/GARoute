"""Общие структуры расчёта, не зависящие от интерфейса."""

from dataclasses import dataclass
from typing import Hashable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:
    from networkx import MultiDiGraph
    from data.restrictions import TurnRules


EdgeId = tuple[Hashable, Hashable, Hashable]


@dataclass(frozen=True, slots=True)
class RoadEdge:
    """Нормализованное ребро с исходными OSM-тегами и их происхождением."""

    edge_id: EdgeId
    geometry_lonlat: tuple[tuple[float, float], ...]
    geometry_source: str
    length_m: float
    length_source: str
    osm_way_ids: tuple[int, ...]
    raw_attributes: Mapping[str, object]
    speed_kph: float | None
    speed_status: str
    speed_source: str | None
    speed_rule: str | None
    raw_speed: object | None

    def tag_status(self, name: str) -> str:
        """Отличить подтверждённый тег от неизвестного признака."""

        return "observed" if name in self.raw_attributes and self.raw_attributes[name] is not None else "unknown"


@dataclass(frozen=True, slots=True)
class Route:
    """Хромосома: упорядоченные направленные рёбра с полными ключами."""

    edges: tuple[EdgeId, ...]


@dataclass(frozen=True, slots=True)
class RouteMetrics:
    """Три измерения маршрута; J хранится отдельно в Individual."""

    length_m: float
    time_s: float
    complexity: float


@dataclass(frozen=True, slots=True)
class EasyReferences:
    """Отдельные от нормализации минимумы для двух ограничений новичка."""

    length_min_m: float
    time_min_s: float


@dataclass(frozen=True, slots=True)
class EasyLimits:
    """Разрешённые доли превышения L_min и T_min, Q05 VERIFY."""

    length_delta: float
    time_delta: float


@dataclass(frozen=True, slots=True)
class EligibilityReport:
    """Результат жёсткого фильтра без изменения стоимости J."""

    eligible: bool
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Individual:
    """Маршрут с рассчитанной извне стоимостью J."""

    route: Route
    j: float
    metrics: RouteMetrics | None = None


@dataclass(frozen=True, slots=True)
class GenerationEvent:
    """Факт операции или приёма без геометрии отвергнутого кандидата."""

    kind: str
    parents: tuple[int, ...] = ()
    child_index: int | None = None
    changed: bool = False
    accepted: bool | None = None
    reasons: tuple[str, ...] = ()
    origin: str | None = None


@dataclass(frozen=True, slots=True)
class OperatorOutcome:
    """Результат оператора с разделением попытки, изменения и отказов."""

    route: Route
    attempted: bool
    changed: bool
    rejected_count: int = 0
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class RouteContext:
    """Неизменяемые для проверки условия одного поиска."""

    graph: "MultiDiGraph"
    start: Hashable
    end: Hashable
    turn_rules: "TurnRules"
    closed_edges: frozenset[EdgeId] = frozenset()


@dataclass(frozen=True, slots=True)
class RouteValidationReport:
    """Коды всех обнаруженных жёстких нарушений маршрута."""

    codes: tuple[str, ...] = ()

    @property
    def valid(self) -> bool:
        return not self.codes


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
    random_weight_strength: float | None = 1.0  # Q06-C VERIFY: 2^U(-s,s)
    stagnation_window: int | None = 10  # Q06-C VERIFY
    stagnation_enabled: bool = False


@dataclass(frozen=True, slots=True)
class FitnessConfig:
    """Параметры стоимости; неутверждённые числа остаются пустыми."""

    preset: str | None = None
    weights: tuple[float, float, float] | None = None  # T, L, C
    penalties: tuple[float, ...] | None = None  # P01–P15, рабочие Q04 VERIFY в default_config
    normalization_mode: str | None = None
    time_reference_s: float | None = None
    length_reference_m: float | None = None
    complexity_reference: float | None = None
    easy_length_delta: float | None = None  # Q05-C VERIFY, default_config задаёт 0.5
    easy_time_delta: float | None = None  # Q05-C VERIFY


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Снимок настроек расчёта для проверки перед запуском."""

    ga: GAConfig
    fitness: FitnessConfig


@dataclass(frozen=True, slots=True)
class ConditionKey:
    """Общие условия серии; пресет и вычисленные эталоны в ключ не входят."""

    graph_version: int
    graph_identity: int | None
    endpoints: tuple[Hashable | None, Hashable | None]
    forbidden_turns: frozenset[tuple[EdgeId, EdgeId]]
    traffic_levels: tuple[tuple[EdgeId, int], ...]
    segment_levels: tuple[tuple[str, int], ...]
    ga: GAConfig
    penalties: tuple[float, ...] | None
    normalization: tuple[str | None, float | None, float | None, float | None]
    easy_limits: tuple[float | None, float | None]
    custom_weights: tuple[float, float, float] | None


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
