"""Стартовая конфигурация и её проверка без зависимости от UI."""

from dataclasses import dataclass
from math import isclose, isfinite
from pyproj import Geod

from algorithm.models import AppConfig, FitnessConfig, GAConfig, ValidationReport


@dataclass(frozen=True, slots=True)
class AreaConfig:
    """Утверждённая прямоугольная область WGS84 и явный шаг расширения."""

    north: float
    south: float
    east: float
    west: float
    expansion_step_m: float = 1000.0
    network_type: str = "drive"

    def __post_init__(self) -> None:
        values = (self.north, self.south, self.east, self.west, self.expansion_step_m)
        if any(type(value) not in (int, float) or not isfinite(value) for value in values):
            raise ValueError("area coordinates and expansion must be finite numbers")
        if not (-90 <= self.south < self.north <= 90 and -180 <= self.west < self.east <= 180):
            raise ValueError("invalid WGS84 area bounds")
        if self.expansion_step_m <= 0 or self.network_type != "drive":
            raise ValueError("area requires positive expansion and drive network")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        """Формат OSMnx: west, south, east, north."""

        return (self.west, self.south, self.east, self.north)

    def expanded(self) -> "AreaConfig":
        """Явно добавить около 1 км с каждой стороны; при загрузке не вызывается."""

        geod = Geod(ellps="WGS84")
        west, _, _ = geod.fwd(self.west, self.north, 270, self.expansion_step_m)
        east, _, _ = geod.fwd(self.east, self.north, 90, self.expansion_step_m)
        _, north, _ = geod.fwd(self.west, self.north, 0, self.expansion_step_m)
        _, south, _ = geod.fwd(self.west, self.south, 180, self.expansion_step_m)
        return AreaConfig(north, south, east, west, self.expansion_step_m, self.network_type)


GAROUTE_AREA = AreaConfig(north=60.025, south=59.995, east=30.415, west=30.355)
OVERPASS_URL = "https://maps.mail.ru/osm/tools/overpass/api"

# Q02-C RESOLVED / VERIFY: рабочая таблица до сквозной проверки модели.
TRAFFIC_SPEED_FACTORS = {level: (11 - level) / 10 for level in range(1, 11)}


@dataclass(frozen=True, slots=True)
class PenaltyDefinition:
    """Позиция штрафа и статус доступности признака в текущей модели."""

    id: str
    name: str
    evidence_note: str
    feature_available: bool | None = None


_PENALTY_CATALOG = (
    PenaltyDefinition("P01", "Правый поворот", "Геометрия направления; угол и пороги ещё не согласованы."),
    PenaltyDefinition("P02", "Левый поворот", "Геометрия направления; отдельно проверять запрет поворота."),
    PenaltyDefinition("P03", "Разворот", "Геометрия и разрешённость; порог ещё не согласован."),
    PenaltyDefinition("P04", "Смена полосы", "Реальная смена полосы не наблюдается в графе уровня дорог; нужен согласованный признак.", False),
    PenaltyDefinition("P05", "Нерегулируемый перекрёсток", "Отсутствие тега светофора не доказывает нерегулируемость."),
    PenaltyDefinition("P06", "Кольцевой перекрёсток", "Нужен признак кольца и одно событие на проезд, не на ребро."),
    PenaltyDefinition("P07", "Светофор", "Нужна привязка объекта к направлению и исключение двойного учёта."),
    PenaltyDefinition("P08", "STOP", "Нужен подтверждённый знак, применимый к направлению движения."),
    PenaltyDefinition("P09", "«Уступи дорогу»", "Нужен подтверждённый знак, применимый к направлению движения."),
    PenaltyDefinition("P10", "Пересечение трамвайных путей", "Нужен проезд через пути на одном уровне, не мост."),
    PenaltyDefinition("P11", "Железнодорожный переезд", "Нужен подтверждённый объект; одно событие на переезд."),
    PenaltyDefinition("P12", "Многополосный участок", "Нужны подтверждённое число полос и единица начисления."),
    PenaltyDefinition("P13", "Узкая жилая улица", "Нужен признак узости; название улицы недостаточно."),
    PenaltyDefinition("P14", "Школьная зона", "Нужна связь зоны с дорогой; близости школы недостаточно."),
    PenaltyDefinition("P15", "Искусственная неровность", "Нужен подтверждённый объект без повторного подсчёта."),
)


def penalty_catalog() -> tuple[PenaltyDefinition, ...]:
    """Вернуть утверждённый состав P01–P15 без числовых коэффициентов."""

    return _PENALTY_CATALOG


def approved_penalty_defaults() -> tuple[float, ...] | None:
    """Вернуть утверждённые числа, когда Q04-C будет решён."""

    return None


def default_config() -> AppConfig:
    """Создать настройки только с утверждёнными стартовыми числами."""

    return AppConfig(ga=GAConfig(), fitness=FitnessConfig())


def _integer_at_least(value: object, minimum: int) -> bool:
    return type(value) is int and value >= minimum


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and isfinite(value)


def _positive_number(value: object) -> bool:
    return _finite_number(value) and value > 0


def _check_ga(config: GAConfig, errors: list[str], unresolved: list[str]) -> None:
    """Проверить размер популяции и параметры операторов."""

    if not _integer_at_least(config.population_size, 1):
        errors.append("population_size должен быть положительным целым")
    if (
        not _integer_at_least(config.elite_count, 0)
        or not _integer_at_least(config.population_size, 1)
        or config.elite_count >= config.population_size
    ):
        errors.append("elite_count должен удовлетворять 0 <= E < N")
    if (
        not _integer_at_least(config.tournament_size, 1)
        or not _integer_at_least(config.population_size, 1)
        or config.tournament_size > config.population_size
    ):
        errors.append("tournament_size должен удовлетворять 1 <= k <= N")

    for name in ("max_generations", "max_copies", "attempts_per_slot", "min_waypoints"):
        if not _integer_at_least(getattr(config, name), 1):
            errors.append(f"{name} должен быть положительным целым")
    if (
        not _integer_at_least(config.min_waypoints, 1)
        or not _integer_at_least(config.max_waypoints, 1)
        or config.max_waypoints < config.min_waypoints
    ):
        errors.append("max_waypoints должен быть не меньше min_waypoints")

    for name in ("crossover_probability", "mutation_probability"):
        value = getattr(config, name)
        if not _finite_number(value) or not 0 <= value <= 1:
            errors.append(f"{name} должен быть конечным числом от 0 до 1")
    for name in ("initial_length_factor", "segment_length_m"):
        if not _positive_number(getattr(config, name)):
            errors.append(f"{name} должен быть положительным конечным числом")
    if config.mutation_method not in ("A", "B"):
        errors.append("mutation_method должен быть A или B")
    if config.seed is not None and type(config.seed) is not int:
        errors.append("seed должен быть целым числом или пустым")

    if config.random_weight_strength is None:
        unresolved.append("Q06-C: сила случайных поисковых весов не согласована")
    elif not _positive_number(config.random_weight_strength):
        errors.append("random_weight_strength должен быть положительным конечным")
    if config.stagnation_window is None:
        unresolved.append("Q06-C: окно стагнации не согласовано")
    elif not _integer_at_least(config.stagnation_window, 1):
        errors.append("stagnation_window должен быть положительным целым")


def _check_fitness(config: FitnessConfig, errors: list[str], unresolved: list[str]) -> None:
    """Отделить ошибки введённых значений от отсутствующих решений."""

    if config.preset is None:
        unresolved.append("Пресет не выбран")
    elif config.preset not in ("fast", "short", "easy", "custom"):
        errors.append("preset неизвестен")

    if config.weights is None:
        unresolved.append("Q04-C: веса Fitness не заданы")
    elif len(config.weights) != 3 or any(
        not _finite_number(value) or value < 0 for value in config.weights
    ) or not isclose(sum(config.weights), 1.0, abs_tol=1e-9):
        errors.append("weights: нужны три неотрицательных конечных веса с суммой 1")

    if config.penalties is None:
        unresolved.append("Q04-C: штрафы P01–P15 не согласованы")
    elif len(config.penalties) != 15 or any(
        not _finite_number(value) or value < 0 for value in config.penalties
    ):
        errors.append("penalties: нужны 15 неотрицательных конечных значений")

    if config.normalization_mode is None:
        unresolved.append("Режим нормализации не выбран")
    elif config.normalization_mode not in ("auto", "manual"):
        errors.append("normalization_mode должен быть auto или manual")

    references = ("time_reference_s", "length_reference_m", "complexity_reference")
    for name in references:
        value = getattr(config, name)
        if value is not None and not _positive_number(value):
            errors.append(f"{name} должен быть положительным конечным числом")
        if config.normalization_mode == "manual" and value is None:
            unresolved.append(f"{name}: ручной эталон не задан")

    for name in ("easy_length_delta", "easy_time_delta"):
        value = getattr(config, name)
        if value is not None and (not _finite_number(value) or value < 0):
            errors.append(f"{name} должен быть неотрицательным конечным числом")
        if config.preset == "easy" and value is None:
            unresolved.append(f"Q05-C: {name} не согласован")


def validate_config(config: AppConfig) -> ValidationReport:
    """Проверить настройку перед запуском, не подставляя значения за пользователя."""

    errors: list[str] = []
    unresolved: list[str] = []
    _check_ga(config.ga, errors, unresolved)
    _check_fitness(config.fitness, errors, unresolved)
    return ValidationReport(tuple(errors), tuple(unresolved))
