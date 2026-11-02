"""Состояние интерфейса, которое позже свяжет виджеты с расчётом."""

from dataclasses import dataclass, field

from algorithm.models import AppConfig
from config import default_config


@dataclass(slots=True)
class UIState:
    """Хранить настройки и готовность графа отдельно от элементов NiceGUI."""

    config: AppConfig = field(default_factory=default_config)
    graph_loaded: bool = False
