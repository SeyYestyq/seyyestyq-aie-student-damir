"""
Конфигурация приложения Drop Forecaster.

Управление конфигурацией через переменные окружения и Pydantic Settings.
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Настройки приложения, загружаемые из переменных окружения."""

    # Database 
    database_url: str = Field(
        default="sqlite:///./data/demo.db",
        description="URL базы данных для загрузки исторических данных",
    )

    #  Model 
    model_path: str = Field(
        default="artifacts/drop_forecaster.cbm",
        description="Путь к файлу модели CatBoost",
    )
    model_version: str = Field(
        default="1.0.0",
        description="Версия модели",
    )

    # --- API ---
    api_host: str = Field(default="0.0.0.0", description="Хост API")
    api_port: int = Field(default=8000, description="Порт API")
    api_debug: bool = Field(default=False, description="Режим отладки")
    api_log_level: str = Field(default="INFO", description="Уровень логирования")

    #  Features 
    enable_batch_predict: bool = Field(
        default=True,
        description="Включить пакетное предсказание",
    )
    max_batch_size: int = Field(
        default=100,
        description="Максимальный размер пакета для /predict-batch",
    )

    # Logging 
    log_format: str = Field(default="json", description="Формат логов: json | text")
    log_file: str = Field(default="logs/api.log", description="Файл логов")

    # Paths 
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Корень проекта",
    )

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Глобальный экземпляр настроек
settings = Settings()
