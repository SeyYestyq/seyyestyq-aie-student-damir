"""
Модуль ML-модели Drop Forecaster.

Содержит:
- Загрузку обученной модели CatBoost
- Препроцессинг входных данных
- Инференс (единичный и пакетный)
- Калибровку доверительного интервала
"""

import math
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import settings

logger = logging.getLogger(__name__)


# Схемы данных

# Категориальные признаки для CatBoost
CAT_FEATURES = ["customer_inn", "region", "fz_type", "procedure_type"]

# Числовые признаки
NUM_FEATURES = [
    "nmck_log1p",
    "participants_count_est",
    "customer_avg_drop",
    "customer_contracts_count",
]

# Все признаки для модели (порядок важен)
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES


# Модель

class DropForecasterModel:
    """
    Обёртка над CatBoost-моделью для прогнозирования drop_fraction.

    Attributes:
        model: Обученная CatBoostRegressor модель
        version: Версия модели
        is_loaded: Флаг загрузки модели
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Инициализация модели.

        Args:
            model_path: Путь к файлу .cbm модели CatBoost.
                        Если None — используется путь из конфигурации.
        """
        self.model: Optional[CatBoostRegressor] = None
        self.version: str = settings.model_version
        self.is_loaded: bool = False
        self._model_path = model_path or settings.model_path

    def load(self) -> None:
        """Загрузка модели из файла."""
        path = Path(self._model_path)
        if not path.exists():
            logger.warning(
                "Файл модели не найден: %s. "
                "Используется fallback-модель (среднее значение).",
                path,
            )
            self.model = None
            self.is_loaded = False
            return

        try:
            self.model = CatBoostRegressor()
            self.model.load_model(str(path))
            self.is_loaded = True
            logger.info("Модель загружена: %s", path)
        except Exception as exc:
            logger.error("Ошибка загрузки модели: %s", exc)
            self.model = None
            self.is_loaded = False

    # Препроцессинг

    @staticmethod
    def preprocess(raw: dict) -> dict:
        """
        Подготовка признаков из «сырого» запроса.

        Args:
            raw: Словарь с данными тендера от пользователя.

        Returns:
            Словарь с вычисленными признаками для модели.
        """
        nmck = raw.get("nmck_rub", 0)
        features = {
            # Категориальные
            "customer_inn": str(raw.get("customer_inn", "unknown")),
            "region": str(raw.get("region", "unknown")),
            "fz_type": str(raw.get("fz_type", "44-ФЗ")),
            "procedure_type": str(raw.get("procedure_type", "Электронный аукцион")),
            # Числовые
            "nmck_log1p": math.log1p(float(nmck)),
            "participants_count_est": float(
                raw.get("participants_count_est", 3)
            ),
            "customer_avg_drop": float(raw.get("customer_avg_drop") or 0.12),
            "customer_contracts_count": float(
                raw.get("customer_contracts_count") or 10
            ),
        }
        return features

    # Инференс

    def predict_single(self, raw: dict) -> dict:
        """
        Предсказание для одного тендера.

        Args:
            raw: Словарь с данными тендера.

        Returns:
            Словарь с предсказанием:
            - predicted_drop_fraction: точечный прогноз
            - predicted_drop_percent: прогноз в процентах
            - confidence_low: нижняя граница CI
            - confidence_high: верхняя граница CI
            - method: метод предсказания
        """
        features = self.preprocess(raw)
        df = pd.DataFrame([features])

        if self.model is not None and self.is_loaded:
            pred = float(self.model.predict(df[ALL_FEATURES])[0])
            method = "catboost"
        else:
            # Fallback: среднее значение + поправка на участников
            base = features["customer_avg_drop"]
            participants_adj = min(
                features["participants_count_est"] * 0.01, 0.1
            )
            pred = base + participants_adj
            method = "fallback_mean"

        # Клиппинг в допустимый диапазон
        pred = float(np.clip(pred, 0.0, 1.0))

        # Простой доверительный интервал (~1 sigma)
        sigma = 0.04 if method == "catboost" else 0.08
        ci_low = float(np.clip(pred - 2 * sigma, 0.0, 1.0))
        ci_high = float(np.clip(pred + 2 * sigma, 0.0, 1.0))

        return {
            "predicted_drop_fraction": round(pred, 4),
            "predicted_drop_percent": round(pred * 100, 2),
            "confidence_low": round(ci_low, 4),
            "confidence_high": round(ci_high, 4),
            "method": method,
        }

    def predict_batch(self, items: list[dict]) -> list[dict]:
        """
        Пакетное предсказание для нескольких тендеров.

        Args:
            items: Список словарей с данными тендеров.

        Returns:
            Список словарей с предсказаниями.
        """
        if not items:
            return []

        results = []
        for item in items:
            result = self.predict_single(item)
            results.append(result)

        return results

    def get_feature_importance(self) -> dict:
        """
        Получение важности признаков.

        Returns:
            Словарь {имя_признака: важность}.
        """
        if self.model is None or not self.is_loaded:
            return {f: 0.0 for f in ALL_FEATURES}

        importance = self.model.get_feature_importance()
        return dict(zip(ALL_FEATURES, [round(v, 4) for v in importance]))

    def get_info(self) -> dict:
        """Информация о модели."""
        return {
            "model_type": "CatBoostRegressor",
            "version": self.version,
            "is_loaded": self.is_loaded,
            "features": ALL_FEATURES,
            "cat_features": CAT_FEATURES,
            "num_features": NUM_FEATURES,
            "model_path": self._model_path,
        }


# Глобальный экземпляр модели (singleton)

_model_instance: Optional[DropForecasterModel] = None


def get_model() -> DropForecasterModel:
    """
    Получить глобальный экземпляр модели (lazy-init singleton).

    Returns:
        Загруженная модель DropForecasterModel.
    """
    global _model_instance
    if _model_instance is None:
        _model_instance = DropForecasterModel()
        _model_instance.load()
    return _model_instance
