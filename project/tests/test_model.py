"""
Тесты для Drop Forecaster.

Покрывает:
- Препроцессинг признаков
- Инференс модели (fallback-режим)
- Пакетный инференс
- API endpoints (health, predict)
- Pydantic-схемы валидации
"""

import math
import pytest
from fastapi.testclient import TestClient

from src.model import DropForecasterModel, ALL_FEATURES, CAT_FEATURES, NUM_FEATURES
from src.data_loader import generate_demo_dataset
from src.api import app


# Fixtures

@pytest.fixture
def model():
    """Модель без загруженного .cbm файла (fallback-режим)."""
    m = DropForecasterModel(model_path="nonexistent_model.cbm")
    m.load()
    return m


@pytest.fixture
def sample_tender():
    """Пример входных данных тендера."""
    return {
        "customer_inn": "7707083893",
        "region": "г. Москва",
        "fz_type": "44-ФЗ",
        "procedure_type": "Электронный аукцион",
        "nmck_rub": 500000,
        "participants_count_est": 5,
        "customer_avg_drop": 0.12,
        "customer_contracts_count": 10,
    }


@pytest.fixture
def client():
    """TestClient для FastAPI."""
    return TestClient(app)


# Тесты препроцессинга

class TestPreprocessing:
    """Тесты модуля препроцессинга."""

    def test_preprocess_basic(self, model, sample_tender):
        """Проверка базового препроцессинга."""
        features = model.preprocess(sample_tender)

        assert isinstance(features, dict)
        assert features["customer_inn"] == "7707083893"
        assert features["region"] == "г. Москва"
        assert features["fz_type"] == "44-ФЗ"
        assert features["nmck_log1p"] == pytest.approx(
            math.log1p(500000), rel=1e-4
        )

    def test_preprocess_defaults(self, model):
        """Проверка значений по умолчанию при неполных данных."""
        features = model.preprocess({})

        assert features["customer_inn"] == "unknown"
        assert features["region"] == "unknown"
        assert features["fz_type"] == "44-ФЗ"
        assert features["nmck_log1p"] == pytest.approx(math.log1p(0))

    def test_preprocess_all_features_present(self, model, sample_tender):
        """Все признаки должны присутствовать в результате."""
        features = model.preprocess(sample_tender)
        for feat in ALL_FEATURES:
            assert feat in features, f"Признак {feat} отсутствует"

    def test_preprocess_cat_features_are_strings(self, model, sample_tender):
        """Категориальные признаки должны быть строками."""
        features = model.preprocess(sample_tender)
        for feat in CAT_FEATURES:
            assert isinstance(features[feat], str), f"{feat} не строка"

    def test_preprocess_num_features_are_float(self, model, sample_tender):
        """Числовые признаки должны быть числами."""
        features = model.preprocess(sample_tender)
        for feat in NUM_FEATURES:
            assert isinstance(features[feat], (int, float)), f"{feat} не число"


# Тесты модели

class TestModel:
    """Тесты инференса модели."""

    def test_predict_single_returns_dict(self, model, sample_tender):
        """predict_single возвращает словарь с нужными ключами."""
        result = model.predict_single(sample_tender)

        assert isinstance(result, dict)
        assert "predicted_drop_fraction" in result
        assert "predicted_drop_percent" in result
        assert "confidence_low" in result
        assert "confidence_high" in result
        assert "method" in result

    def test_predict_single_range(self, model, sample_tender):
        """Предсказание должно быть в диапазоне [0, 1]."""
        result = model.predict_single(sample_tender)

        assert 0.0 <= result["predicted_drop_fraction"] <= 1.0
        assert 0.0 <= result["predicted_drop_percent"] <= 100.0
        assert result["confidence_low"] <= result["predicted_drop_fraction"]
        assert result["confidence_high"] >= result["predicted_drop_fraction"]

    def test_predict_single_fallback_method(self, model, sample_tender):
        """Без загруженной модели должен использоваться fallback."""
        result = model.predict_single(sample_tender)
        assert result["method"] == "fallback_mean"

    def test_predict_batch_empty(self, model):
        """Пустой пакет должен возвращать пустой список."""
        results = model.predict_batch([])
        assert results == []

    def test_predict_batch_multiple(self, model, sample_tender):
        """Пакетный инференс для нескольких элементов."""
        items = [sample_tender, sample_tender, sample_tender]
        results = model.predict_batch(items)

        assert len(results) == 3
        for r in results:
            assert 0.0 <= r["predicted_drop_fraction"] <= 1.0

    def test_model_info(self, model):
        """Информация о модели содержит нужные поля."""
        info = model.get_info()

        assert info["model_type"] == "CatBoostRegressor"
        assert isinstance(info["features"], list)
        assert len(info["features"]) > 0

    def test_feature_importance_fallback(self, model):
        """Важность признаков в fallback-режиме — нули."""
        importance = model.get_feature_importance()

        assert isinstance(importance, dict)
        assert all(v == 0.0 for v in importance.values())


# Тесты данных

class TestDataLoader:
    """Тесты генерации данных."""

    def test_generate_demo_shape(self):
        """Размер сгенерированного датасета."""
        df = generate_demo_dataset(n_samples=100, seed=42)
        assert len(df) == 100

    def test_generate_demo_columns(self):
        """Все необходимые колонки присутствуют."""
        df = generate_demo_dataset(n_samples=50)
        required = [
            "customer_inn", "region", "fz_type", "procedure_type",
            "nmck_rub", "participants_count_est", "drop_fraction",
        ]
        for col in required:
            assert col in df.columns, f"Колонка {col} отсутствует"

    def test_generate_demo_drop_range(self):
        """drop_fraction в допустимом диапазоне [0, 0.85]."""
        df = generate_demo_dataset(n_samples=1000)
        assert df["drop_fraction"].min() >= 0.0
        assert df["drop_fraction"].max() <= 0.85

    def test_generate_demo_reproducibility(self):
        """Одинаковый seed → одинаковые данные."""
        df1 = generate_demo_dataset(n_samples=100, seed=123)
        df2 = generate_demo_dataset(n_samples=100, seed=123)
        assert df1.equals(df2)


# Тесты API

class TestAPI:
    """Тесты FastAPI endpoints."""

    def test_health(self, client):
        """Health-check возвращает 200."""
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert "uptime_seconds" in data

    def test_info(self, client):
        """Endpoint /info возвращает информацию о модели."""
        resp = client.get("/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "model_type" in data
        assert "features" in data

    def test_predict_valid(self, client):
        """Валидный запрос к /predict."""
        payload = {
            "customer_inn": "7707083893",
            "region": "г. Москва",
            "fz_type": "44-ФЗ",
            "procedure_type": "Электронный аукцион",
            "nmck_rub": 500000,
            "participants_count_est": 5,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "predicted_drop_fraction" in data
        assert 0.0 <= data["predicted_drop_fraction"] <= 1.0

    def test_predict_missing_required(self, client):
        """Запрос без обязательных полей → 422."""
        payload = {"fz_type": "44-ФЗ"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_invalid_nmck(self, client):
        """Отрицательная НМЦК → 422."""
        payload = {
            "customer_inn": "7707083893",
            "region": "г. Москва",
            "nmck_rub": -100,
        }
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_predict_batch_valid(self, client):
        """Пакетный запрос к /predict-batch."""
        item = {
            "customer_inn": "7707083893",
            "region": "г. Москва",
            "nmck_rub": 500000,
        }
        payload = {"items": [item, item]}
        resp = client.post("/predict-batch", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["predictions"]) == 2
