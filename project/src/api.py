"""
FastAPI-сервис для прогнозирования снижения цены на тендерах.

Endpoints:
    GET  /health         — проверка работоспособности
    GET  /info           — информация о модели
    POST /predict        — единичное предсказание
    POST /predict-batch  — пакетное предсказание
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .model import get_model, DropForecasterModel
from .config import settings

# Логирование

logging.basicConfig(
    level=getattr(logging, settings.api_log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("drop_forecaster.api")


# Pydantic-схемы

class TenderInput(BaseModel):
    """Входные данные для предсказания по одному тендеру."""

    customer_inn: str = Field(
        ...,
        description="ИНН заказчика",
        examples=["7707083893"],
    )
    region: str = Field(
        ...,
        description="Регион заказчика",
        examples=["г. Москва"],
    )
    fz_type: str = Field(
        default="44-ФЗ",
        description="Тип закупки: 44-ФЗ или 223-ФЗ",
        examples=["44-ФЗ"],
    )
    procedure_type: str = Field(
        default="Электронный аукцион",
        description="Тип процедуры закупки",
        examples=["Электронный аукцион"],
    )
    nmck_rub: float = Field(
        ...,
        gt=0,
        description="Начальная максимальная цена контракта (руб.)",
        examples=[500000.0],
    )
    participants_count_est: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Ожидаемое число участников",
        examples=[5],
    )
    customer_avg_drop: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Историческое среднее снижение заказчика",
    )
    customer_contracts_count: Optional[int] = Field(
        default=None,
        ge=0,
        description="Число прошлых контрактов заказчика",
    )


class PredictionOutput(BaseModel):
    """Результат предсказания."""

    predicted_drop_fraction: float = Field(
        description="Прогноз снижения цены (доля от НМЦК)",
    )
    predicted_drop_percent: float = Field(
        description="Прогноз снижения цены в процентах",
    )
    confidence_low: float = Field(
        description="Нижняя граница 95%-доверительного интервала",
    )
    confidence_high: float = Field(
        description="Верхняя граница 95%-доверительного интервала",
    )
    method: str = Field(
        description="Метод прогнозирования (catboost / fallback_mean)",
    )


class BatchInput(BaseModel):
    """Входные данные для пакетного предсказания."""

    items: list[TenderInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Список тендеров для предсказания",
    )


class BatchOutput(BaseModel):
    """Результат пакетного предсказания."""

    predictions: list[PredictionOutput]
    count: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    """Ответ health-check."""

    status: str
    model_loaded: bool
    model_version: str
    uptime_seconds: float


class InfoResponse(BaseModel):
    """Информация о модели."""

    model_type: str
    version: str
    is_loaded: bool
    features: list[str]
    cat_features: list[str]
    num_features: list[str]
    model_path: str


# FastAPI — приложение

_start_time: float = 0.0
_model: Optional[DropForecasterModel] = None


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Жизненный цикл приложения: загрузка модели при старте."""
    global _start_time, _model
    _start_time = time.time()

    logger.info("Загрузка модели Drop Forecaster...")
    _model = get_model()
    logger.info(
        "Модель загружена: loaded=%s, version=%s",
        _model.is_loaded,
        _model.version,
    )
    yield
    logger.info("Завершение работы API")


app = FastAPI(
    title="Drop Forecaster API",
    description=(
        "Сервис прогнозирования снижения цены на государственных тендерах. "
        "Использует CatBoost-модель, обученную на исторических данных ЕИС."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Endpoints

@app.get("/", include_in_schema=False)
async def root():
    """Редирект на документацию API."""
    from starlette.responses import RedirectResponse
    return RedirectResponse(url="/docs")

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Проверка работоспособности сервиса."""
    return HealthResponse(
        status="ok",
        model_loaded=_model.is_loaded if _model else False,
        model_version=_model.version if _model else "unknown",
        uptime_seconds=round(time.time() - _start_time, 2),
    )


@app.get("/info", response_model=InfoResponse, tags=["system"])
async def model_info():
    """Информация о загруженной модели."""
    model = _model or get_model()
    info = model.get_info()
    return InfoResponse(**info)


@app.post("/predict", response_model=PredictionOutput, tags=["prediction"])
async def predict(tender: TenderInput):
    """
    Предсказание снижения цены для одного тендера.

    Принимает характеристики тендера и возвращает ожидаемое
    снижение цены (drop_fraction) с доверительным интервалом.
    """
    model = _model or get_model()

    logger.info(
        "Запрос /predict: inn=%s region=%s nmck=%.0f",
        tender.customer_inn,
        tender.region,
        tender.nmck_rub,
    )

    try:
        raw = tender.model_dump()
        result = model.predict_single(raw)
        logger.info(
            "Результат: drop=%.4f method=%s",
            result["predicted_drop_fraction"],
            result["method"],
        )
        return PredictionOutput(**result)
    except Exception as exc:
        logger.error("Ошибка предсказания: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/predict-batch", response_model=BatchOutput, tags=["prediction"])
async def predict_batch(batch: BatchInput):
    """
    Пакетное предсказание для нескольких тендеров (до 100).

    Возвращает список предсказаний и статистику выполнения.
    """
    model = _model or get_model()

    if not settings.enable_batch_predict:
        raise HTTPException(
            status_code=403,
            detail="Пакетное предсказание отключено",
        )

    logger.info("Запрос /predict-batch: %d элементов", len(batch.items))

    t0 = time.time()
    raw_items = [item.model_dump() for item in batch.items]
    results = model.predict_batch(raw_items)
    elapsed = (time.time() - t0) * 1000

    predictions = [PredictionOutput(**r) for r in results]

    logger.info(
        "Batch готов: %d предсказаний за %.1f мс",
        len(predictions),
        elapsed,
    )

    return BatchOutput(
        predictions=predictions,
        count=len(predictions),
        elapsed_ms=round(elapsed, 2),
    )


# Запуск (для отладки)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_debug,
    )
