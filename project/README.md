# Drop Forecaster — прогнозирование снижения цены на тендерах

## Паспорт проекта

| Поле | Значение |
|------|----------|
| **Название** | Drop Forecaster |
| **Автор** | Ахмеров Дамир Зинурович |
| **Группа** | ЭФБО-13-24 |
| **Контакт** | damirxxl_XXl@mail.ru / Telegram: @sey_yestyq |
| **Репозиторий** | https://github.com/SeyYestyq/seyyestyq-aie-student-damir/tree/main/project |

---

## Краткое описание

Сервис **Drop Forecaster** предсказывает ожидаемое снижение цены (`drop_fraction`)
на государственных тендерах на основе исторических данных о контрактах.

**Входные данные:** характеристики тендера (ИНН заказчика, регион, тип закупки, НМЦК и др.)

**Выходные данные:** ожидаемое снижение цены в процентах + доверительный интервал

---

## Предметная область

Государственные закупки в России (44-ФЗ / 223-ФЗ) — более 4 млн тендеров в год.
При торгах цена контракта снижается на 1–60% от НМЦК. Прогнозирование снижения
помогает участникам оценить маржинальность, а заказчикам — скорректировать бюджет.

---

## Постановка задачи

| Параметр | Значение |
|----------|----------|
| **Тип задачи** | Регрессия |
| **Целевая** | `drop_fraction ∈ [0, 1]` |
| **Метрики** | MAE, RMSE, R² |
| **Лучшая модель** | CatBoost |
| **Результат** | MAE=0.023, RMSE=0.029, R²=0.665 |

---

## Структура проекта

```
project/
├── README.md               # Паспорт проекта
├── report.md               # Полный отчёт
├── self-checklist.md       # Чеклист самопроверки (10/10)
├── requirements.txt        # Зависимости Python
├── Dockerfile              # Контейнеризация
├── docker-compose.yml      # Docker Compose
├── conftest.py             # Конфигурация pytest
│
├── notebooks/
│   ├── 01_eda_and_data_audit.ipynb    # EDA и аудит данных
│   ├── 02_model_comparison.ipynb      # Сравнение 6 моделей (вкл. MLP)
│   └── 03_system_evaluation_and_demo.ipynb  # Демо и оценка системы
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Конфигурация (Pydantic Settings)
│   ├── model.py            # ML-модель (CatBoost, инференс)
│   ├── data_loader.py      # Загрузка и генерация данных
│   ├── api.py              # FastAPI сервис
│   └── train.py            # Обучение моделей
│
├── configs/
│   ├── .env.example        # Шаблон переменных окружения
│   └── model_configs/      # Гиперпараметры моделей
│
├── data/
│   ├── history_drop_dataset.csv  # Основной датасет (5000 строк)
│   └── sample/             # Примеры данных
│
├── tests/
│   └── test_model.py       # 22 unit-теста
│
├── artifacts/
│   ├── drop_forecaster.cbm # Обученная модель CatBoost
│   └── sample_outputs/     # Примеры JSON-ответов API
```

---

## Быстрый старт

### 1. Установка зависимостей

```bash
cd project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Обучение моделей

```bash
python -m src.train
```

### 3. Запуск API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Тестирование

```bash
pytest tests/ -v
```

### 5. Docker

```bash
docker compose up -d
```

---

## API Endpoints

| Метод | Endpoint | Описание |
|-------|----------|----------|
| GET | `/health` | Health-check (модель, uptime) |
| GET | `/info` | Информация о модели |
| POST | `/predict` | Единичное предсказание |
| POST | `/predict-batch` | Пакетное предсказание (до 100) |
| GET | `/docs` | Swagger UI (автодокументация) |

### Пример запроса

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customer_inn": "7707083893",
    "region": "г. Москва",
    "fz_type": "44-ФЗ",
    "procedure_type": "Электронный аукцион",
    "nmck_rub": 500000,
    "participants_count_est": 5
  }'
```

### Пример ответа

```json
{
  "predicted_drop_fraction": 0.15,
  "predicted_drop_percent": 15.0,
  "confidence_low": 0.08,
  "confidence_high": 0.22,
  "method": "catboost"
}
```

---

## Результаты сравнения моделей

| Модель | Тип | MAE | RMSE | R² |
|--------|-----|-----|------|-----|
| **CatBoost** | gradient_boosting | **0.0233** | **0.0294** | **0.665** |
| RandomForest | ensemble | 0.0240 | 0.0301 | 0.648 |
| LightGBM | gradient_boosting | 0.0256 | 0.0323 | 0.596 |
| XGBoost | gradient_boosting | 0.0260 | 0.0327 | 0.585 |
| Ridge | linear | 0.0268 | 0.0336 | 0.563 |
| MLP | neural_network | 0.0282 | 0.0352 | 0.520 |

---

## Сценарий демонстрации на защите

1. Показать структуру проекта и README
2. Открыть EDA-ноутбук — визуализации и выводы
3. Открыть ноутбук сравнения моделей — 6 моделей
4. Запустить API: `uvicorn src.api:app --reload`
5. Отправить тестовый запрос через Swagger UI (`/docs`)
6. Показать health-check и логи
7. Запустить тесты: `pytest tests/ -v`
8. Обсудить выбор модели и дальнейшие шаги
