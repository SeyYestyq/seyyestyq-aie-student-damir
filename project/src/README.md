# Исходный код — Drop Forecaster

## Модули

| Файл | Описание |
|------|----------|
| `__init__.py` | Инициализация пакета (версия, автор) |
| `config.py` | Конфигурация через Pydantic Settings + .env |
| `model.py` | ML-модель CatBoost: загрузка, препроцессинг, инференс |
| `data_loader.py` | Генератор синтетических данных + загрузка из CSV |
| `api.py` | FastAPI: `/predict`, `/predict-batch`, `/health`, `/info` |
