# Описание данных — Drop Forecaster

## Файлы данных

| Файл | Описание | Строк | Колонок |
|------|----------|-------|---------|
| `history_drop_dataset.csv` | Основной датасет для обучения | 5 000 | 10 |

## Структура `history_drop_dataset.csv`

| Колонка | Тип | Описание |
|---------|-----|----------|
| `customer_inn` | str | ИНН заказчика |
| `region` | str | Регион заказчика |
| `fz_type` | str | Тип закупки (44-ФЗ / 223-ФЗ) |
| `procedure_type` | str | Тип процедуры закупки |
| `nmck_rub` | float | НМЦК в рублях |
| `nmck_log1p` | float | log(1 + НМЦК) |
| `participants_count_est` | int | Число участников |
| `customer_avg_drop` | float | Среднее историческое снижение заказчика |
| `customer_contracts_count` | int | Число контрактов заказчика |
| `drop_fraction` | float | **Целевая:** доля снижения цены [0, 1] |

## Источники

| Источник | Описание |
|----------|----------|
| ЕИС (zakupki.gov.ru) | Реестр контрактов (оригинальные данные) |
| Демо-генератор (`src/data_loader.py`) | Синтетические данные для демонстрации |

## Регенерация

```bash
python -m src.data_loader --generate-demo --n-samples 5000 --output data/history_drop_dataset.csv
```
