# S03 – eda_cli: мини-EDA для CSV

Небольшое CLI-приложение для базового анализа CSV-файлов.
Используется в рамках Семинара 03 курса «Инженерия ИИ».

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (S03):

```bash
uv sync
```

Эта команда:

- создаст виртуальное окружение `.venv`;
- установит зависимости из `pyproject.toml`;
- установит сам проект `eda-cli` в окружение.

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli overview data/example.csv
```

Параметры:

- `--sep` – разделитель (по умолчанию `,`);
- `--encoding` – кодировка (по умолчанию `utf-8`).

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```
Параметры:
- `--limit-unique` - Порог уникальных значений для категориальных признаков (по умолчанию `55`)
- `--top-k` - количество топ категорий (по умолчанию `5`)  



В результате в каталоге `reports/` появятся:

- `report.md` – основной отчёт в Markdown;
- `summary.csv` – таблица по колонкам;
- `missing.csv` – пропуски по колонкам;
- `correlation.csv` – корреляционная матрица (если есть числовые признаки);
- `top_categories/*.csv` – top-k категорий по строковым признакам;
- `hist_*.png` – гистограммы числовых колонок;
- `missing_matrix.png` – визуализация пропусков;
- `correlation_heatmap.png` – тепловая карта корреляций.

## Тесты

```bash
uv run pytest -q
```
Небольшое CLI-приложение для базового анализа CSV-файлов с HTTP-сервисом на FastAPI.
## HTTP-сервис

Запуск:
uv run uvicorn eda_cli.api:app --reload --port 8000
Документация: `http://localhost:8000/docs`

**Эндпоинты:**
- `GET /health` — проверка работоспособности
- `POST /quality` — оценка качества по агрегированным признакам
- `POST /quality-from-csv` — оценка качества по CSV-файлу
- `POST /quality-flags-from-csv` — возвращает все булевы флаги качества из CSV (включая `has_suspicious_id_dublictes`, `has_high_cardinality_categoricals` из HW03)