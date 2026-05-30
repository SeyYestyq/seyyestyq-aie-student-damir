# Sample Outputs — примеры ответов API

Реальные примеры ответов API Drop Forecaster, сгенерированные программно.

## Файлы

| Файл | Endpoint | Описание |
|------|----------|----------|
| `predict_single_response.json` | `POST /predict` | Единичное предсказание (Москва, 500K, аукцион) |
| `predict_batch_response.json` | `POST /predict-batch` | Пакетное предсказание (3 тендера) |
| `health_response.json` | `GET /health` | Health-check ответ |
| `info_response.json` | `GET /info` | Информация о модели |

## Как воспроизвести

```bash
# Запустить API
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Единичный запрос
curl -s http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @artifacts/sample_outputs/predict_single_response.json | python3 -m json.tool

# Health
curl -s http://localhost:8000/health | python3 -m json.tool
```
