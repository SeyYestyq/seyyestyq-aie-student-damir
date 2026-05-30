# Артефакты проекта Drop Forecaster

## Содержимое

| Файл/Папка | Описание |
|------------|----------|
| `model_metrics.md` | Таблица сравнения моделей и feature importance |
| `sample_outputs/` | Примеры JSON-ответов API |
| `*.png` | Графики из EDA и экспериментов |

## Генерация графиков

Графики создаются при запуске ноутбуков:

```bash
cd notebooks
jupyter nbconvert --execute 01_eda_and_data_audit.ipynb
jupyter nbconvert --execute 02_model_comparison.ipynb
```

Результат сохраняется в `artifacts/`:
- `eda_target_distribution.png`
- `eda_correlation_matrix.png`
- `eda_drop_vs_participants.png`
- `eda_drop_by_procedure.png`
- `eda_drop_by_region.png`
- `eda_drop_by_fz_type.png`
- `eda_drop_vs_nmck.png`
- `model_comparison_metrics.png`
- `feature_importance.png`
- `error_analysis.png`
