# Ноутбуки проекта Drop Forecaster

## Состав

| Файл | Описание |
|------|----------|
| `01_eda_and_data_audit.ipynb` | Разведочный анализ данных, визуализации, описательная статистика |
| `02_model_comparison.ipynb` | Сравнение 6 моделей (CatBoost, XGBoost, LightGBM, RandomForest, MLP, Ridge), кросс-валидация, feature importance |
| `03_system_evaluation_and_demo.ipynb` | Оценка системы, демонстрация API, анализ ошибок |

## Данные

Все ноутбуки читают данные из локального файла `../data/history_drop_dataset.csv` (5000 записей).

## Графики

Сгенерированные графики сохраняются в `../artifacts/`:

- `eda_target_distribution.png` — распределение целевой переменной
- `eda_correlation_matrix.png` — матрица корреляций
- `eda_drop_by_procedure.png` — снижение по типу процедуры
- `model_comparison_metrics.png` — сравнение 6 моделей
- `feature_importance.png` — важность признаков CatBoost
- `error_analysis.png` — анализ ошибок
- `mlp_learning_curve.png` — кривая обучения MLP
