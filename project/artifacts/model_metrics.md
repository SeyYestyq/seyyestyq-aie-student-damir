# Метрики моделей — Drop Forecaster

## Сравнение 6 моделей (holdout test, 20%)

| Модель | Тип | MAE | RMSE | R² | Время обуч. (с) |
|--------|-----|-----|------|-----|------------------|
| **CatBoost** | gradient_boosting | **0.0233** | **0.0294** | **0.665** | 1.5 |
| RandomForest | ensemble | 0.0240 | 0.0301 | 0.648 | 0.5 |
| LightGBM | gradient_boosting | 0.0256 | 0.0323 | 0.596 | 1.9 |
| XGBoost | gradient_boosting | 0.0260 | 0.0327 | 0.585 | 1.3 |
| Ridge | linear | 0.0268 | 0.0336 | 0.563 | 0.0 |
| MLP | neural_network | 0.0282 | 0.0352 | 0.520 | 0.5 |

## Кросс-валидация CatBoost (5-fold)

| Fold | MAE | RMSE | R² |
|------|-----|------|-----|
| 1 | 0.0234 | 0.0294 | 0.666 |
| 2 | 0.0236 | 0.0302 | 0.640 |
| 3 | 0.0245 | 0.0310 | 0.622 |
| 4 | 0.0241 | 0.0299 | 0.649 |
| 5 | 0.0241 | 0.0299 | 0.649 |
| **Среднее** | **0.0239 ± 0.0004** | **0.0301 ± 0.0006** | **0.645 ± 0.016** |

## Feature Importance (CatBoost)

| # | Признак | Важность (%) |
|---|---------|--------------|
| 1 | `participants_count_est` | 35.3 |
| 2 | `procedure_type` | 20.7 |
| 3 | `nmck_log1p` | 17.1 |
| 4 | `customer_inn` | 8.9 |
| 5 | `customer_avg_drop` | 7.3 |
| 6 | `region` | 5.9 |
| 7 | `customer_contracts_count` | 3.5 |
| 8 | `fz_type` | 1.3 |

## Выводы

1. CatBoost — лучшая модель, стабильна на CV (MAE std = 0.0004)
2. MLP (нейросетевой базлайн) уступает всем бустингам — ожидаемо для табличных данных
3. Число участников — главный предиктор (35.3%)
4. R² = 0.665 — модель объясняет ~67% дисперсии, что хорошо для стохастической задачи

## Графики

- `model_comparison_metrics.png` — сравнение 6 моделей по MAE/RMSE/R²
- `feature_importance.png` — важность признаков CatBoost
- `error_analysis.png` — predicted vs actual, residuals
- `mlp_learning_curve.png` — кривая обучения MLP
