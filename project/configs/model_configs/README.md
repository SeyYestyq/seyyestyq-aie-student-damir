# Конфигурации модели

## Файлы

| Файл | Описание |
|------|----------|
| `catboost_params.yaml` | Гиперпараметры финальной CatBoost-модели |

## Гиперпараметры

```yaml
iterations: 1000
learning_rate: 0.05
depth: 6
l2_leaf_reg: 3.0
loss_function: RMSE
```

Подробнее в `notebooks/02_model_comparison.ipynb`
