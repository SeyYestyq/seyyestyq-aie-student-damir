"""

Запуск:
    python src/train.py

"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBRegressor


# Конфигурация

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "history_drop_dataset.csv"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
EXPERIMENT_NAME = "drop_forecaster"
SEED = 42

CAT_FEATURES = ["customer_inn", "region", "fz_type", "procedure_type"]
NUM_FEATURES = [
    "nmck_log1p",
    "participants_count_est",
    "customer_avg_drop",
    "customer_contracts_count",
]
ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "drop_fraction"


def load_data():
    """Загрузка и разбиение данных."""
    df = pd.read_csv(DATA_PATH)
    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    return X_train, X_test, y_train, y_test


def encode_categorical(X_train, X_test):
    """LabelEncoding для моделей без нативной поддержки категорий."""
    encoders = {}
    X_train_enc = X_train.copy()
    X_test_enc = X_test.copy()

    for col in CAT_FEATURES:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train[col].astype(str))
        X_test_enc[col] = X_test[col].astype(str).map(
            lambda x, _le=le: x if x in _le.classes_ else _le.classes_[0]
        )
        X_test_enc[col] = le.transform(X_test_enc[col])
        encoders[col] = le

    return X_train_enc, X_test_enc, encoders


def compute_metrics(y_true, y_pred):
    """Вычисление метрик."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def get_models():
    """Определение моделей для сравнения."""
    return {
        "CatBoost": {
            "model": CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                l2_leaf_reg=3.0,
                random_seed=SEED,
                verbose=0,
                cat_features=[0, 1, 2, 3],
            ),
            "use_cat_native": True,
            "params": {
                "iterations": 1000,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 3.0,
            },
        },
        "XGBoost": {
            "model": XGBRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                reg_lambda=3.0,
                random_state=SEED,
                verbosity=0,
            ),
            "use_cat_native": False,
            "params": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "reg_lambda": 3.0,
            },
        },
        "LightGBM": {
            "model": LGBMRegressor(
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=6,
                reg_lambda=3.0,
                random_state=SEED,
                verbose=-1,
            ),
            "use_cat_native": False,
            "params": {
                "n_estimators": 1000,
                "learning_rate": 0.05,
                "max_depth": 6,
                "reg_lambda": 3.0,
            },
        },
        "RandomForest": {
            "model": RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_leaf=5,
                random_state=SEED,
                n_jobs=-1,
            ),
            "use_cat_native": False,
            "params": {
                "n_estimators": 500,
                "max_depth": 12,
                "min_samples_leaf": 5,
            },
        },
        "MLP": {
            "model": MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation="relu",
                solver="adam",
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=SEED,
                verbose=False,
            ),
            "use_cat_native": False,
            "needs_scaling": True,
            "params": {
                "hidden_layers": "128-64-32",
                "activation": "relu",
                "solver": "adam",
                "learning_rate_init": 0.001,
                "max_iter": 500,
                "early_stopping": True,
            },
        },
        "Ridge": {
            "model": Ridge(alpha=1.0),
            "use_cat_native": False,
            "params": {"alpha": 1.0},
        },
    }


def train_and_log(
    name, cfg, X_train, X_test, y_train, y_test, X_train_enc, X_test_enc
):
    """Обучение модели и вывод метрик."""
    model = cfg["model"]
    params = cfg["params"]

    # Выбор данных
    if cfg.get("use_cat_native"):
        Xtr, Xte = X_train, X_test
    else:
        Xtr, Xte = X_train_enc, X_test_enc

    # Масштабирование для MLP
    scaler = None
    if cfg.get("needs_scaling"):
        scaler = StandardScaler()
        Xtr = pd.DataFrame(
            scaler.fit_transform(Xtr), columns=Xtr.columns, index=Xtr.index
        )
        Xte = pd.DataFrame(
            scaler.transform(Xte), columns=Xte.columns, index=Xte.index
        )

    # Обучение
    t0 = time.time()
    model.fit(Xtr, y_train)
    train_time = time.time() - t0

    # Предсказание
    t0 = time.time()
    y_pred = model.predict(Xte)
    infer_time = (time.time() - t0) / len(Xte) * 1000

    # Метрики
    metrics = compute_metrics(y_test, y_pred)
    metrics["train_time_sec"] = round(train_time, 3)
    metrics["infer_time_ms"] = round(infer_time, 4)

    print(
        f"  {name:15s} | MAE={metrics['mae']:.4f} | "
        f"RMSE={metrics['rmse']:.4f} | R²={metrics['r2']:.4f} | "
        f"Train={train_time:.1f}s"
    )

    return {
        "name": name,
        **metrics,
        "model": model,
        "scaler": scaler,
    }


def main():
    """Основная функция обучения."""
    # Загрузка данных
    X_train, X_test, y_train, y_test = load_data()
    print(f"Данные: Train={X_train.shape}, Test={X_test.shape}\n")

    # Кодирование категорий
    X_train_enc, X_test_enc, _ = encode_categorical(X_train, X_test)
    print(f"Эксперимент: {EXPERIMENT_NAME}\n")

    # Обучение всех моделей
    print(f"{'-' * 50}")
    print(f"{'Модель':<15} | {'MAE':>8} | {'RMSE':>8} | {'R²':>6} | {'Время':>6}")
    print(f"{'-' * 50}")

    models = get_models()
    results = []

    for name, cfg in models.items():
        result = train_and_log(
            name, cfg, X_train, X_test, y_train, y_test, X_train_enc, X_test_enc
        )
        results.append(result)

    # Итоговая таблица
    results_df = pd.DataFrame(results).drop(columns=["model", "scaler"])
    results_df = results_df.sort_values("mae")
    print("\nИТОГОВАЯ ТАБЛИЦА:")
    print(results_df.to_string(index=False))

    # Лучшая модель
    best = results_df.iloc[0]
    print(f"\nЛучшая модель: {best['name']}")
    print(f"  MAE={best['mae']:.4f}, RMSE={best['rmse']:.4f}, R²={best['r2']:.4f}")

    # Сохранение CatBoost как финальной
    for r in results:
        if r["name"] == "CatBoost":
            model_path = str(ARTIFACTS_PATH / "drop_forecaster.cbm")
            r["model"].save_model(model_path)
            print(f"\nФинальная модель сохранена: {model_path}")



if __name__ == "__main__":
    main()
