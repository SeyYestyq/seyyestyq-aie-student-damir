"""
Загрузчик данных для Drop Forecaster.

Поддерживает:
- Загрузку исторических данных из CSV
- Генерацию демо-данных для тестирования
- Экспорт данных в CSV
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Генерация демо-данных

REGIONS = [
    "г. Москва", "г. Санкт-Петербург", "Московская область",
    "Свердловская область", "Новосибирская область",
    "Республика Татарстан", "Краснодарский край",
    "Красноярский край", "Челябинская область", "Нижегородская область",
]

PROCEDURE_TYPES = [
    "Электронный аукцион",
    "Открытый конкурс",
    "Запрос котировок",
    "Запрос предложений",
    "Закупка у единственного поставщика",
]

FZ_TYPES = ["44-ФЗ", "223-ФЗ"]


def generate_demo_dataset(
    n_samples: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Генерация синтетического датасета для демонстрации.

    Данные моделируют реальное распределение снижения цены
    на тендерах в зависимости от числа участников, региона,
    типа процедуры и размера контракта.

    Args:
        n_samples: Число записей.
        seed: Случайное зерно для воспроизводимости.

    Returns:
        DataFrame с синтетическими данными.
    """
    rng = np.random.default_rng(seed)

    # ИНН заказчиков (синтетические)
    inns = [f"770{rng.integers(1000000, 9999999)}" for _ in range(200)]

    data = {
        "customer_inn": rng.choice(inns, n_samples),
        "region": rng.choice(REGIONS, n_samples, p=[
            0.25, 0.15, 0.12, 0.08, 0.07, 0.07, 0.07, 0.06, 0.07, 0.06
        ]),
        "fz_type": rng.choice(FZ_TYPES, n_samples, p=[0.7, 0.3]),
        "procedure_type": rng.choice(PROCEDURE_TYPES, n_samples, p=[
            0.55, 0.15, 0.12, 0.08, 0.10
        ]),
        "nmck_rub": rng.lognormal(mean=13.0, sigma=1.8, size=n_samples),
    }

    df = pd.DataFrame(data)

    # Число участников (Пуассон + смещение по типу процедуры)
    base_participants = rng.poisson(lam=3.5, size=n_samples) + 1
    # Аукционы привлекают больше участников
    auction_mask = df["procedure_type"] == "Электронный аукцион"
    base_participants[auction_mask] += rng.integers(0, 3, size=auction_mask.sum())
    df["participants_count_est"] = base_participants.clip(1, 30)

    # nmck_log1p
    df["nmck_log1p"] = np.log1p(df["nmck_rub"])

    # Моделирование drop_fraction
    # Базовое снижение — зависит от числа участников
    base_drop = 0.03 + 0.015 * df["participants_count_est"]

    # Поправка на размер НМЦК (большие контракты — меньше снижение)
    nmck_adj = -0.01 * (df["nmck_log1p"] - 13.0).clip(-2, 2)

    # Поправка на тип процедуры
    proc_adj = df["procedure_type"].map({
        "Электронный аукцион": 0.02,
        "Открытый конкурс": 0.01,
        "Запрос котировок": 0.03,
        "Запрос предложений": 0.00,
        "Закупка у единственного поставщика": -0.05,
    }).fillna(0)

    # Шум
    noise = rng.normal(0, 0.03, n_samples)

    df["drop_fraction"] = (base_drop + nmck_adj + proc_adj + noise).clip(0.0, 0.85)

    # Среднее снижение по заказчику (агрегат)
    customer_avg = df.groupby("customer_inn")["drop_fraction"].transform("mean")
    df["customer_avg_drop"] = customer_avg + rng.normal(0, 0.01, n_samples)

    # Число контрактов заказчика
    customer_counts = df.groupby("customer_inn")["drop_fraction"].transform("count")
    df["customer_contracts_count"] = customer_counts

    # Округление
    df["drop_fraction"] = df["drop_fraction"].round(4)
    df["nmck_rub"] = df["nmck_rub"].round(2)

    logger.info("Сгенерировано %d демо-записей", len(df))
    return df


# CLI

def main():
    """Точка входа CLI для генерации данных."""
    parser = argparse.ArgumentParser(description="Data Loader для Drop Forecaster")
    parser.add_argument(
        "--generate-demo",
        action="store_true",
        help="Сгенерировать демо-данные",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/history_drop_dataset.csv",
        help="Путь для сохранения данных",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Число записей для генерации",
    )

    args = parser.parse_args()

    if args.generate_demo:
        df = generate_demo_dataset(n_samples=args.n_samples)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Демо-данные сохранены: {output_path} ({len(df)} записей)")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
