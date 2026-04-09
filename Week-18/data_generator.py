from __future__ import annotations

from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def generate_sleep_activity_data(rows: int = 600, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = datetime.now() - timedelta(minutes=30 * rows)

    records = []
    for i in range(rows):
        ts = start + timedelta(minutes=30 * i)
        hour = ts.hour

        # Base daily pattern: higher sleep metric overnight, higher activity daytime.
        if 23 <= hour or hour <= 6:
            sleep_hours = rng.normal(7.4, 0.7)
            activity_minutes = rng.normal(10, 6)
        elif 7 <= hour <= 10:
            sleep_hours = rng.normal(6.3, 0.8)
            activity_minutes = rng.normal(25, 10)
        elif 11 <= hour <= 18:
            sleep_hours = rng.normal(5.9, 0.8)
            activity_minutes = rng.normal(42, 14)
        else:
            sleep_hours = rng.normal(6.4, 0.9)
            activity_minutes = rng.normal(22, 9)

        sleep_hours = float(np.clip(sleep_hours, 2.5, 9.5))
        activity_minutes = float(np.clip(activity_minutes, 0, 120))

        anomaly = 0

        # Inject realistic irregular behavior.
        if rng.random() < 0.08:
            anomaly = 1
            pattern = rng.choice(["overtrained", "underslept", "sudden_drop"])

            if pattern == "overtrained":
                sleep_hours = float(np.clip(rng.normal(3.8, 0.5), 2.5, 5.5))
                activity_minutes = float(np.clip(rng.normal(85, 12), 60, 120))
            elif pattern == "underslept":
                sleep_hours = float(np.clip(rng.normal(3.5, 0.6), 2.5, 5.2))
                activity_minutes = float(np.clip(rng.normal(5, 3), 0, 18))
            else:
                sleep_hours = float(np.clip(rng.normal(8.8, 0.4), 7.8, 9.5))
                activity_minutes = float(np.clip(rng.normal(2, 2), 0, 12))

        records.append(
            {
                "timestamp": ts.isoformat(timespec="minutes"),
                "sleep_hours": round(sleep_hours, 2),
                "activity_minutes": round(activity_minutes, 2),
                "is_anomaly": anomaly,
            }
        )

    return pd.DataFrame(records)


def save_dataset(path: str = "sleep_activity_data.csv", rows: int = 600, seed: int = 42) -> pd.DataFrame:
    df = generate_sleep_activity_data(rows=rows, seed=seed)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    df_out = save_dataset()
    print(f"Generated {len(df_out)} rows -> sleep_activity_data.csv")
