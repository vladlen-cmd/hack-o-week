from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler

# Keep TensorFlow logs quiet in normal runs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


@dataclass
class PredictionBundle:
    metrics: dict
    threshold: float
    rows: list[dict]


def _make_sequences(data: np.ndarray, labels: np.ndarray, timestamps: np.ndarray, window: int):
    x, y, t = [], [], []
    for i in range(window, len(data)):
        x.append(data[i - window:i])
        y.append(labels[i])
        t.append(timestamps[i])
    return np.array(x), np.array(y), np.array(t)


def train_predict_lstm(csv_path: str, window: int = 12, epochs: int = 12, batch_size: int = 32) -> PredictionBundle:
    df = pd.read_csv(csv_path)
    if not {"timestamp", "sleep_hours", "activity_minutes", "is_anomaly"}.issubset(df.columns):
        raise ValueError("Dataset missing required columns.")

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    feature_cols = ["sleep_hours", "activity_minutes"]
    features = df[feature_cols].astype(float).values
    labels = df["is_anomaly"].astype(int).values
    timestamps = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M").values

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    x, y, ts = _make_sequences(scaled, labels, timestamps, window)
    if len(x) < 50:
        raise ValueError("Not enough sequence data to train. Increase rows in generated dataset.")

    split = int(len(x) * 0.8)
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    ts_test = ts[split:]

    tf.random.set_seed(42)
    np.random.seed(42)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(x.shape[1], x.shape[2])),
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(16),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.2,
    )

    # Choose threshold from train scores to better separate rare anomalies.
    train_scores = model.predict(x_train, verbose=0).flatten()
    threshold = float(np.quantile(train_scores, 0.92))

    test_scores = model.predict(x_test, verbose=0).flatten()
    y_pred = (test_scores >= threshold).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "samples_test": int(len(y_test)),
        "anomaly_rate_test": round(float(np.mean(y_test)), 4),
    }

    rows = []
    for i in range(len(y_test)):
        rows.append(
            {
                "timestamp": str(ts_test[i]),
                "sleep_hours": round(float(df.iloc[window + split + i]["sleep_hours"]), 2),
                "activity_minutes": round(float(df.iloc[window + split + i]["activity_minutes"]), 2),
                "actual_anomaly": int(y_test[i]),
                "predicted_anomaly": int(y_pred[i]),
                "anomaly_probability": round(float(test_scores[i]), 4),
            }
        )

    return PredictionBundle(metrics=metrics, threshold=round(threshold, 4), rows=rows)
