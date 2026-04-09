from __future__ import annotations

from pathlib import Path
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from data_generator import save_dataset
from model import train_predict_lstm

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sleep_activity_data.csv"
PORT = 5016

app = Flask(__name__, static_folder=str(BASE_DIR / "static"))
CORS(app)

_cache: dict | None = None


def ensure_data() -> None:
    if not DATA_PATH.exists():
        save_dataset(str(DATA_PATH), rows=720, seed=42)


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "dataset_exists": DATA_PATH.exists()})


@app.route("/api/regenerate", methods=["POST"])
def regenerate():
    global _cache
    df = save_dataset(str(DATA_PATH), rows=720)
    _cache = None
    return jsonify({"success": True, "rows": int(len(df))})


@app.route("/api/predictions")
def predictions():
    global _cache
    ensure_data()

    if _cache is None:
        bundle = train_predict_lstm(str(DATA_PATH), window=12, epochs=12, batch_size=32)
        # Keep payload responsive in browser while preserving trend visibility.
        rows = bundle.rows[-240:]
        _cache = {
            "success": True,
            "metrics": bundle.metrics,
            "threshold": bundle.threshold,
            "rows": rows,
            "total_rows": len(bundle.rows),
        }

    return jsonify(_cache)


if __name__ == "__main__":
    ensure_data()
    print("=" * 60)
    print("Advanced ML Analytics - LSTM Sleep/Activity Anomaly Prediction")
    print("=" * 60)
    print(f"Dataset: {DATA_PATH}")
    print(f"Dashboard: http://localhost:{PORT}")
    app.run(port=PORT, debug=False)
