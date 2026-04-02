# Hack-O-Week

> 16-week Set up backend service to monitor streams, detect anomalies (e.g., high BPM), and push encrypted notifications.

---

## January

**Week 1** — Collect hourly meter data from dorms; apply moving average smoothing and linear regression to predict evening peaks based on past week. Visualize trends on a live Plotly dashboard.

**Week 2** — Use sensor data (occupancy via Wi-Fi logs) to train a simple ARIMA model for next-hour room electricity draw; dashboard shows confidence intervals.

**Week 3** — Predict lunch-hour surges using temperature/weather data and **Linear Regression**; real-time line chart updates via **WebSocket** integration.

**Week 4** — Predict cafeteria lunch-hour surges using temperature/weather data and **Linear Regression**; real-time line chart updates via **WebSocket** integration.

---

## February

**Week 5** — Train a basic **Decision Tree** on occupancy/temperature data to forecast cooling needs; dashboard with **heatmaps** for zone-wise predictions across 4 lab zones.

**Week 6** — Use **RNN (simple LSTM)** on hourly patterns to predict post-event electricity; interactive dashboard filters by **day type** (weekday/weekend/event).

**Week 7** — Apply **K-Means clustering** on usage profiles then regress clusters for forecasts; **pie charts** show savings potential on dashboard.

**Week 8** — Sensor-based vehicle count data fed into **Polynomial Regression** for light usage; real-time **bar chart** with alerts for **anomalies**.

**Week 9** — Time-series data with **Naive Bayes** for usage categories, then forecast via **Prophet**; dashboard **timeline slider** for what-if scenarios.

**Week 10** — Campus-wide sustainability tracker with **Ensemble model** (LR + exponential smoothing); multi-building **drill-down** with **carbon KPI** cards and solar tracking.

---

## March

**Week 11** — User registration portal with **JWT authentication** (PyJWT), **encrypted profiles** (SHA-256), and **wearable sync** endpoint; professional login/register UI with user directory.

**Week 12** — Wearable data ingestion via **WebSocket** (flask-socketio); incoming data encrypted with **Fernet** before database insert; live heart rate chart.

**Week 13** — Dashboard visualization built with **React** (CDN); fetches and displays **decrypted** wearable activity data (steps, HR, calories, sleep) with daily trends.

**Week 14** — Data encryption pipeline: **CryptoJS** (client-side AES) → **Fernet** (server-side) → **SQLite** storage; dual-layer encrypt/decrypt with pipeline timing metrics.

**Week 15** — Basic anomaly detection on **heart rate** data using **Isolation Forest**; confusion matrix, anomaly score histogram, and REST API to check individual readings.

---

## April

**Week 16** — Set up backend service to **monitor streams**, **detect anomalies** (e.g., high BPM), and push encrypted notifications.

---

## Quick Start

```bash
cd Projects/Week-<N>
pip install -r requirements.txt
python3 data_generator.py      # generate synthetic data (skip for Week-11, 12, 14)
python3 model.py               # train ML model (skip for Week-11, 12, 14)
python3 app.py                 # start dashboard
```

> **Note:** Week-16 uses `detector.py` instead of `model.py`.

## Tech Stack

| Layer | Tools |
|-------|-------|
| Backend | Flask, Flask-CORS, Flask-SocketIO, SQLite |
| ML | scikit-learn (LinearRegression, DecisionTree, KMeans, PolynomialFeatures, GaussianNB, IsolationForest), TensorFlow/Keras (LSTM) |
| Forecast | Prophet (with moving-average fallback) |
| Auth | PyJWT (JSON Web Tokens), SHA-256 |
| Crypto | cryptography (Fernet), CryptoJS (client-side AES) |
| Frontend | Plotly.js, React 18 (CDN), Font Awesome 6, Socket.IO, vanilla CSS (dark themes) |
| Data | pandas, NumPy, synthetic generators |
