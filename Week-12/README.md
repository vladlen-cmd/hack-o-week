# ⌚ Wearable Data Ingestion
Real-time wearable health data ingestion pipeline with multi-user analytics dashboard.

## Quick Start
```bash
pip install -r requirements.txt && python data_simulator.py && python app.py
```
Dashboard: http://localhost:5011

## Features
- 💓 **Heart Rate Streaming** — Multi-user real-time HR visualization
- 👟 **Step Tracking** — Daily and hourly step patterns
- 🫁 **SpO2 Monitoring** — Blood oxygen saturation trends
- 📡 **POST Ingest API** — Accept new data points via `/api/ingest`
- 📱 **Device Registry** — Track 5 different wearable device types

## API
- `POST /api/ingest` · `GET /api/data` · `GET /api/users` · `GET /api/stats` · `GET /api/hourly`

## License: MIT — Hack-O-Week
