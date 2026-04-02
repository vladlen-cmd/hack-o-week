# Week-16: Real-Time Alert System

Backend service that monitors incoming wearable streams, detects anomalies (for example high BPM), and pushes encrypted notifications in real-time.

## Features
- Real-time ingestion via REST and WebSocket
- Rule-based anomaly detection:
  - High BPM and critical BPM
  - Low BPM
  - Low SpO2
  - High body temperature
  - Sudden BPM jump in short window
- Encrypted alert notifications using Fernet
- SQLite-backed storage for readings and notifications
- Alert acknowledgement endpoint for delivery tracking

## Quick Start
```bash
pip install -r requirements.txt
python app.py
```

Service URL: http://localhost:5015

## REST API
- `GET /api/health` health check
- `POST /api/stream` ingest one stream event
- `GET /api/alerts?limit=25` list encrypted alerts
- `POST /api/alerts/<id>/ack` mark alert delivered
- `POST /api/decrypt` decrypt a ciphertext (demo/admin)
- `GET /api/stats` service metrics

### Sample Ingest
```bash
curl -X POST http://localhost:5015/api/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_id":"student-1001",
    "device":"smart-band",
    "bpm":135,
    "spo2":90,
    "temp_c":38.4
  }'
```

## WebSocket Events
Client can emit:
- `stream_data` with payload matching `/api/stream`

Server emits:
- `stream_processed` summary of each reading
- `encrypted_alert` encrypted notification object
- `ingest_ack` per-client acknowledgement

## Notes
- The encryption key is stored in `alert_fernet.key` and generated automatically on first run.
- Keep `alert_fernet.key` private if you deploy this service.
