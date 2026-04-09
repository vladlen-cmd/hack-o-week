# Advanced ML Analytics (Week 18)
LSTM-based anomaly prediction from sleep/activity patterns with interactive visualization.

## Features
- LSTM model trained on sequence windows of sleep hours and activity minutes
- Anomaly probability prediction with adaptive thresholding
- Professional dashboard with:
  - model metrics (accuracy, precision, recall, f1)
  - anomaly probability trend and threshold line
  - sleep/activity trend chart
  - recent predicted anomaly table
- Dataset regeneration endpoint for new simulation runs

## Quick Start
```bash
pip install -r requirements.txt
python app.py
```

Open: http://localhost:5016

## API
- `GET /api/health` - app health
- `POST /api/regenerate` - regenerate synthetic dataset
- `GET /api/predictions` - train LSTM and fetch predictions/metrics

## Data Schema
Generated file: `sleep_activity_data.csv`
- `timestamp`
- `sleep_hours`
- `activity_minutes`
- `is_anomaly`
