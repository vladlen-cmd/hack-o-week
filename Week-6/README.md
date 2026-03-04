# 🏟️ Sports Facility Night Usage

Predict sports facility energy consumption with focus on evening/night usage patterns using SVR (Support Vector Regression).

## Quick Start
```bash
pip install -r requirements.txt
python data_generator.py
python model.py
python app.py  # → http://localhost:5005
```

## Features
- 🌙 **Night vs Day Analytics** — Compare energy patterns during night hours
- 🤖 **SVR Prediction** — Support Vector Regression for next-hour forecast
- 💡 **Floodlight Tracking** — Monitor high-energy outdoor lighting
- 🏆 **Event Detection** — Tournament energy spikes

## Structure
```
Week-6/
├── data_generator.py  · model.py  · app.py  · requirements.txt
└── static/ (index.html · style.css · app.js)
```

## API: `/api/historical` · `/api/predict` · `/api/stats` · `/api/night-analysis`

## License: MIT — Hack-O-Week
