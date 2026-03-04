# ❄️ HVAC Optimization in Labs

Predict and optimize HVAC energy in university labs using Ridge Regression, accounting for equipment heat, occupancy, and outdoor temperature.

## Features
- 🏔️ **Ridge Regression** — Predicts next-hour HVAC energy with regularization
- 💡 **Optimization Insights** — Active vs idle savings potential
- 📊 **Interactive Dashboard** — Dark blue/cyan theme, 6 Plotly.js charts
- 🔄 **Auto-refresh** every 30 seconds

## Quick Start
```bash
pip install -r requirements.txt
python data_generator.py
python model.py
python app.py  # → http://localhost:5004
```

## Project Structure
```
Week-5/
├── data_generator.py      # 75 days of HVAC lab data
├── model.py               # Ridge Regression + optimization insights
├── app.py                 # Flask API (port 5004)
├── requirements.txt
├── static/
│   ├── index.html
│   ├── style.css
│   └── app.js
└── README.md
```

## API Endpoints
- `GET /api/historical` — Historical HVAC data
- `GET /api/predict` — Next-hour forecast
- `GET /api/stats` — Model metrics & feature importance
- `GET /api/optimization` — Energy savings insights

## License
MIT — Hack-O-Week
