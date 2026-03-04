# 🏢 Admin Building Weekend Dip
Analyze admin building energy dips on weekends and Monday morning surges using ElasticNet Regression.

## Quick Start
```bash
pip install -r requirements.txt && python data_generator.py && python model.py && python app.py
```
Dashboard: http://localhost:5006

## Features
- 📉 **Weekend Dip Analysis** — Quantify energy savings on weekends
- 📈 **Monday Surge Detection** — Track morning ramp-up patterns
- 🤖 **ElasticNet Regression** — L1+L2 regularized prediction
- 📊 **6 Interactive Charts** — Day-of-week profiles, heatmaps, trends

## API: `/api/historical` · `/api/predict` · `/api/stats` · `/api/weekend-dip`
## License: MIT — Hack-O-Week
