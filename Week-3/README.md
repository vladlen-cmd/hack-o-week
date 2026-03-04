# 📚 Library Energy During Exams

An interactive dashboard analyzing library energy consumption patterns during exam periods vs normal operations, using Random Forest prediction.

## Features

- 🌲 **Random Forest Prediction** — Forecasts next-hour electricity consumption
- 📕 **Exam vs Normal Comparison** — Side-by-side energy and occupancy analytics
- 📊 **Interactive Dashboard** — Real-time Plotly.js visualizations with dark theme
- 🔄 **Auto-refresh** — Dashboard updates every 30 seconds
- 🔥 **Feature Importance** — See what drives library energy consumption
- 🗺️ **Weekly Heatmap** — Consumption by hour and day of week

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate synthetic data**:
   ```bash
   python data_generator.py
   ```

3. **Train the Random Forest model**:
   ```bash
   python model.py
   ```

4. **Start the dashboard**:
   ```bash
   python app.py
   ```

5. **Open your browser**:
   ```
   http://localhost:5002
   ```

## Project Structure

```
Week-3/
├── data_generator.py      # Generates 90 days of synthetic library data
├── model.py               # Random Forest regressor with feature engineering
├── app.py                 # Flask API server
├── requirements.txt       # Python dependencies
├── library_data.csv       # Generated data (after running data_generator.py)
├── model.pkl              # Trained model (after running model.py)
├── README.md              # This file
└── static/
    ├── index.html         # Dashboard HTML
    ├── style.css          # Emerald-themed dark styles
    └── app.js             # Dashboard JavaScript
```

## API Endpoints

- `GET /` — Dashboard homepage
- `GET /api/historical` — Historical occupancy and energy data
- `GET /api/predict` — Next-hour prediction with confidence bounds
- `GET /api/stats` — Model metrics and feature importances
- `GET /api/exam-comparison` — Exam vs normal aggregate statistics

## How It Works

1. **Data Generation**: 90 days of synthetic hourly data with two exam periods (midterms ~day 30-37, finals ~day 75-85). During exams, library occupancy increases 2-3× with extended late-night hours.

2. **Feature Engineering**: Hour, day of week, exam flag, occupancy, temperature, lagged electricity values, rolling statistics — 16 features total.

3. **Random Forest Model**: 200 trees, max depth 15, trained on 80% chronological split. Achieves high R² with low MAPE.

4. **Dashboard**: Six interactive charts — timeline with exam zone shading, hourly exam vs normal comparison, feature importance, occupancy correlation scatter, and weekly heatmap.

## Technologies

- **Backend**: Python, Flask, pandas, scikit-learn
- **Frontend**: HTML, CSS, JavaScript, Plotly.js
- **Model**: Random Forest Regressor (scikit-learn)

## License

MIT License — created for Hack-O-Week.
