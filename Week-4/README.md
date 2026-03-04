# 🍽️ Cafeteria Load Prediction

An interactive dashboard predicting cafeteria electrical load by meal period using Gradient Boosting regression.

## Features

- 🌳 **Gradient Boosting Prediction** — Forecasts next-hour electricity consumption
- 🥐 **Meal Period Analytics** — Breakdown by Breakfast, Lunch, Snacks, Dinner
- 📊 **Interactive Dashboard** — Real-time Plotly.js charts with dark theme
- 🔄 **Auto-refresh** — Dashboard updates every 30 seconds
- 🔥 **Feature Importance** — Discover what drives cafeteria energy usage
- 🗺️ **Weekly Heatmap** — Load by hour and day of week

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate synthetic data**:
   ```bash
   python data_generator.py
   ```

3. **Train the Gradient Boosting model**:
   ```bash
   python model.py
   ```

4. **Start the dashboard**:
   ```bash
   python app.py
   ```

5. **Open your browser**:
   ```
   http://localhost:5003
   ```

## Project Structure

```
Week-4/
├── data_generator.py      # Generates 60 days of synthetic cafeteria data
├── model.py               # Gradient Boosting regressor with meal analysis
├── app.py                 # Flask API server
├── requirements.txt       # Python dependencies
├── cafeteria_data.csv     # Generated data (after running data_generator.py)
├── model.pkl              # Trained model (after running model.py)
├── README.md              # This file
└── static/
    ├── index.html         # Dashboard HTML
    ├── style.css          # Amber/orange themed dark styles
    └── app.js             # Dashboard JavaScript
```

## API Endpoints

- `GET /` — Dashboard homepage
- `GET /api/historical` — Historical footfall and energy data
- `GET /api/predict` — Next-hour prediction with confidence bounds
- `GET /api/stats` — Model metrics and feature importances
- `GET /api/meal-analysis` — Per-meal-period aggregate statistics

## How It Works

1. **Data Generation**: 60 days of hourly data with meal-period spikes (breakfast, lunch, snacks, dinner), footfall patterns, weather, and cooking equipment loads.

2. **Feature Engineering**: Hour, day of week, meal period one-hot encoding, footfall, temperature, lagged values, rolling stats — 20 features total.

3. **Gradient Boosting Model**: 300 estimators, learning rate 0.08, trained on 80% chronological split.

4. **Dashboard**: Six interactive charts — timeline colored by meal period, meal-period energy bars, feature importance, footfall correlation scatter, and weekly heatmap.

## Technologies

- **Backend**: Python, Flask, pandas, scikit-learn
- **Frontend**: HTML, CSS, JavaScript, Plotly.js
- **Model**: Gradient Boosting Regressor (scikit-learn)

## License

MIT License — created for Hack-O-Week.
