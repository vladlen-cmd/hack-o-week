# Classroom Usage Forecasting

A real-time classroom electricity forecasting system that uses Wi-Fi occupancy sensor data to predict next-hour electricity consumption using ARIMA time series modeling.

## Features

- 📊 **ARIMA Time Series Forecasting** - Predicts next-hour electricity consumption
- 📡 **Wi-Fi Occupancy Tracking** - Synthetic sensor data simulating device connections
- 📈 **Confidence Intervals** - 95% prediction confidence bands
- 🎨 **Interactive Dashboard** - Real-time visualization with Plotly
- 🔄 **Auto-refresh** - Dashboard updates every 30 seconds
- 📉 **Correlation Analysis** - Visualize occupancy vs electricity relationship

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Generate synthetic data**:
   ```bash
   python data_generator.py
   ```

3. **Train the ARIMA model**:
   ```bash
   python arima_model.py
   ```

4. **Start the dashboard**:
   ```bash
   python app.py
   ```

5. **Open your browser** and navigate to:
   ```
   http://localhost:5000
   ```

## Project Structure

```
.
├── data_generator.py      # Generates synthetic Wi-Fi occupancy data
├── arima_model.py         # ARIMA forecasting model implementation
├── app.py                 # Flask API server
├── requirements.txt       # Python dependencies
├── classroom_data.csv     # Generated data (created after running data_generator.py)
├── arima_model.pkl        # Trained model (created after running arima_model.py)
└── static/
    ├── index.html         # Dashboard HTML
    ├── style.css          # Dashboard styles
    └── app.js             # Dashboard JavaScript
```

## API Endpoints

- `GET /` - Dashboard homepage
- `GET /api/historical` - Returns historical occupancy and electricity data
- `GET /api/predict` - Returns next-hour prediction with confidence intervals
- `POST /api/retrain` - Retrains the ARIMA model with latest data
- `GET /api/stats` - Returns model evaluation metrics

## How It Works

1. **Data Generation**: Synthetic Wi-Fi occupancy data is generated with realistic patterns (higher during class hours, lower at night)

2. **Correlation**: Electricity consumption is correlated with occupancy (base load + occupancy-dependent load + HVAC)

3. **ARIMA Modeling**: The ARIMA(2,1,2) model is trained on historical electricity consumption data

4. **Forecasting**: The model predicts next-hour electricity draw with 95% confidence intervals

5. **Visualization**: The dashboard displays:
   - Next-hour prediction with confidence bounds
   - Historical time series with forecast
   - Occupancy vs electricity correlation
   - Daily usage patterns

## Technologies

- **Backend**: Python, Flask, pandas, statsmodels
- **Frontend**: HTML, CSS, JavaScript, Plotly.js
- **Model**: ARIMA (AutoRegressive Integrated Moving Average)

## Model Details

- **Model Type**: ARIMA(2, 1, 2)
  - p=2: Autoregressive terms
  - d=1: Differencing order
  - q=2: Moving average terms
- **Confidence Level**: 95%
- **Training Data**: 30 days of hourly readings (720 data points)

## Customization

### Adjust Data Generation

Edit `data_generator.py` to change:
- Number of days: `generate_classroom_data(days=30)`
- Occupancy patterns
- Electricity correlation factors

### Modify ARIMA Parameters

Edit `arima_model.py` to change:
- Model order: `self.order = (p, d, q)`
- Confidence level: `predict_next_hour(alpha=0.05)`

### Dashboard Refresh Rate

Edit `static/app.js`:
```javascript
const REFRESH_INTERVAL = 30000; // milliseconds
```
