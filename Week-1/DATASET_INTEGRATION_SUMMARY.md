# Dorm Energy Consumption Dataset - Integration Summary

## ✅ What Was Done

### 1. **Data Extraction from PDF**
   - Extracted 168 hourly records from the "Dorm_Energy_Consumption_7_Days.pdf"
   - Date range: January 1-7, 2026
   - Converted energy consumption from kWh to MW for consistency with the dashboard

### 2. **Dataset Details**
   - **Records**: 168 hours (7 days)
   - **Time Range**: 2026-01-01 00:00:00 to 2026-01-07 23:00:00
   - **Load Range**: 0.014 - 0.074 MW (14-74 kWh)
   - **Format**: CSV with columns `datetime` and `load_mw`
   - **Location**: `data/dorm_energy_7days.csv`

### 3. **Code Adaptations**
   - Modified `data_processor.py` to handle small datasets (< 14 days)
   - Adaptive lag features: Uses 24h and 48h lags for 7-day dataset (instead of 168h)
   - Updated `run_dashboard.py` to load the dorm energy dataset

### 4. **Model Performance**
   - **Training**: 96 samples
   - **Testing**: 24 samples
   - **Metrics**:
     - RMSE: 0.00 MW
     - MAE: 0.00 MW
     - R² Score: 0.30
     - MAPE: 4.57%

### 5. **Top Features**
   1. `load_lag_24h` (yesterday's load)
   2. `load_rolling_std_7d` (volatility)
   3. `load_rolling_mean_7d` (average trend)
   4. `load_rolling_max_7d` (peak capacity)
   5. `load_rolling_min_7d` (minimum load)

## 🚀 How to Run

```bash
# Navigate to project directory
cd /Users/vlad/Downloads/Hack-O-Week/January/Week-1

# Run the dashboard
.venv/bin/python run_dashboard.py
```

Then open your browser to: **http://localhost:8050**

## 📊 Dashboard Features

The dashboard now displays your dorm energy data with:
- ✅ **Time Series Plot**: Raw vs smoothed consumption patterns
- ✅ **Prediction Plot**: Model predictions vs actual values
- ✅ **Evening Peak Trends**: Focused analysis on 6 PM - 10 PM
- ✅ **Weekly Heatmap**: Consumption patterns by day and hour
- ✅ **Feature Importance**: Which factors drive predictions
- ✅ **Interactive Controls**: Adjust smoothing window (6-72 hours)

## 📁 Generated Files

1. `data/dorm_energy_7days_raw.csv` - Raw extracted data from PDF
2. `data/dorm_energy_7days.csv` - Processed data for dashboard
3. `extract_pdf_data.py` - PDF extraction script
4. `process_pdf_to_csv.py` - Data conversion script

## 🔍 Key Insights

The dataset shows typical dorm energy patterns:
- **Low consumption** during early morning hours (2-6 AM): ~14-20 kWh
- **Moderate consumption** during daytime (9 AM - 5 PM): ~27-40 kWh
- **Peak consumption** during evening (6 PM - 10 PM): ~50-74 kWh
- **Evening peaks** are ~3-4x higher than nighttime lows

This confirms the dataset description: "evening energy consumption is higher than morning usage."

## 🎯 Next Steps

You can now:
1. View the interactive dashboard at http://localhost:8050
2. Adjust the smoothing window to see different trend patterns
3. Analyze prediction accuracy for evening peaks
4. Export visualizations for presentations
5. Extend the dataset with more days for better predictions

---

**Status**: ✅ Dashboard running successfully with 7-day dorm energy dataset!
