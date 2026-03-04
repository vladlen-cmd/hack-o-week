# ⚡ Peak Hour Electricity Analysis - IIITD Campus Dataset

An interactive dashboard for analyzing electricity consumption data from IIIT Delhi campus buildings, featuring moving average smoothing and linear regression for predicting evening peak loads.

## 📋 Overview

This project analyzes the **I-BLEND (Indian BuiLdings Energy coNsumption Dataset)** - a 52-month energy dataset from seven buildings (commercial and residential) at IIIT Delhi campus. The analysis includes:
- Apply **moving average smoothing** to reduce noise in consumption patterns
- Use **linear regression** to predict evening peak hours (6 PM - 10 PM) based on past week data
- Visualize trends and predictions on a **live Plotly dashboard**

### About the Dataset

The I-BLEND dataset is published in **Nature Scientific Data** and contains:
- **Duration**: 52 months of continuous data
- **Buildings**: 7 buildings (mix of commercial and residential)
- **Sampling Rate**: 1-minute intervals
- **Parameters**: Power (watts), current, voltage, frequency, power factor
- **Location**: IIIT Delhi, India
- **Timezone**: Asia/Kolkata (UTC+5:30)

**Citation**: Rashid, H., Singh, P., Stankovic, V. & Stankovic, L. I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset. Sci Data 6, 190015 (2019). https://doi.org/10.1038/sdata.2019.15

## 🎯 Features

- **Data Processing**: Automated cleaning, feature extraction, and time series analysis
- **Moving Average Smoothing**: Configurable rolling window (default: 24 hours)
- **Predictive Modeling**: Linear regression with multiple features (lagged values, rolling statistics, temporal features)
- **Interactive Dashboard**: Real-time visualizations with Plotly Dash
  - Time series comparison (raw vs smoothed)
  - Actual vs predicted load
  - Evening peak trends
  - Weekly consumption heatmap
  - Feature importance analysis
- **Model Metrics**: RMSE, MAE, R², and MAPE

## 🗂️ Project Structure

```
Week-1/
├── data/                       # Dataset directory
│   └── electricity_data.csv    # (Place your Kaggle dataset here)
├── src/                        # Source code
│   ├── data_processor.py       # Data loading and preprocessing
│   ├── predictor.py            # Linear regression model
│   └── dashboard.py            # Plotly Dash application
├── requirements.txt            # Python dependencies
├── run_dashboard.py            # Main entry point
└── README.md                   # This file
```

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Clone or navigate to the project directory**
   ```bash
   cd /Users/vlad/Downloads/Hack-O-Week/January/Week-1
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download IIITD Dataset**
   
   **Option A: Direct Download**
   ```bash
   cd data
   # Download the dataset (561 MB)
   curl -L -o iiitd_energy_dataset.zip "https://springernature.figshare.com/ndownloader/files/10797959"
   
   # Extract the dataset
   unzip iiitd_energy_dataset.zip
   
   # The dataset contains 7 CSV files (one per building)
   # Use any building file, e.g., Building1.csv
   ```
   
   **Option B: Manual Download**
   - Go to [IIITD Energy Dataset on Figshare](https://springernature.figshare.com/articles/dataset/Energy_dataset_of_IIITD/6007637)
   - Click "Download (561.93 MB)"
   - Extract the zip file in the `data/` directory
   - Choose one of the building CSV files (Building1.csv to Building7.csv)
   
   **Option C: Use Synthetic Data**
   - No action needed! The application will automatically generate realistic synthetic data for demonstration
   
   **Dataset Format**: The IIITD dataset contains columns:
   - `Unix`: Unix timestamp (seconds since Jan 1, 1970)
   - `Power`: Power consumption in watts
   - `Current`, `Voltage`, `Frequency`, `PowerFactor`: Additional electrical parameters

## 🚀 Usage

### Run the Dashboard

```bash
python run_dashboard.py
```

The dashboard will:
1. Load and process the data
2. Train the linear regression model
3. Launch the web server at `http://localhost:8050`

### Access the Dashboard

Open your web browser and navigate to:
```
http://localhost:8050
```

### Interactive Controls

- **Smoothing Window Slider**: Adjust the moving average window (6-72 hours) to see how it affects smoothing and predictions
- **Hover**: Hover over charts to see detailed values
- **Zoom**: Click and drag to zoom into specific time periods

## 📊 Dashboard Components

### 1. **Metrics Cards**
- **RMSE**: Root Mean Square Error (MW)
- **MAE**: Mean Absolute Error (MW)
- **R² Score**: Model fit quality (0-1)
- **MAPE**: Mean Absolute Percentage Error (%)

### 2. **Time Series Plot**
- Compares raw data vs smoothed data
- Shows last 30 days for clarity

### 3. **Prediction Plot**
- Actual vs predicted load on test data
- Visualizes model accuracy

### 4. **Evening Peak Trends**
- Daily maximum evening load (6 PM - 10 PM)
- Trend line showing overall pattern

### 5. **Weekly Heatmap**
- Average consumption by hour and day of week
- Identifies peak consumption patterns

### 6. **Feature Importance**
- Shows which features most influence predictions
- Based on linear regression coefficients

## 🔬 Methodology

### Data Processing
1. **Loading**: Parse CSV with datetime indexing
2. **Cleaning**: Remove duplicates, handle missing values
3. **Smoothing**: Apply rolling mean with configurable window
4. **Feature Engineering**:
   - Temporal: hour, day of week, weekend indicator
   - Lagged: 24h, 48h, 168h (1 week) previous values
   - Rolling statistics: mean, std, min, max over past week

### Modeling
- **Algorithm**: Linear Regression (scikit-learn)
- **Features**: 10 engineered features
- **Train/Test Split**: 80/20 chronological split
- **Evaluation**: RMSE, MAE, R², MAPE

### Prediction
- Forecasts evening peak hours based on historical patterns
- Uses past week data for feature calculation
- Updates dynamically with smoothing window changes

## 🎨 Design

The dashboard features a modern dark theme with:
- **Cyborg Bootstrap theme** for professional appearance
- **Vibrant color scheme**: Cyan, gold, and gradient accents
- **Responsive layout**: Works on different screen sizes
- **Interactive elements**: Smooth animations and hover effects

## 📈 Expected Results

With the IIITD dataset:
- **Data Scale**: Building-level power consumption (typically 0.001-0.1 MW range)
- **R² Score**: Typically 0.70-0.85 (good predictive power)
- **Time Range**: Up to 52 months of continuous data
- **Clear patterns**: Daily cycles, evening peaks, weekend effects, seasonal variations
- **Unique characteristics**: Voltage fluctuations, power failures typical of Indian infrastructure

## 🛠️ Customization

### Change Smoothing Window
Modify in `run_dashboard.py`:
```python
processor.apply_moving_average(window=48)  # Change from 24 to 48 hours
```

### Adjust Train/Test Split
Modify in `run_dashboard.py`:
```python
train_df, test_df = processor.get_train_test_split(df_features, test_size=0.3)
```

### Change Dashboard Port
Modify in `run_dashboard.py`:
```python
dashboard.run(debug=True, port=8080)  # Change from 8050 to 8080
```

## 📚 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: Machine learning (Linear Regression)
- **plotly**: Interactive visualizations
- **dash**: Web dashboard framework
- **dash-bootstrap-components**: UI components

## 🐛 Troubleshooting

### Dashboard won't start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check if port 8050 is available
- Try a different port in `run_dashboard.py`

### No data loaded
- The app will generate synthetic data automatically
- To use real data, place CSV in `data/` directory with columns: `datetime`, `load_mw`

### Poor model performance
- Try adjusting the smoothing window
- Check data quality and completeness
- Ensure sufficient historical data (at least 2 weeks recommended)

## 📝 License

This project is created for educational purposes as part of Hack-O-Week.

## 👥 Authors

Created for Peak Hour Electricity Analysis project using the I-BLEND dataset from IIIT Delhi.

## 📚 Dataset Reference

**I-BLEND Dataset**:
- **Paper**: Rashid, H., Singh, P., Stankovic, V. & Stankovic, L. I-BLEND, a campus-scale commercial and residential buildings electrical energy dataset. *Sci Data* 6, 190015 (2019).
- **DOI**: https://doi.org/10.1038/sdata.2019.15
- **Dataset**: https://springernature.figshare.com/articles/dataset/Energy_dataset_of_IIITD/6007637
- **License**: CC0 (Public Domain)

---

**Enjoy analyzing electricity consumption patterns! ⚡📊**
