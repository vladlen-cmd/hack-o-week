import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import warnings
import pickle
import os

warnings.filterwarnings('ignore')

class ClassroomElectricityForecaster:
    def __init__(self, data_file='classroom_data.csv', model_file='arima_model.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.data = None
        self.order = (2, 1, 2)
        
    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp')
        self.data.set_index('timestamp', inplace=True)
        print(f"✓ Loaded {len(self.data)} records")
        return self.data
    
    def check_stationarity(self, series):
        result = adfuller(series.dropna())
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")
        
        if result[1] <= 0.05:
            print("✓ Series is stationary")
            return True
        else:
            print("⚠ Series is non-stationary (differencing needed)")
            return False
    
    def train(self):
        if self.data is None:
            self.load_data()
        
        print("\nTraining ARIMA model...")
        print(f"Using order: {self.order}")
        y = self.data['electricity_kwh']
        print("\nStationarity check:")
        self.check_stationarity(y)

        self.model = ARIMA(y, order=self.order)
        self.model_fit = self.model.fit()
        
        print("\n✓ Model trained successfully")
        print(f"AIC: {self.model_fit.aic:.2f}")
        print(f"BIC: {self.model_fit.bic:.2f}")
        
        return self.model_fit
    
    def predict_next_hour(self, steps=1, alpha=0.05):
        if self.model_fit is None:
            raise ValueError("Model not trained. Call train() first.")

        forecast = self.model_fit.forecast(steps=steps, alpha=alpha)
        forecast_df = self.model_fit.get_forecast(steps=steps, alpha=alpha).summary_frame()
        
        results = {
            'prediction': float(forecast.iloc[0]) if steps == 1 else forecast.tolist(),
            'lower_ci': float(forecast_df['mean_ci_lower'].iloc[0]) if steps == 1 else forecast_df['mean_ci_lower'].tolist(),
            'upper_ci': float(forecast_df['mean_ci_upper'].iloc[0]) if steps == 1 else forecast_df['mean_ci_upper'].tolist(),
            'confidence_level': int((1 - alpha) * 100)
        }
        
        return results
    
    def save_model(self):
        if self.model_fit is None:
            raise ValueError("No model to save. Train the model first.")
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.model_fit, f)
        
        print(f"✓ Model saved to {self.model_file}")
    
    def load_model(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file {self.model_file} not found")
        
        with open(self.model_file, 'rb') as f:
            self.model_fit = pickle.load(f)
        
        print(f"✓ Model loaded from {self.model_file}")
        return self.model_fit
    
    def evaluate(self):
        if self.model_fit is None:
            raise ValueError("Model not trained. Call train() first.")

        residuals = self.model_fit.resid
        
        print("\nModel Evaluation:")
        print(f"Mean Absolute Error: {np.mean(np.abs(residuals)):.3f} kWh")
        print(f"Root Mean Squared Error: {np.sqrt(np.mean(residuals**2)):.3f} kWh")
        print(f"Mean Residual: {np.mean(residuals):.3f} kWh")
        
        return {
            'mae': float(np.mean(np.abs(residuals))),
            'rmse': float(np.sqrt(np.mean(residuals**2))),
            'mean_residual': float(np.mean(residuals))
        }

if __name__ == '__main__':
    print("=" * 60)
    print("Classroom Electricity Forecasting - ARIMA Model")
    print("=" * 60)

    forecaster = ClassroomElectricityForecaster()
    forecaster.load_data()
    forecaster.train()
    forecaster.evaluate()

    print("\n" + "=" * 60)
    print("Next-Hour Prediction:")
    print("=" * 60)
    prediction = forecaster.predict_next_hour()
    print(f"Predicted electricity: {prediction['prediction']:.2f} kWh")
    print(f"{prediction['confidence_level']}% Confidence Interval: [{prediction['lower_ci']:.2f}, {prediction['upper_ci']:.2f}] kWh")

    forecaster.save_model()
    print("\n✓ All done!")
