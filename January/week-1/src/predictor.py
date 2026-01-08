"""
Predictor Module
Implements linear regression model for predicting evening electricity peaks.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class PeakPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.feature_columns = None
        self.is_trained = False
        self.metrics = {}
        
    def prepare_features(self, df):
        self.feature_columns = [
            'hour',
            'day_of_week',
            'is_weekend',
            'load_lag_24h',
            'load_lag_48h',
            'load_lag_168h',
            'load_rolling_mean_7d',
            'load_rolling_std_7d',
            'load_rolling_min_7d',
            'load_rolling_max_7d'
        ]
        
        # Ensure all feature columns exist
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")
        
        X = df[self.feature_columns].values
        y = df['load_smoothed'].values
        
        return X, y
    
    def train(self, train_df):
        X_train, y_train = self.prepare_features(train_df)
        
        print(f"Training model on {len(X_train)} samples with {len(self.feature_columns)} features...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        print(f"Training completed!")
        print(f"  RMSE: {train_rmse:.2f} MW")
        print(f"  MAE: {train_mae:.2f} MW")
        print(f"  R²: {train_r2:.4f}")
        
        return self.model
    
    def predict(self, test_df):
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test, _ = self.prepare_features(test_df)
        predictions = self.model.predict(X_test)
        
        return predictions
    
    def evaluate(self, test_df):
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test, y_test = self.prepare_features(test_df)
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }
        
        print(f"\nModel Evaluation:")
        print(f"  RMSE: {rmse:.2f} MW")
        print(f"  MAE: {mae:.2f} MW")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return self.metrics
    
    def get_feature_importance(self):
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'coefficient': self.model.coef_
        })
        importance_df['abs_coefficient'] = np.abs(importance_df['coefficient'])
        importance_df = importance_df.sort_values('abs_coefficient', ascending=False)
        
        return importance_df
    
    def predict_evening_peaks(self, df, days_ahead=7):
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Get the last date in the dataset
        last_date = df['datetime'].max()
        
        # Create future dates for evening hours (6 PM - 10 PM)
        future_dates = []
        for day in range(1, days_ahead + 1):
            for hour in range(18, 23):  # 6 PM to 10 PM
                future_dates.append(last_date + pd.Timedelta(days=day, hours=hour-last_date.hour))
        
        # Create a DataFrame for future predictions
        future_df = pd.DataFrame({'datetime': future_dates})
        
        # Extract features (simplified - using last known values)
        future_df['hour'] = future_df['datetime'].dt.hour
        future_df['day_of_week'] = future_df['datetime'].dt.dayofweek
        future_df['is_weekend'] = (future_df['day_of_week'] >= 5).astype(int)
        
        # Use recent historical values for lagged features
        recent_mean = df['load_smoothed'].tail(168).mean()
        recent_std = df['load_smoothed'].tail(168).std()
        recent_min = df['load_smoothed'].tail(168).min()
        recent_max = df['load_smoothed'].tail(168).max()
        
        future_df['load_lag_24h'] = recent_mean
        future_df['load_lag_48h'] = recent_mean
        future_df['load_lag_168h'] = recent_mean
        future_df['load_rolling_mean_7d'] = recent_mean
        future_df['load_rolling_std_7d'] = recent_std
        future_df['load_rolling_min_7d'] = recent_min
        future_df['load_rolling_max_7d'] = recent_max
        
        # Make predictions
        X_future = future_df[self.feature_columns].values
        future_df['predicted_load'] = self.model.predict(X_future)
        
        return future_df[['datetime', 'predicted_load']]
