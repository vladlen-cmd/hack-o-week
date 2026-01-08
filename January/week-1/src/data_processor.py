import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataProcessor:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.df = None
        self.smoothed_df = None
        
    def load_data(self, data_path=None):
        if data_path:
            self.data_path = data_path
            
        if not self.data_path:
            print("No data path provided. Generating synthetic data for demonstration...")
            self.df = self._generate_synthetic_data()
        else:
            try:
                self.df = pd.read_csv(self.data_path, parse_dates=['datetime'])
                print(f"Loaded {len(self.df)} records from {self.data_path}")
            except FileNotFoundError:
                print(f"File not found: {self.data_path}")
                print("Generating synthetic data for demonstration...")
                self.df = self._generate_synthetic_data()
            except Exception as e:
                print(f"Error loading data: {e}")
                print("Generating synthetic data for demonstration...")
                self.df = self._generate_synthetic_data()
        return self.df
    
    def _generate_synthetic_data(self):
        start_date = datetime.now() - timedelta(days=60)
        dates = pd.date_range(start=start_date, periods=60*24, freq='h')
        base_load = 1000
        consumption = []  # Initialize consumption list
        for dt in dates:
            hour = dt.hour
            day_of_week = dt.dayofweek
            if 0 <= hour < 6:  # Night (low)
                hourly_factor = 0.6 + np.random.normal(0, 0.05)
            elif 6 <= hour < 9:  # Morning ramp-up
                hourly_factor = 0.7 + (hour - 6) * 0.1 + np.random.normal(0, 0.05)
            elif 9 <= hour < 17:  # Day (moderate-high)
                hourly_factor = 0.9 + np.random.normal(0, 0.08)
            elif 17 <= hour < 22:  # Evening peak
                hourly_factor = 1.1 + (hour - 17) * 0.05 + np.random.normal(0, 0.08)
            else:  # Late evening
                hourly_factor = 0.8 + np.random.normal(0, 0.05)
            
            if day_of_week >= 5:
                hourly_factor *= 0.95
            week_number = (dt - start_date).days // 7
            trend_factor = 1 + (week_number * 0.01)
            
            load = base_load * hourly_factor * trend_factor
            consumption.append(max(0, load))
        
        df = pd.DataFrame({
            'datetime': dates,
            'load_mw': consumption
        })
        
        print(f"Generated {len(df)} synthetic records (60 days of hourly data)")
        return df
    
    def clean_data(self):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        initial_len = len(self.df)
        self.df = self.df.drop_duplicates(subset=['datetime'])
        
        self.df = self.df.sort_values('datetime').reset_index(drop=True)
        
        self.df = self.df.ffill().bfill()
        
        print(f"Cleaned data: {initial_len} -> {len(self.df)} records")
        return self.df
    
    def apply_moving_average(self, window=24):
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        self.smoothed_df = self.df.copy()
        self.smoothed_df['load_smoothed'] = self.df['load_mw'].rolling(
            window=window, 
            center=True,
            min_periods=1
        ).mean()
        
        print(f"Applied moving average with window={window} hours")
        return self.smoothed_df
    
    def extract_features(self):
        if self.smoothed_df is None:
            self.apply_moving_average()
        
        df = self.smoothed_df.copy()
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 18) & (df['hour'] <= 22)).astype(int)
        
        for lag in [24, 48, 168]:  # 1 day, 2 days, 1 week
            df[f'load_lag_{lag}h'] = df['load_smoothed'].shift(lag)
        
        df['load_rolling_mean_7d'] = df['load_smoothed'].rolling(window=168, min_periods=1).mean()
        df['load_rolling_std_7d'] = df['load_smoothed'].rolling(window=168, min_periods=1).std()
        df['load_rolling_min_7d'] = df['load_smoothed'].rolling(window=168, min_periods=1).min()
        df['load_rolling_max_7d'] = df['load_smoothed'].rolling(window=168, min_periods=1).max()
        
        df = df.dropna()
        
        print(f"Extracted features: {len(df)} records with {len(df.columns)} columns")
        return df
    
    def get_evening_peaks(self, df=None):
        if df is None:
            df = self.smoothed_df if self.smoothed_df is not None else self.df
        
        if df is None:
            raise ValueError("No data available")
    
        evening_df = df[df['datetime'].dt.hour.between(18, 22)].copy()
        print(f"Extracted {len(evening_df)} evening peak records")
        return evening_df
    
    def get_train_test_split(self, df, test_size=0.2):
        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"Split data: {len(train_df)} training, {len(test_df)} testing records")
        return train_df, test_df
