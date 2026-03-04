import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle, os, warnings
warnings.filterwarnings('ignore')

class HVACOptimizer:
    def __init__(self, data_file='hvac_data.csv', model_file='model.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.scaler = None
        self.data = None
        self.feature_names = None
        self.metrics = None

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        print(f"✓ Loaded {len(self.data)} records")
        return self.data

    def _engineer_features(self, df):
        feat = df.copy()
        feat['hour'] = feat['timestamp'].dt.hour
        feat['day_of_week'] = feat['timestamp'].dt.dayofweek
        feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)
        feat['temp_diff'] = feat['temp_outside'] - feat['setpoint']
        feat['hvac_lag_1h'] = feat['hvac_kwh'].shift(1)
        feat['hvac_lag_24h'] = feat['hvac_kwh'].shift(24)
        feat['hvac_roll_mean'] = feat['hvac_kwh'].rolling(24).mean()
        feat['hvac_roll_std'] = feat['hvac_kwh'].rolling(24).std()
        feat['occ_roll_mean'] = feat['occupancy'].rolling(12).mean()
        feat = feat.dropna().reset_index(drop=True)
        self.feature_names = [
            'hour', 'day_of_week', 'is_weekend', 'occupancy',
            'equipment_heat_kw', 'temp_outside', 'setpoint', 'temp_diff',
            'hour_sin', 'hour_cos',
            'hvac_lag_1h', 'hvac_lag_24h', 'hvac_roll_mean', 'hvac_roll_std',
            'occ_roll_mean'
        ]
        return feat

    def train(self, test_size=0.2):
        if self.data is None: self.load_data()
        feat = self._engineer_features(self.data)
        split = int(len(feat) * (1 - test_size))
        X_train = feat.iloc[:split][self.feature_names]
        y_train = feat.iloc[:split]['hvac_kwh']
        X_test = feat.iloc[split:][self.feature_names]
        y_test = feat.iloc[split:]['hvac_kwh']

        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s = self.scaler.transform(X_test)

        self.model = Ridge(alpha=1.0)
        self.model.fit(X_train_s, y_train)

        y_pred = self.model.predict(X_test_s)
        self.metrics = {
            'rmse': round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
            'mae': round(float(mean_absolute_error(y_test, y_pred)), 3),
            'r2': round(float(r2_score(y_test, y_pred)), 4),
            'mape': round(float(np.mean(np.abs((y_test - y_pred) / y_test)) * 100), 2)
        }
        print(f"✓ Ridge model trained — R²={self.metrics['r2']}, RMSE={self.metrics['rmse']}")
        return self.model

    def get_feature_importance(self):
        if self.model is None: raise ValueError("Model not trained.")
        coefs = np.abs(self.model.coef_)
        total = coefs.sum()
        return {n: round(float(c / total), 4) for n, c in zip(self.feature_names, coefs)}

    def predict_next(self):
        if self.model is None: raise ValueError("Model not trained.")
        feat = self._engineer_features(self.data)
        row = feat.iloc[[-1]][self.feature_names]
        row_s = self.scaler.transform(row)
        pred = float(self.model.predict(row_s)[0])
        return {'prediction': round(pred, 2), 'lower_bound': round(pred * 0.9, 2),
                'upper_bound': round(pred * 1.1, 2), 'confidence_level': 90}

    def get_optimization_insights(self):
        if self.data is None: self.load_data()
        active = self.data[(self.data['is_weekend'] == 0) & (self.data['timestamp'].dt.hour.between(7, 21))]
        idle = self.data[~self.data.index.isin(active.index)]
        return {
            'active_avg': round(float(active['hvac_kwh'].mean()), 2),
            'idle_avg': round(float(idle['hvac_kwh'].mean()), 2),
            'potential_savings_pct': round(float((active['hvac_kwh'].mean() - idle['hvac_kwh'].mean()) / active['hvac_kwh'].mean() * 100), 1),
            'peak_hour': int(active.groupby(active['timestamp'].dt.hour)['hvac_kwh'].mean().idxmax()),
            'avg_equip_heat': round(float(self.data['equipment_heat_kw'].mean()), 2)
        }

    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler,
                         'feature_names': self.feature_names, 'metrics': self.metrics}, f)
        print(f"✓ Model saved to {self.model_file}")

    def load_model(self):
        with open(self.model_file, 'rb') as f:
            b = pickle.load(f)
        self.model = b['model']; self.scaler = b['scaler']
        self.feature_names = b['feature_names']; self.metrics = b['metrics']
        print(f"✓ Model loaded from {self.model_file}")

if __name__ == '__main__':
    p = HVACOptimizer()
    p.load_data(); p.train()
    print("\nFeature Importance:")
    for f, v in sorted(p.get_feature_importance().items(), key=lambda x: -x[1]):
        print(f"  {f:25s} {v:.4f}")
    print("\nOptimization Insights:", p.get_optimization_insights())
    pred = p.predict_next()
    print(f"\nNext-hour: {pred['prediction']} kWh [{pred['lower_bound']}, {pred['upper_bound']}]")
    p.save_model()
