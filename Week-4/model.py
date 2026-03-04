import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


class CafeteriaLoadPredictor:
    def __init__(self, data_file='cafeteria_data.csv', model_file='model.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.data = None
        self.feature_names = None
        self.metrics = None
    
    def load_data(self):
        """Load and parse cafeteria CSV."""
        self.data = pd.read_csv(self.data_file)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        print(f"✓ Loaded {len(self.data)} records")
        return self.data
    
    def _engineer_features(self, df):
        """Build features for Gradient Boosting."""
        feat = df.copy()
        feat['hour'] = feat['timestamp'].dt.hour
        feat['day_of_week'] = feat['timestamp'].dt.dayofweek
        feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)
        feat['dow_sin'] = np.sin(2 * np.pi * feat['day_of_week'] / 7)
        feat['dow_cos'] = np.cos(2 * np.pi * feat['day_of_week'] / 7)
        
        # One-hot encode meal period
        for mp in range(5):
            feat[f'meal_{mp}'] = (feat['meal_period'] == mp).astype(int)
        
        # Lagged electricity
        feat['elec_lag_1h'] = feat['electricity_kwh'].shift(1)
        feat['elec_lag_24h'] = feat['electricity_kwh'].shift(24)
        feat['elec_lag_168h'] = feat['electricity_kwh'].shift(168)
        
        # Rolling stats
        feat['elec_roll_mean_24h'] = feat['electricity_kwh'].rolling(24).mean()
        feat['elec_roll_std_24h'] = feat['electricity_kwh'].rolling(24).std()
        feat['footfall_roll_mean_24h'] = feat['footfall'].rolling(24).mean()
        
        feat = feat.dropna().reset_index(drop=True)
        
        self.feature_names = [
            'hour', 'day_of_week', 'is_weekend', 'footfall', 'temperature',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'meal_0', 'meal_1', 'meal_2', 'meal_3', 'meal_4',
            'elec_lag_1h', 'elec_lag_24h', 'elec_lag_168h',
            'elec_roll_mean_24h', 'elec_roll_std_24h', 'footfall_roll_mean_24h'
        ]
        
        return feat
    
    def train(self, test_size=0.2):
        """Train the Gradient Boosting model."""
        if self.data is None:
            self.load_data()
        
        print("\nEngineering features...")
        feat = self._engineer_features(self.data)
        
        split_idx = int(len(feat) * (1 - test_size))
        train = feat.iloc[:split_idx]
        test = feat.iloc[split_idx:]
        
        X_train = train[self.feature_names]
        y_train = train['electricity_kwh']
        X_test = test[self.feature_names]
        y_test = test['electricity_kwh']
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set:     {len(X_test)} samples")
        
        print("\nTraining Gradient Boosting Regressor...")
        self.model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        self.metrics = {
            'rmse': round(float(rmse), 3),
            'mae': round(float(mae), 3),
            'r2': round(float(r2), 4),
            'mape': round(float(mape), 2),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        print(f"\n✓ Model trained successfully")
        print(f"  RMSE:  {rmse:.3f} kWh")
        print(f"  MAE:   {mae:.3f} kWh")
        print(f"  R²:    {r2:.4f}")
        print(f"  MAPE:  {mape:.2f}%")
        
        return self.model
    
    def get_feature_importance(self):
        """Return feature importances."""
        if self.model is None:
            raise ValueError("Model not trained.")
        importances = self.model.feature_importances_
        return {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}
    
    def predict_next(self):
        """Predict next hour's electricity."""
        if self.model is None:
            raise ValueError("Model not trained.")
        if self.data is None:
            self.load_data()
        
        feat = self._engineer_features(self.data)
        last_row = feat.iloc[[-1]][self.feature_names]
        prediction = float(self.model.predict(last_row)[0])
        
        # Staged predictions for confidence estimation
        staged_preds = list(self.model.staged_predict(last_row))
        recent_preds = [float(p[0]) for p in staged_preds[-50:]]
        std = float(np.std(recent_preds))
        
        return {
            'prediction': round(prediction, 2),
            'lower_bound': round(prediction - 1.96 * std, 2),
            'upper_bound': round(prediction + 1.96 * std, 2),
            'confidence_level': 95
        }
    
    def get_meal_analysis(self):
        """Return per-meal-period statistics."""
        if self.data is None:
            self.load_data()
        
        meal_names = {0: 'Off-hours', 1: 'Breakfast', 2: 'Lunch', 3: 'Snacks', 4: 'Dinner'}
        result = {}
        
        for mp, name in meal_names.items():
            subset = self.data[self.data['meal_period'] == mp]
            result[name] = {
                'avg_electricity': round(float(subset['electricity_kwh'].mean()), 2),
                'max_electricity': round(float(subset['electricity_kwh'].max()), 2),
                'avg_footfall': round(float(subset['footfall'].mean()), 1),
                'max_footfall': int(subset['footfall'].max()),
                'total_records': len(subset)
            }
        
        return result
    
    def save_model(self):
        if self.model is None:
            raise ValueError("No model to save.")
        with open(self.model_file, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'metrics': self.metrics
            }, f)
        print(f"✓ Model saved to {self.model_file}")
    
    def load_model(self):
        if not os.path.exists(self.model_file):
            raise FileNotFoundError(f"Model file {self.model_file} not found")
        with open(self.model_file, 'rb') as f:
            bundle = pickle.load(f)
        self.model = bundle['model']
        self.feature_names = bundle['feature_names']
        self.metrics = bundle['metrics']
        print(f"✓ Model loaded from {self.model_file}")


if __name__ == '__main__':
    print("=" * 60)
    print("Cafeteria Load Predictor — Gradient Boosting")
    print("=" * 60)
    
    predictor = CafeteriaLoadPredictor()
    predictor.load_data()
    predictor.train()
    
    print("\n" + "=" * 60)
    print("Feature Importance:")
    print("=" * 60)
    for feat, imp in sorted(predictor.get_feature_importance().items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:25s} {imp:.4f} {bar}")
    
    print("\n" + "=" * 60)
    print("Meal Period Analysis:")
    print("=" * 60)
    for meal, stats in predictor.get_meal_analysis().items():
        print(f"  {meal:12s}: avg={stats['avg_electricity']:.2f} kWh, "
              f"max={stats['max_electricity']:.2f} kWh, "
              f"avg footfall={stats['avg_footfall']:.0f}")
    
    print("\n" + "=" * 60)
    print("Next-Hour Prediction:")
    print("=" * 60)
    pred = predictor.predict_next()
    print(f"  Predicted: {pred['prediction']} kWh")
    print(f"  95% CI: [{pred['lower_bound']}, {pred['upper_bound']}] kWh")
    
    predictor.save_model()
    print("\n✓ All done!")
