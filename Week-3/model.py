import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pickle
import os
import warnings

warnings.filterwarnings('ignore')


class LibraryEnergyPredictor:
    def __init__(self, data_file='library_data.csv', model_file='model.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.data = None
        self.feature_names = None
        self.metrics = None
    
    def load_data(self):
        """Load and parse the library energy CSV."""
        self.data = pd.read_csv(self.data_file)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        print(f"✓ Loaded {len(self.data)} records")
        return self.data
    
    def _engineer_features(self, df):
        """Create features for the Random Forest model."""
        feat = df.copy()
        feat['hour'] = feat['timestamp'].dt.hour
        feat['day_of_week'] = feat['timestamp'].dt.dayofweek
        feat['is_weekend'] = (feat['day_of_week'] >= 5).astype(int)
        feat['hour_sin'] = np.sin(2 * np.pi * feat['hour'] / 24)
        feat['hour_cos'] = np.cos(2 * np.pi * feat['hour'] / 24)
        feat['dow_sin'] = np.sin(2 * np.pi * feat['day_of_week'] / 7)
        feat['dow_cos'] = np.cos(2 * np.pi * feat['day_of_week'] / 7)
        
        # Lagged features
        feat['electricity_lag_1h'] = feat['electricity_kwh'].shift(1)
        feat['electricity_lag_24h'] = feat['electricity_kwh'].shift(24)
        feat['electricity_lag_168h'] = feat['electricity_kwh'].shift(168)
        
        # Rolling statistics (past 24 hours)
        feat['electricity_roll_mean_24h'] = feat['electricity_kwh'].rolling(24).mean()
        feat['electricity_roll_std_24h'] = feat['electricity_kwh'].rolling(24).std()
        feat['occupancy_roll_mean_24h'] = feat['occupancy'].rolling(24).mean()
        
        # Drop rows with NaN from lagging/rolling
        feat = feat.dropna().reset_index(drop=True)
        
        self.feature_names = [
            'hour', 'day_of_week', 'is_weekend', 'is_exam_period',
            'occupancy', 'temperature_outside',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'electricity_lag_1h', 'electricity_lag_24h', 'electricity_lag_168h',
            'electricity_roll_mean_24h', 'electricity_roll_std_24h',
            'occupancy_roll_mean_24h'
        ]
        
        return feat
    
    def train(self, test_size=0.2):
        """Train the Random Forest model."""
        if self.data is None:
            self.load_data()
        
        print("\nEngineering features...")
        feat = self._engineer_features(self.data)
        
        # Chronological split
        split_idx = int(len(feat) * (1 - test_size))
        train = feat.iloc[:split_idx]
        test = feat.iloc[split_idx:]
        
        X_train = train[self.feature_names]
        y_train = train['electricity_kwh']
        X_test = test[self.feature_names]
        y_test = test['electricity_kwh']
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        print("\nTraining Random Forest Regressor...")
        self.model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        
        # Evaluate
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
        """Return feature importance as a dict."""
        if self.model is None:
            raise ValueError("Model not trained.")
        importances = self.model.feature_importances_
        return {name: round(float(imp), 4) for name, imp in zip(self.feature_names, importances)}
    
    def predict_next(self):
        """Predict the next hour's electricity consumption."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        if self.data is None:
            self.load_data()
        
        feat = self._engineer_features(self.data)
        last_row = feat.iloc[[-1]][self.feature_names]
        prediction = float(self.model.predict(last_row)[0])
        
        # Estimate bounds using tree variance
        tree_preds = np.array([tree.predict(last_row)[0] for tree in self.model.estimators_])
        std = float(np.std(tree_preds))
        
        return {
            'prediction': round(prediction, 2),
            'lower_bound': round(prediction - 1.96 * std, 2),
            'upper_bound': round(prediction + 1.96 * std, 2),
            'confidence_level': 95
        }
    
    def get_exam_comparison(self):
        """Return aggregate stats comparing exam vs normal periods."""
        if self.data is None:
            self.load_data()
        
        exam = self.data[self.data['is_exam_period'] == 1]
        normal = self.data[self.data['is_exam_period'] == 0]
        
        def stats(df):
            return {
                'avg_electricity': round(float(df['electricity_kwh'].mean()), 2),
                'max_electricity': round(float(df['electricity_kwh'].max()), 2),
                'avg_occupancy': round(float(df['occupancy'].mean()), 1),
                'max_occupancy': int(df['occupancy'].max()),
                'total_records': len(df)
            }
        
        return {
            'exam': stats(exam),
            'normal': stats(normal)
        }
    
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
    print("Library Energy Predictor — Random Forest")
    print("=" * 60)
    
    predictor = LibraryEnergyPredictor()
    predictor.load_data()
    predictor.train()
    
    print("\n" + "=" * 60)
    print("Feature Importance:")
    print("=" * 60)
    for feat, imp in sorted(predictor.get_feature_importance().items(), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        print(f"  {feat:30s} {imp:.4f} {bar}")
    
    print("\n" + "=" * 60)
    print("Exam vs Normal Comparison:")
    print("=" * 60)
    comparison = predictor.get_exam_comparison()
    print(f"  Exam avg electricity:   {comparison['exam']['avg_electricity']} kWh")
    print(f"  Normal avg electricity: {comparison['normal']['avg_electricity']} kWh")
    
    print("\n" + "=" * 60)
    print("Next-Hour Prediction:")
    print("=" * 60)
    pred = predictor.predict_next()
    print(f"  Predicted: {pred['prediction']} kWh")
    print(f"  95% CI: [{pred['lower_bound']}, {pred['upper_bound']}] kWh")
    
    predictor.save_model()
    print("\n✓ All done!")
