import pandas as pd, numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle, warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    def __init__(self, data_file='sensor_data.csv', model_file='detector.pkl'):
        self.data_file = data_file; self.model_file = model_file
        self.data = None; self.model = None; self.scaler = None
        self.feature_names = None; self.metrics = None

    def load_data(self):
        self.data = pd.read_csv(self.data_file)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        return self.data

    def _features(self, df):
        f = df.copy()
        f['hour'] = f['timestamp'].dt.hour
        f['dow'] = f['timestamp'].dt.dayofweek
        f['power_roll'] = f['power_kwh'].rolling(6, min_periods=1).mean()
        f['temp_roll'] = f['server_temp_c'].rolling(6, min_periods=1).mean()
        f['net_roll'] = f['network_mbps'].rolling(6, min_periods=1).mean()
        f['power_diff'] = f['power_kwh'].diff().fillna(0)
        f['temp_diff'] = f['server_temp_c'].diff().fillna(0)
        f['net_diff'] = f['network_mbps'].diff().fillna(0)
        self.feature_names = ['power_kwh', 'server_temp_c', 'network_mbps', 'hour', 'dow',
                              'power_roll', 'temp_roll', 'net_roll', 'power_diff', 'temp_diff', 'net_diff']
        return f

    def train(self, contamination=0.03):
        if self.data is None: self.load_data()
        f = self._features(self.data)
        X = f[self.feature_names].values
        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)
        self.model = IsolationForest(n_estimators=200, contamination=contamination,
                                     max_samples='auto', random_state=42, n_jobs=-1)
        self.model.fit(X_s)
        # Predict on training data
        preds = self.model.predict(X_s)
        scores = self.model.decision_function(X_s)
        self.data['predicted_anomaly'] = (preds == -1).astype(int)
        self.data['anomaly_score'] = scores

        # Z-Score method for comparison
        for col in ['power_kwh', 'server_temp_c', 'network_mbps']:
            mu, sigma = self.data[col].mean(), self.data[col].std()
            self.data[f'{col}_zscore'] = ((self.data[col] - mu) / sigma).abs()
        self.data['zscore_anomaly'] = ((self.data['power_kwh_zscore'] > 3) |
                                       (self.data['server_temp_c_zscore'] > 3) |
                                       (self.data['network_mbps_zscore'] > 3)).astype(int)

        # Metrics
        actual = self.data['is_anomaly'].values
        predicted = self.data['predicted_anomaly'].values
        tp = int(((predicted == 1) & (actual == 1)).sum())
        fp = int(((predicted == 1) & (actual == 0)).sum())
        fn = int(((predicted == 0) & (actual == 1)).sum())
        tn = int(((predicted == 0) & (actual == 0)).sum())
        precision = round(tp / max(tp + fp, 1), 3)
        recall = round(tp / max(tp + fn, 1), 3)
        f1 = round(2 * precision * recall / max(precision + recall, 0.001), 3)
        self.metrics = {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
                        'precision': precision, 'recall': recall, 'f1_score': f1,
                        'total_anomalies_detected': int(predicted.sum()),
                        'actual_anomalies': int(actual.sum())}
        print(f"✓ IsolationForest — Precision={precision}, Recall={recall}, F1={f1}")
        return self.model

    def get_anomalies(self, method='isolation_forest'):
        col = 'predicted_anomaly' if method == 'isolation_forest' else 'zscore_anomaly'
        anoms = self.data[self.data[col] == 1].copy()
        anoms['timestamp'] = anoms['timestamp'].astype(str)
        return anoms[['timestamp', 'power_kwh', 'server_temp_c', 'network_mbps',
                       'anomaly_score', 'is_anomaly', 'anomaly_type']].to_dict(orient='records')

    def get_summary(self):
        return {
            'total_points': len(self.data),
            'isolation_forest_count': int(self.data['predicted_anomaly'].sum()),
            'zscore_count': int(self.data['zscore_anomaly'].sum()),
            'actual_count': int(self.data['is_anomaly'].sum()),
            'avg_power': round(float(self.data['power_kwh'].mean()), 2),
            'avg_temp': round(float(self.data['server_temp_c'].mean()), 2),
            'avg_network': round(float(self.data['network_mbps'].mean()), 1),
        }

    def save_model(self):
        with open(self.model_file, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler,
                         'features': self.feature_names, 'metrics': self.metrics}, f)

    def load_model(self):
        with open(self.model_file, 'rb') as f:
            b = pickle.load(f)
        self.model = b['model']; self.scaler = b['scaler']
        self.feature_names = b['features']; self.metrics = b['metrics']

if __name__ == '__main__':
    d = AnomalyDetector(); d.load_data(); d.train()
    print("Summary:", d.get_summary()); d.save_model()
