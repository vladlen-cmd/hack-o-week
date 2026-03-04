from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os, numpy as np
from detector import AnomalyDetector

app = Flask(__name__, static_folder='static')
CORS(app)
det = AnomalyDetector()

def init():
    try:
        if os.path.exists('detector.pkl') and os.path.exists('sensor_data.csv'):
            det.load_data(); det.load_model()
            # Re-run predictions on loaded data
            from sklearn.preprocessing import StandardScaler
            f = det._features(det.data)
            X = det.scaler.transform(f[det.feature_names].values)
            preds = det.model.predict(X)
            scores = det.model.decision_function(X)
            det.data['predicted_anomaly'] = (preds == -1).astype(int)
            det.data['anomaly_score'] = scores
            for col in ['power_kwh', 'server_temp_c', 'network_mbps']:
                mu, sigma = det.data[col].mean(), det.data[col].std()
                det.data[f'{col}_zscore'] = ((det.data[col] - mu) / sigma).abs()
            det.data['zscore_anomaly'] = ((det.data['power_kwh_zscore'] > 3) |
                                          (det.data['server_temp_c_zscore'] > 3) |
                                          (det.data['network_mbps_zscore'] > 3)).astype(int)
    except Exception as e: print(f"⚠ {e}")

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/data')
def api_data():
    try:
        if det.data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        d = det.data.copy()
        d['timestamp'] = d['timestamp'].astype(str)
        cols = ['timestamp', 'power_kwh', 'server_temp_c', 'network_mbps',
                'is_anomaly', 'predicted_anomaly', 'anomaly_score', 'anomaly_type']
        return jsonify({'success': True, 'data': d[cols].to_dict(orient='records')})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/anomalies')
def anomalies():
    try:
        return jsonify({'success': True, 'anomalies': det.get_anomalies()})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    try:
        if det.metrics is None: return jsonify({'success': False, 'error': 'Not trained'}), 400
        return jsonify({'success': True, 'metrics': det.metrics, 'summary': det.get_summary()})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60); print("Basic Anomaly Detection — Dashboard"); print("=" * 60)
    init(); print("✓ http://localhost:5014")
    app.run(debug=False, port=5014)
