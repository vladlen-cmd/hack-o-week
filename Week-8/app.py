from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, numpy as np, pickle, os

app = Flask(__name__, static_folder='static'); CORS(app)
data, bundle = None, None

def load():
    global data, bundle
    if os.path.exists('parking_data.csv'):
        data = pd.read_csv('parking_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.weekday
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f: bundle = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/historical')
def historical():
    if data is None or bundle is None: return jsonify({'success': False}), 400
    d = data.copy(); m = bundle['model']; th = bundle['threshold']
    feats = bundle['features']
    preds = m.predict(d[feats].values)
    d['predicted'] = np.round(preds, 2)
    d['residual'] = np.abs(d['electricity_kwh'] - d['predicted'])
    d['is_anomaly'] = (d['residual'] > th).astype(int)
    # Rename for frontend compatibility
    d = d.rename(columns={'electricity_kwh': 'lighting_kwh', 'vehicles': 'vehicle_count'})
    d['timestamp'] = d['timestamp'].astype(str)
    return jsonify({'success': True, 'data': d.to_dict(orient='records'),
                    'anomaly_count': int(d['is_anomaly'].sum())})

@app.route('/api/stats')
def stats():
    if bundle is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'metrics': bundle['metrics'],
        'summary': {'records': len(data), 'avg_lighting': round(float(data['electricity_kwh'].mean()), 2),
                    'avg_vehicles': int(data['vehicles'].mean())}})

if __name__ == '__main__':
    print("=" * 50); print("Parking Lot Lighting Forecast — Dashboard"); print("=" * 50)
    load(); print("-> http://localhost:5007"); app.run(port=5007, debug=False)
