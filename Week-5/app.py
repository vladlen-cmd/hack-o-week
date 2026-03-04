from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, numpy as np, pickle, os

app = Flask(__name__, static_folder='static'); CORS(app)
data, model_bundle = None, None

def load():
    global data, model_bundle
    if os.path.exists('hvac_data.csv'):
        data = pd.read_csv('hvac_data.csv'); data['timestamp'] = pd.to_datetime(data['timestamp'])
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f: model_bundle = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/historical')
def historical():
    if data is None: return jsonify({'success': False}), 400
    d = data.copy(); d['timestamp'] = d['timestamp'].astype(str)
    return jsonify({'success': True, 'data': d.to_dict(orient='records')})

@app.route('/api/stats')
def stats():
    if model_bundle is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'metrics': model_bundle['metrics'],
        'summary': {'records': len(data), 'avg_cooling': round(float(data['cooling_kwh'].mean()), 2),
            'zones': data['zone_name'].nunique()}})

@app.route('/api/heatmap')
def heatmap():
    """Zone × Hour heatmap data."""
    if data is None: return jsonify({'success': False}), 400
    zones = sorted(data['zone_name'].unique())
    matrix = []
    for z in zones:
        row = []
        for h in range(24):
            subset = data[(data['zone_name'] == z) & (data['hour'] == h)]
            row.append(round(float(subset['cooling_kwh'].mean()), 2) if len(subset) > 0 else 0)
        matrix.append(row)
    return jsonify({'success': True, 'zones': zones, 'hours': list(range(24)), 'matrix': matrix})

if __name__ == '__main__':
    print("=" * 50); print("HVAC Optimization in Labs — Dashboard"); print("=" * 50)
    load(); print("-> http://localhost:5004"); app.run(port=5004, debug=False)
