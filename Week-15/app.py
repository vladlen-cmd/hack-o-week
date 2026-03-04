from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, numpy as np, pickle, os

app = Flask(__name__, static_folder='static'); CORS(app)
data, bundle = None, None

def load():
    global data, bundle
    if os.path.exists('heart_data_scored.csv'):
        data = pd.read_csv('heart_data_scored.csv')
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f: bundle = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/stats')
def stats():
    if bundle is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'metrics': bundle['metrics'],
        'summary': {'total_points': len(data), 'avg_hr': round(float(data['heart_rate'].mean()), 1)}})

@app.route('/api/data')
def get_data():
    if data is None: return jsonify({'success': False}), 400
    d = data.copy(); d['timestamp'] = d['timestamp'].astype(str)
    return jsonify({'success': True, 'data': d.to_dict(orient='records')})

@app.route('/api/check', methods=['POST'])
def check():
    """REST API: flag irregularities in heart rate data."""
    if bundle is None: return jsonify({'success': False}), 400
    d = request.get_json()
    hr = d.get('heart_rate', 72); rhr = d.get('resting_hr', 60); hrv = d.get('hr_variability', 40)
    X = np.array([[hr, rhr, hrv]])
    pred = bundle['model'].predict(X)[0]
    score = float(bundle['model'].decision_function(X)[0])
    return jsonify({'success': True, 'is_anomaly': int(pred == -1), 'anomaly_score': round(score, 4),
                    'heart_rate': hr, 'status': 'IRREGULAR' if pred == -1 else 'NORMAL'})

if __name__ == '__main__':
    print("=" * 50); print("Basic Anomaly Detection — Heart Rate (Isolation Forest)"); print("=" * 50)
    load(); print("-> http://localhost:5014"); app.run(port=5014, debug=False)
