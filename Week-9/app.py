from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, numpy as np, pickle, os

app = Flask(__name__, static_folder='static'); CORS(app)
data, bundle = None, None

def load():
    global data, bundle
    if os.path.exists('laundry_data.csv'):
        data = pd.read_csv('laundry_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.weekday
        # Rename for frontend consistency
        data.rename(columns={'electricity_kwh': 'load_kwh', 'machines_active': 'active_washers'}, inplace=True)
        data['active_dryers'] = (data['active_washers'] * 0.7).astype(int)
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f: bundle = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/historical')
def historical():
    if data is None: return jsonify({'success': False}), 400
    start = request.args.get('start')
    end = request.args.get('end')
    d = data.copy()
    if start: d = d[d['timestamp'] >= start]
    if end: d = d[d['timestamp'] <= end]
    d['timestamp'] = d['timestamp'].astype(str)
    return jsonify({'success': True, 'data': d.to_dict(orient='records'), 'total': len(d)})

@app.route('/api/stats')
def stats():
    if bundle is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'nb_accuracy': bundle['nb_accuracy'], 'engine': bundle['engine'],
        'thresholds': bundle['category_thresholds'],
        'summary': {'records': len(data), 'avg_load': round(float(data['load_kwh'].mean()), 2),
                    'sunday_avg': round(float(data[data['day_of_week'] == 6]['load_kwh'].mean()), 2)}})

@app.route('/api/forecast')
def forecast():
    if bundle is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'forecast': bundle['forecast'], 'engine': bundle['engine']})

if __name__ == '__main__':
    print("=" * 50); print("Hostel Laundry Peak Prediction — Dashboard"); print("=" * 50)
    load(); print("-> http://localhost:5008"); app.run(port=5008, debug=False)
