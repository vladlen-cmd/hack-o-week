from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd, numpy as np, pickle, os, threading, time

app = Flask(__name__, static_folder='static')
CORS(app); socketio = SocketIO(app, cors_allowed_origins="*")
data, model_bundle = None, None

def load():
    global data, model_bundle
    if os.path.exists('cafeteria_data.csv'):
        data = pd.read_csv('cafeteria_data.csv'); data['timestamp'] = pd.to_datetime(data['timestamp'])
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
    if data is None or model_bundle is None: return jsonify({'success': False}), 400
    lunch = data[(data['hour'].between(11, 14)) & (data['is_weekend'] == 0)]
    return jsonify({'success': True, 'metrics': model_bundle['metrics'],
        'summary': {'total_records': len(data), 'avg_energy': round(float(data['energy_kwh'].mean()), 2),
            'lunch_surge_avg': round(float(lunch['energy_kwh'].mean()), 2),
            'avg_footfall': int(data['footfall'].mean()),
            'surge_delta': round(float(lunch['energy_kwh'].mean() - data['energy_kwh'].mean()), 2)}})

def stream_data():
    idx = 0
    while True:
        if data is not None and model_bundle is not None:
            row = data.iloc[idx % len(data)]
            X = row[model_bundle['features']].values.reshape(1, -1)
            pred = float(model_bundle['model'].predict(X)[0])
            socketio.emit('live_data', {'timestamp': str(row['timestamp']), 'actual': float(row['energy_kwh']),
                'predicted': round(pred, 2), 'hour': int(row['hour']), 'footfall': int(row['footfall']),
                'temperature': float(row['temperature_c']),
                'is_lunch': int(11 <= row['hour'] <= 14 and row['is_weekend'] == 0)})
            idx += 1
        time.sleep(2)

@socketio.on('connect')
def on_connect(): emit('status', {'msg': 'Connected'})

if __name__ == '__main__':
    print("=" * 50); print("Cafeteria Load Prediction — WebSocket Dashboard"); print("=" * 50)
    load(); t = threading.Thread(target=stream_data, daemon=True); t.start()
    print("-> http://localhost:5003"); socketio.run(app, port=5003, debug=False, allow_unsafe_werkzeug=True)
