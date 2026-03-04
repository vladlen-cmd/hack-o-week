from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import pandas as pd, numpy as np, pickle, os, threading, time

app = Flask(__name__, static_folder='static')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

data, model_bundle = None, None

def load():
    global data, model_bundle
    if os.path.exists('library_data.csv'):
        data = pd.read_csv('library_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model_bundle = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/historical')
def historical():
    if data is None: return jsonify({'success': False}), 400
    d = data.copy(); d['timestamp'] = d['timestamp'].astype(str)
    return jsonify({'success': True, 'data': d.to_dict(orient='records')})

@app.route('/api/predict')
def predict():
    if model_bundle is None: return jsonify({'success': False}), 400
    m = model_bundle['model']
    hour = int(pd.Timestamp.now().hour)
    sample = np.array([[hour, pd.Timestamp.now().weekday(), int(pd.Timestamp.now().weekday() >= 5),
                        0, 30.0, 0, 120]])
    pred = float(m.predict(sample)[0])
    return jsonify({'success': True, 'predicted_kwh': round(pred, 2), 'hour': hour,
                    'coefficients': model_bundle['metrics']['coefficients']})

@app.route('/api/stats')
def stats():
    if data is None or model_bundle is None: return jsonify({'success': False}), 400
    lunch = data[(data['hour'].between(12, 14)) & (data['is_weekend'] == 0)]
    non_lunch = data[(~data['hour'].between(12, 14)) & (data['is_weekend'] == 0)]
    return jsonify({'success': True, 'metrics': model_bundle['metrics'],
        'summary': {'total_records': len(data), 'avg_energy': round(float(data['energy_kwh'].mean()), 2),
            'lunch_surge_avg': round(float(lunch['energy_kwh'].mean()), 2),
            'non_lunch_avg': round(float(non_lunch['energy_kwh'].mean()), 2),
            'surge_delta': round(float(lunch['energy_kwh'].mean() - non_lunch['energy_kwh'].mean()), 2)}})

def stream_data():
    """Background thread: emit real-time data points via WebSocket every 2 seconds."""
    idx = 0
    while True:
        if data is not None and model_bundle is not None:
            row = data.iloc[idx % len(data)]
            m = model_bundle['model']
            features = model_bundle['features']
            X = row[features].values.reshape(1, -1)
            pred = float(m.predict(X)[0])
            socketio.emit('live_data', {
                'timestamp': str(row['timestamp']), 'actual': float(row['energy_kwh']),
                'predicted': round(pred, 2), 'hour': int(row['hour']),
                'temperature': float(row['temperature_c']), 'occupancy': int(row['occupancy']),
                'is_lunch': int(12 <= row['hour'] <= 14 and row['is_weekend'] == 0)
            })
            idx += 1
        time.sleep(2)

@socketio.on('connect')
def on_connect():
    emit('status', {'msg': 'Connected to live stream'})

if __name__ == '__main__':
    print("=" * 50); print("Library Energy During Exams — WebSocket Dashboard"); print("=" * 50)
    load()
    t = threading.Thread(target=stream_data, daemon=True); t.start()
    print("-> http://localhost:5002")
    socketio.run(app, port=5002, debug=False, allow_unsafe_werkzeug=True)
