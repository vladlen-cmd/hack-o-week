from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, pickle, os

app = Flask(__name__, static_folder='static'); CORS(app)
data, meta, day_analysis = None, None, None

def load():
    global data, meta, day_analysis
    if os.path.exists('sports_data.csv'):
        data = pd.read_csv('sports_data.csv'); data['timestamp'] = pd.to_datetime(data['timestamp'])
    if os.path.exists('model_meta.pkl'):
        with open('model_meta.pkl', 'rb') as f: meta = pickle.load(f)
    if os.path.exists('day_analysis.pkl'):
        with open('day_analysis.pkl', 'rb') as f: day_analysis = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/historical')
def historical():
    if data is None: return jsonify({'success': False}), 400
    day_type = request.args.get('day_type')
    d = data if day_type is None else data[data['day_type'] == day_type]
    d = d.copy(); d['timestamp'] = d['timestamp'].astype(str)
    return jsonify({'success': True, 'data': d.to_dict(orient='records'), 'total': len(d)})

@app.route('/api/stats')
def stats():
    if meta is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'metrics': meta['metrics'], 'day_analysis': day_analysis,
        'summary': {'records': len(data), 'avg_energy': round(float(data['energy_kwh'].mean()), 2)}})

if __name__ == '__main__':
    print("=" * 50); print("Sports Facility Night Usage — LSTM Dashboard"); print("=" * 50)
    load(); print("-> http://localhost:5005"); app.run(port=5005, debug=False)
