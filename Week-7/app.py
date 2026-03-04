from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, pickle, os

app = Flask(__name__, static_folder='static'); CORS(app)
data, bundle = None, None

def load():
    global data, bundle
    if os.path.exists('admin_data_clustered.csv'):
        data = pd.read_csv('admin_data_clustered.csv'); data['timestamp'] = pd.to_datetime(data['timestamp'])
    elif os.path.exists('admin_data.csv'):
        data = pd.read_csv('admin_data.csv'); data['timestamp'] = pd.to_datetime(data['timestamp'])
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f: bundle = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/historical')
def historical():
    if data is None: return jsonify({'success': False}), 400
    d = data.copy(); d['timestamp'] = d['timestamp'].astype(str)
    return jsonify({'success': True, 'data': d.to_dict(orient='records')})

@app.route('/api/stats')
def stats():
    if bundle is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'metrics': bundle['metrics'], 'savings': bundle['savings'],
                    'potential_kwh': bundle['potential_kwh'], 'centers': bundle['centers']})

if __name__ == '__main__':
    print("=" * 50); print("Admin Building Weekend Dip — KMeans Dashboard"); print("=" * 50)
    load(); print("-> http://localhost:5006"); app.run(port=5006, debug=False)
