from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, pickle, os

app = Flask(__name__, static_folder='static'); CORS(app)
data, bundle = None, None

def load():
    global data, bundle
    if os.path.exists('campus_data.csv'):
        data = pd.read_csv('campus_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.weekday
        # Rename for frontend compatibility
        data.rename(columns={
            'total_electricity_kwh': 'energy_kwh',
            'solar_generation_kwh': 'solar_kwh',
            'water_consumption_l': 'water_liters',
            'temperature': 'outdoor_temp_c'
        }, inplace=True)
        # No "building" column in campus data — use a placeholder
        if 'building' not in data.columns:
            data['building'] = 'Campus-Wide'
        if 'occupancy' not in data.columns:
            data['occupancy'] = 0
    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f: bundle = pickle.load(f)

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/sustainability')
def sustainability():
    if data is None or bundle is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'metrics': bundle['metrics'], 'carbon': bundle['carbon'],
        'summary': {'records': len(data), 'buildings': int(data['building'].nunique()),
                    'avg_energy': round(float(data['energy_kwh'].mean()), 2)}})

@app.route('/api/drilldown')
def drilldown():
    if data is None: return jsonify({'success': False}), 400
    building = request.args.get('building')
    sub = data[data['building'] == building].copy() if building else data.copy()
    sub['timestamp'] = sub['timestamp'].astype(str)
    buildings = sorted(data['building'].unique().tolist())
    return jsonify({'success': True, 'data': sub.to_dict(orient='records'), 'buildings': buildings})

if __name__ == '__main__':
    print("=" * 50); print("Campus-Wide Sustainability Tracker — Dashboard"); print("=" * 50)
    load(); print("-> http://localhost:5009"); app.run(port=5009, debug=False)
