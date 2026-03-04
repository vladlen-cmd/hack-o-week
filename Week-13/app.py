from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, os

app = Flask(__name__, static_folder='static')
CORS(app)
data = None

def load():
    global data
    if os.path.exists('dashboard_data.csv'):
        data = pd.read_csv('dashboard_data.csv')
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        print(f"✓ Loaded {len(data)} records")

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/data')
def api_data():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        d = data.copy(); d['timestamp'] = d['timestamp'].astype(str)
        return jsonify({'success': True, 'data': d.to_dict(orient='records')})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/summary')
def api_summary():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        last7 = data[data['timestamp'] >= data['timestamp'].max() - pd.Timedelta(days=7)]
        return jsonify({'success': True, 'summary': {
            'total_records': len(data),
            'days_covered': int((data['timestamp'].max() - data['timestamp'].min()).days),
            'avg_energy': round(float(data['energy_kwh'].mean()), 1),
            'avg_bandwidth': round(float(data['bandwidth_mbps'].mean()), 0),
            'avg_footfall': int(data['footfall'].mean()),
            'avg_aqi': int(data['aqi'].mean()),
            'total_events': int((data['event_attendance'] > 0).sum()),
            'energy_trend': round(float(last7['energy_kwh'].mean() - data['energy_kwh'].mean()), 1),
            'aqi_trend': round(float(last7['aqi'].mean() - data['aqi'].mean()), 1),
        }})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/hourly')
def api_hourly():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        h = data.copy(); h['hour'] = h['timestamp'].dt.hour
        agg = h.groupby('hour').agg({'energy_kwh': 'mean', 'bandwidth_mbps': 'mean', 'footfall': 'mean', 'aqi': 'mean'}).reset_index()
        return jsonify({'success': True, 'hourly': agg.round(2).to_dict(orient='records')})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/daily')
def api_daily():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        d = data.copy(); d['date'] = d['timestamp'].dt.date.astype(str)
        agg = d.groupby('date').agg({'energy_kwh': 'sum', 'bandwidth_mbps': 'mean', 'footfall': 'sum', 'aqi': 'mean', 'event_attendance': 'sum'}).reset_index()
        return jsonify({'success': True, 'daily': agg.round(2).to_dict(orient='records')})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60); print("Dashboard Visualization"); print("=" * 60)
    load(); print("✓ http://localhost:5012")
    app.run(debug=False, port=5012)
