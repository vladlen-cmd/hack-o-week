from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, numpy as np, json, os
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app)
DATA_FILE = 'wearable_data.csv'
data = None

def load_data():
    global data
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        print(f"✓ Loaded {len(data)} records")
    else:
        print("⚠ Run data_simulator.py first")

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/ingest', methods=['POST'])
def ingest():
    """Accept new wearable data points via POST."""
    try:
        global data
        payload = request.get_json()
        if isinstance(payload, list):
            new_df = pd.DataFrame(payload)
        else:
            new_df = pd.DataFrame([payload])
        new_df['timestamp'] = pd.to_datetime(new_df['timestamp'])
        if data is not None:
            data = pd.concat([data, new_df], ignore_index=True)
        else:
            data = new_df
        return jsonify({'success': True, 'message': f'Ingested {len(new_df)} records', 'total': len(data)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/data')
def get_data():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        user = request.args.get('user_id')
        d = data if user is None else data[data['user_id'] == user]
        d_out = d.copy()
        d_out['timestamp'] = d_out['timestamp'].astype(str)
        limit = int(request.args.get('limit', 2000))
        return jsonify({'success': True, 'data': d_out.tail(limit).to_dict(orient='records'), 'total': len(d)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/users')
def get_users():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        users = []
        for uid, grp in data.groupby('user_id'):
            users.append({
                'user_id': uid,
                'device_type': grp['device_type'].iloc[0],
                'records': len(grp),
                'avg_hr': round(float(grp['heart_rate_bpm'].mean()), 1),
                'total_steps': int(grp['steps'].sum()),
                'avg_spo2': round(float(grp['spo2_pct'].mean()), 1),
                'avg_calories': round(float(grp['calories_kcal'].mean()), 1),
                'last_seen': str(grp['timestamp'].max())
            })
        return jsonify({'success': True, 'users': users})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        return jsonify({'success': True, 'stats': {
            'total_records': len(data),
            'total_users': data['user_id'].nunique(),
            'total_devices': data['device_type'].nunique(),
            'date_range': f"{data['timestamp'].min().date()} to {data['timestamp'].max().date()}",
            'avg_heart_rate': round(float(data['heart_rate_bpm'].mean()), 1),
            'avg_spo2': round(float(data['spo2_pct'].mean()), 1),
            'total_steps': int(data['steps'].sum()),
            'avg_daily_steps_per_user': int(data.groupby([data['timestamp'].dt.date, 'user_id'])['steps'].sum().mean()),
            'avg_calories': round(float(data['calories_kcal'].mean()), 1),
            'avg_skin_temp': round(float(data['skin_temp_c'].mean()), 1),
            'ingestion_rate': f"{len(data)} records ingested"
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/hourly')
def hourly():
    try:
        if data is None: return jsonify({'success': False, 'error': 'No data'}), 400
        h = data.copy()
        h['hour'] = h['timestamp'].dt.hour
        agg = h.groupby('hour').agg({'heart_rate_bpm': 'mean', 'steps': 'mean', 'spo2_pct': 'mean', 'calories_kcal': 'mean'}).reset_index()
        return jsonify({'success': True, 'hourly': agg.round(2).to_dict(orient='records')})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60); print("Wearable Data Ingestion — Dashboard"); print("=" * 60)
    load_data()
    print("✓ http://localhost:5011")
    app.run(debug=False, port=5011)
