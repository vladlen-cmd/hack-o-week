from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd, numpy as np, os, json
from cryptography.fernet import Fernet

app = Flask(__name__, static_folder='static'); CORS(app)
data = None; FERNET_KEY = Fernet.generate_key(); fernet = Fernet(FERNET_KEY)

def load():
    global data
    if os.path.exists('activity_data.csv'):
        data = pd.read_csv('activity_data.csv')

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/summary')
def summary():
    if data is None: return jsonify({'success': False}), 400
    return jsonify({'success': True, 'summary': {
        'days': int(data['date'].nunique()), 'avg_steps': int(data['steps'].mean()),
        'avg_hr': int(data['heart_rate'].mean()), 'avg_calories': int(data['calories'].mean())}})

@app.route('/api/timeseries')
def timeseries():
    """Return decrypted activity data (simulates fetch from encrypted store)."""
    if data is None: return jsonify({'success': False}), 400
    # Simulate: encrypt then decrypt to demonstrate the pipeline
    records = data.to_dict(orient='records')
    decrypted = []
    for r in records:
        cipher = fernet.encrypt(json.dumps(r).encode())
        plain = json.loads(fernet.decrypt(cipher).decode())
        decrypted.append(plain)
    return jsonify({'success': True, 'data': decrypted, 'encryption': 'Fernet (decrypted for display)'})

if __name__ == '__main__':
    print("=" * 50); print("Dashboard Visualization — React + Decrypted Data"); print("=" * 50)
    load(); print("-> http://localhost:5012"); app.run(port=5012, debug=False)
