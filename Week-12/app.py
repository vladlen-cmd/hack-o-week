from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from cryptography.fernet import Fernet
import pandas as pd, numpy as np, json, os, sqlite3
from datetime import datetime

app = Flask(__name__, static_folder='static')
CORS(app); socketio = SocketIO(app, cors_allowed_origins="*")
FERNET_KEY = Fernet.generate_key()
fernet = Fernet(FERNET_KEY)
DB = 'wearable.db'

def init_db():
    conn = sqlite3.connect(DB); conn.execute('''CREATE TABLE IF NOT EXISTS readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, device TEXT,
        encrypted_payload TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit(); conn.close()

def encrypt_payload(data):
    """Encrypt JSON payload before database insert."""
    return fernet.encrypt(json.dumps(data).encode()).decode()

def decrypt_payload(ciphertext):
    return json.loads(fernet.decrypt(ciphertext.encode()).decode())

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/stats')
def stats():
    conn = sqlite3.connect(DB)
    total = conn.execute('SELECT COUNT(*) FROM readings').fetchone()[0]
    users = conn.execute('SELECT COUNT(DISTINCT user_id) FROM readings').fetchone()[0]
    last = conn.execute('SELECT timestamp FROM readings ORDER BY id DESC LIMIT 1').fetchone()
    conn.close()
    return jsonify({'success': True, 'total_records': total, 'unique_users': users,
                    'last_reading': last[0] if last else None, 'encryption': 'Fernet (AES-128-CBC)'})

@app.route('/api/readings')
def readings():
    conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM readings ORDER BY id DESC LIMIT 100').fetchall()
    conn.close()
    result = []
    for r in rows:
        try:
            decrypted = decrypt_payload(r['encrypted_payload'])
            result.append({**decrypted, 'id': r['id'], 'timestamp': r['timestamp'], 'user_id': r['user_id'], 'device': r['device']})
        except: pass
    return jsonify({'success': True, 'data': result})

@app.route('/api/ingest', methods=['POST'])
def rest_ingest():
    """REST fallback for non-WebSocket clients."""
    d = request.get_json()
    encrypted = encrypt_payload({'heart_rate': d.get('heart_rate'), 'steps': d.get('steps')})
    conn = sqlite3.connect(DB)
    conn.execute('INSERT INTO readings (user_id, device, encrypted_payload) VALUES (?, ?, ?)',
                 (d.get('user_id', 'web'), d.get('device', 'REST'), encrypted))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'encrypted': True})

@socketio.on('wearable_data')
def handle_data(data):
    """WebSocket handler: encrypt and store incoming wearable data."""
    payload = {'heart_rate': data.get('heart_rate'), 'steps': data.get('steps'),
               'spo2': data.get('spo2'), 'calories': data.get('calories')}
    encrypted = encrypt_payload(payload)
    conn = sqlite3.connect(DB)
    conn.execute('INSERT INTO readings (user_id, device, encrypted_payload) VALUES (?, ?, ?)',
                 (data.get('user_id', 'ws_user'), data.get('device', 'WebSocket'), encrypted))
    conn.commit(); conn.close()
    emit('ack', {'success': True, 'encrypted_size': len(encrypted), 'timestamp': datetime.utcnow().isoformat()})
    socketio.emit('new_reading', {**payload, 'user_id': data.get('user_id', 'ws_user'),
                                   'timestamp': datetime.utcnow().isoformat()}, broadcast=True)

@socketio.on('connect')
def on_connect(): emit('status', {'msg': 'Connected', 'encryption': 'Fernet'})

if __name__ == '__main__':
    print("=" * 50); print("Wearable Data Ingestion — WebSocket + Encryption"); print("=" * 50)
    init_db(); print("-> http://localhost:5011")
    socketio.run(app, port=5011, debug=False, allow_unsafe_werkzeug=True)
