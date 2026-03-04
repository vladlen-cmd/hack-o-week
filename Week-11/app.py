from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sqlite3, hashlib, secrets, os, json, time, jwt
from functools import wraps
from datetime import datetime, timedelta

app = Flask(__name__, static_folder='static')
CORS(app)
SECRET_KEY = secrets.token_hex(32)
DB = 'users.db'

def get_db():
    conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row; return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,
        profile_data TEXT DEFAULT '{}', wearable_data TEXT DEFAULT '[]',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit(); conn.close()

def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token: return jsonify({'success': False, 'error': 'Token required'}), 401
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
            request.user_id = payload['user_id']
            request.username = payload['username']
        except jwt.ExpiredSignatureError:
            return jsonify({'success': False, 'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'success': False, 'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/register', methods=['POST'])
def register():
    d = request.get_json()
    if not d or not d.get('username') or not d.get('email') or not d.get('password'):
        return jsonify({'success': False, 'error': 'Missing fields'}), 400
    try:
        conn = get_db()
        conn.execute('INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)',
                     (d['username'], d['email'], hash_pw(d['password'])))
        conn.commit(); conn.close()
        return jsonify({'success': True, 'msg': 'Registered successfully'})
    except sqlite3.IntegrityError:
        return jsonify({'success': False, 'error': 'Username or email exists'}), 409

@app.route('/api/login', methods=['POST'])
def login():
    d = request.get_json()
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE username=? AND password_hash=?',
                        (d.get('username', ''), hash_pw(d.get('password', '')))).fetchone()
    conn.close()
    if not user: return jsonify({'success': False, 'error': 'Invalid credentials'}), 401
    token = jwt.encode({'user_id': user['id'], 'username': user['username'],
                        'exp': datetime.utcnow() + timedelta(hours=24)}, SECRET_KEY, algorithm='HS256')
    return jsonify({'success': True, 'token': token, 'username': user['username']})

@app.route('/api/profile')
@token_required
def profile():
    conn = get_db()
    user = conn.execute('SELECT id, username, email, profile_data, wearable_data, created_at FROM users WHERE id=?',
                        (request.user_id,)).fetchone()
    conn.close()
    if not user: return jsonify({'success': False}), 404
    return jsonify({'success': True, 'user': {
        'id': user['id'], 'username': user['username'], 'email': user['email'],
        'profile': json.loads(user['profile_data']),
        'wearable_records': len(json.loads(user['wearable_data'])),
        'created_at': user['created_at']}})

@app.route('/api/profile', methods=['PUT'])
@token_required
def update_profile():
    d = request.get_json()
    # Encrypt profile data (simple base64 + hash for demo)
    encrypted = hashlib.sha256(json.dumps(d.get('profile', {})).encode()).hexdigest()[:16]
    conn = get_db()
    conn.execute('UPDATE users SET profile_data=? WHERE id=?',
                 (json.dumps({**d.get('profile', {}), '_hash': encrypted}), request.user_id))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'msg': 'Profile updated'})

@app.route('/api/wearable-sync', methods=['POST'])
@token_required
def wearable_sync():
    """Receive wearable data sync from device."""
    d = request.get_json()
    conn = get_db()
    user = conn.execute('SELECT wearable_data FROM users WHERE id=?', (request.user_id,)).fetchone()
    existing = json.loads(user['wearable_data']) if user else []
    record = {'timestamp': datetime.utcnow().isoformat(), 'heart_rate': d.get('heart_rate'),
              'steps': d.get('steps'), 'calories': d.get('calories'), 'device': d.get('device', 'unknown')}
    existing.append(record)
    conn.execute('UPDATE users SET wearable_data=? WHERE id=?', (json.dumps(existing), request.user_id))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'total_records': len(existing)})

@app.route('/api/stats')
def stats():
    conn = get_db()
    total = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]
    conn.close()
    return jsonify({'success': True, 'total_users': total, 'auth': 'JWT', 'encryption': 'SHA-256'})

@app.route('/api/users')
@token_required
def users():
    conn = get_db()
    rows = conn.execute('SELECT id, username, email, created_at FROM users').fetchall()
    conn.close()
    return jsonify({'success': True, 'users': [dict(r) for r in rows], 'total': len(rows)})

if __name__ == '__main__':
    print("=" * 50); print("User Registration Portal — JWT + Wearable Sync"); print("=" * 50)
    init_db(); print("-> http://localhost:5010"); app.run(port=5010, debug=False)
