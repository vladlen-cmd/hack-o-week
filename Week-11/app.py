from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
import sqlite3, hashlib, secrets, os, re
from datetime import datetime

app = Flask(__name__, static_folder='static')
app.secret_key = secrets.token_hex(32)
CORS(app, supports_credentials=True)
DB = 'users.db'

def get_db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT DEFAULT '',
        department TEXT DEFAULT '',
        role TEXT DEFAULT 'student',
        avatar_color TEXT DEFAULT '#8b5cf6',
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        last_login TEXT
    )''')
    conn.commit(); conn.close()

def hash_password(password, salt=None):
    if salt is None: salt = secrets.token_hex(16)
    h = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{h}"

def verify_password(password, stored):
    salt, hashed = stored.split(':')
    return hash_password(password, salt) == stored

def validate_email(email):
    return bool(re.match(r'^[\w.-]+@[\w.-]+\.\w+$', email))

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        username = data.get('username', '').strip()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        department = data.get('department', '').strip()

        if not username or not email or not password:
            return jsonify({'success': False, 'error': 'Username, email, and password required'}), 400
        if len(username) < 3:
            return jsonify({'success': False, 'error': 'Username must be at least 3 characters'}), 400
        if len(password) < 6:
            return jsonify({'success': False, 'error': 'Password must be at least 6 characters'}), 400
        if not validate_email(email):
            return jsonify({'success': False, 'error': 'Invalid email format'}), 400

        colors = ['#8b5cf6','#ec4899','#06b6d4','#f59e0b','#10b981','#ef4444','#3b82f6','#6366f1']
        avatar_color = colors[hash(username) % len(colors)]
        conn = get_db()
        try:
            conn.execute('INSERT INTO users (username, email, password_hash, full_name, department, avatar_color) VALUES (?,?,?,?,?,?)',
                         (username, email, hash_password(password), full_name, department, avatar_color))
            conn.commit()
        except sqlite3.IntegrityError as e:
            return jsonify({'success': False, 'error': 'Username or email already exists'}), 409
        finally:
            conn.close()

        return jsonify({'success': True, 'message': 'Registration successful'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        identifier = data.get('identifier', '').strip()
        password = data.get('password', '')
        if not identifier or not password:
            return jsonify({'success': False, 'error': 'Credentials required'}), 400

        conn = get_db()
        user = conn.execute('SELECT * FROM users WHERE username=? OR email=?', (identifier, identifier.lower())).fetchone()
        if not user or not verify_password(password, user['password_hash']):
            conn.close()
            return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

        conn.execute('UPDATE users SET last_login=? WHERE id=?', (datetime.now().isoformat(), user['id']))
        conn.commit(); conn.close()

        session['user_id'] = user['id']
        return jsonify({'success': True, 'user': {
            'id': user['id'], 'username': user['username'], 'email': user['email'],
            'full_name': user['full_name'], 'department': user['department'],
            'role': user['role'], 'avatar_color': user['avatar_color'],
            'created_at': user['created_at']
        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/profile', methods=['GET'])
def get_profile():
    uid = session.get('user_id')
    if not uid: return jsonify({'success': False, 'error': 'Not logged in'}), 401
    conn = get_db()
    user = conn.execute('SELECT * FROM users WHERE id=?', (uid,)).fetchone()
    conn.close()
    if not user: return jsonify({'success': False, 'error': 'User not found'}), 404
    return jsonify({'success': True, 'user': {
        'id': user['id'], 'username': user['username'], 'email': user['email'],
        'full_name': user['full_name'], 'department': user['department'],
        'role': user['role'], 'avatar_color': user['avatar_color'],
        'created_at': user['created_at'], 'last_login': user['last_login']
    }})

@app.route('/api/profile', methods=['PUT'])
def update_profile():
    uid = session.get('user_id')
    if not uid: return jsonify({'success': False, 'error': 'Not logged in'}), 401
    data = request.get_json()
    conn = get_db()
    conn.execute('UPDATE users SET full_name=?, department=? WHERE id=?',
                 (data.get('full_name', ''), data.get('department', ''), uid))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'message': 'Profile updated'})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/api/users')
def list_users():
    conn = get_db()
    users = conn.execute('SELECT id, username, full_name, department, role, avatar_color, created_at FROM users ORDER BY created_at DESC').fetchall()
    conn.close()
    return jsonify({'success': True, 'users': [dict(u) for u in users], 'total': len(users)})

if __name__ == '__main__':
    print("=" * 60); print("User Registration Portal"); print("=" * 60)
    init_db(); print("✓ Database initialized")
    print("✓ http://localhost:5010")
    app.run(debug=False, port=5010)
