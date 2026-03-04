from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from cryptography.fernet import Fernet
import sqlite3, json, os
from datetime import datetime

app = Flask(__name__, static_folder='static'); CORS(app)
FERNET_KEY = Fernet.generate_key(); fernet = Fernet(FERNET_KEY)
DB = 'encrypted_store.db'

def init_db():
    conn = sqlite3.connect(DB)
    conn.execute('''CREATE TABLE IF NOT EXISTS ciphertexts (
        id INTEGER PRIMARY KEY AUTOINCREMENT, label TEXT,
        algorithm TEXT, ciphertext TEXT, original_size INTEGER,
        encrypted_size INTEGER, timestamp TEXT DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit(); conn.close()

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/encrypt', methods=['POST'])
def encrypt():
    d = request.get_json()
    text = d.get('text', ''); label = d.get('label', 'unlabeled')
    # Server-side Fernet encryption
    ct = fernet.encrypt(text.encode()).decode()
    # Store in DB
    conn = sqlite3.connect(DB)
    conn.execute('INSERT INTO ciphertexts (label, algorithm, ciphertext, original_size, encrypted_size) VALUES (?,?,?,?,?)',
                 (label, 'Fernet+CryptoJS', ct, len(text), len(ct)))
    conn.commit(); conn.close()
    return jsonify({'success': True, 'ciphertext': ct, 'original_size': len(text),
                    'encrypted_size': len(ct), 'algorithm': 'Fernet'})

@app.route('/api/decrypt', methods=['POST'])
def decrypt():
    d = request.get_json()
    try:
        plain = fernet.decrypt(d.get('ciphertext', '').encode()).decode()
        return jsonify({'success': True, 'plaintext': plain})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/store')
def store():
    """List stored ciphertexts from DB."""
    conn = sqlite3.connect(DB); conn.row_factory = sqlite3.Row
    rows = conn.execute('SELECT * FROM ciphertexts ORDER BY id DESC LIMIT 50').fetchall()
    conn.close()
    return jsonify({'success': True, 'records': [dict(r) for r in rows], 'total': len(rows)})

@app.route('/api/stats')
def stats():
    conn = sqlite3.connect(DB)
    total = conn.execute('SELECT COUNT(*) FROM ciphertexts').fetchone()[0]
    total_bytes = conn.execute('SELECT COALESCE(SUM(encrypted_size),0) FROM ciphertexts').fetchone()[0]
    conn.close()
    return jsonify({'success': True, 'total_records': total, 'total_encrypted_bytes': total_bytes,
                    'server_algorithm': 'Fernet (AES-128-CBC)', 'client_algorithm': 'CryptoJS AES'})

if __name__ == '__main__':
    print("=" * 50); print("Data Encryption Pipeline — CryptoJS + Fernet + DB"); print("=" * 50)
    init_db(); print("-> http://localhost:5013"); app.run(port=5013, debug=False)
