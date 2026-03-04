from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from encryption_engine import EncryptionEngine

app = Flask(__name__, static_folder='static')
CORS(app)
engine = EncryptionEngine()

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/encrypt', methods=['POST'])
def encrypt():
    try:
        data = request.get_json()
        text = data.get('text', '')
        method = data.get('method', 'fernet')
        if not text: return jsonify({'success': False, 'error': 'No text'}), 400
        if method == 'fernet':
            r = engine.fernet_encrypt(text)
        elif method == 'aes':
            r = engine.aes_encrypt(text)
        else:
            return jsonify({'success': False, 'error': f'Unknown method: {method}'}), 400
        return jsonify({'success': True, **r})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/decrypt', methods=['POST'])
def decrypt():
    try:
        data = request.get_json()
        ct = data.get('ciphertext', '')
        method = data.get('method', 'fernet')
        if not ct: return jsonify({'success': False, 'error': 'No ciphertext'}), 400
        if method == 'fernet':
            r = engine.fernet_decrypt(ct)
        elif method == 'aes':
            r = engine.aes_decrypt(ct)
        else:
            return jsonify({'success': False, 'error': f'Unknown method'}), 400
        return jsonify({'success': True, **r})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/hash', methods=['POST'])
def hash_data():
    try:
        data = request.get_json()
        text = data.get('text', '')
        algo = data.get('algorithm', 'sha256')
        if not text: return jsonify({'success': False, 'error': 'No text'}), 400
        r = engine.hash_data(text, algo)
        return jsonify({'success': True, **r})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pipeline', methods=['POST'])
def pipeline():
    try:
        data = request.get_json()
        text = data.get('text', '')
        if not text: return jsonify({'success': False, 'error': 'No text'}), 400
        results = engine.get_pipeline_demo(text)
        return jsonify({'success': True, 'steps': results})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def stats():
    try:
        return jsonify({'success': True, 'stats': engine.get_stats()})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60); print("Data Encryption Pipeline"); print("=" * 60)
    print("✓ http://localhost:5013")
    app.run(debug=False, port=5013)
