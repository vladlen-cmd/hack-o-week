"""
Data Encryption Engine — AES, Fernet, SHA-256, and Base64 pipelines.
"""
import hashlib, base64, json, os, time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

class EncryptionEngine:
    def __init__(self):
        self.fernet_key = Fernet.generate_key()
        self.fernet = Fernet(self.fernet_key)
        self.aes_key = os.urandom(32)  # 256-bit
        self.stats = {'total_encrypted': 0, 'total_decrypted': 0, 'total_hashed': 0,
                      'bytes_processed': 0, 'history': []}

    def _log(self, method, input_size, output_size, elapsed):
        self.stats['bytes_processed'] += input_size
        self.stats['history'].append({
            'timestamp': time.time(), 'method': method,
            'input_size': input_size, 'output_size': output_size,
            'elapsed_ms': round(elapsed * 1000, 3)
        })
        if len(self.stats['history']) > 500:
            self.stats['history'] = self.stats['history'][-500:]

    # ---- FERNET (symmetric, authenticated) ----
    def fernet_encrypt(self, plaintext):
        t = time.time()
        data = plaintext.encode('utf-8') if isinstance(plaintext, str) else plaintext
        ct = self.fernet.encrypt(data)
        elapsed = time.time() - t
        self.stats['total_encrypted'] += 1
        self._log('Fernet', len(data), len(ct), elapsed)
        return {'ciphertext': ct.decode('utf-8'), 'method': 'Fernet',
                'input_size': len(data), 'output_size': len(ct), 'elapsed_ms': round(elapsed * 1000, 3)}

    def fernet_decrypt(self, ciphertext):
        t = time.time()
        pt = self.fernet.decrypt(ciphertext.encode('utf-8'))
        elapsed = time.time() - t
        self.stats['total_decrypted'] += 1
        self._log('Fernet-Decrypt', len(ciphertext), len(pt), elapsed)
        return {'plaintext': pt.decode('utf-8'), 'method': 'Fernet',
                'elapsed_ms': round(elapsed * 1000, 3)}

    # ---- AES-256-CBC ----
    def aes_encrypt(self, plaintext):
        t = time.time()
        iv = os.urandom(16)
        data = plaintext.encode('utf-8') if isinstance(plaintext, str) else plaintext
        # PKCS7 padding
        pad_len = 16 - (len(data) % 16)
        data_padded = data + bytes([pad_len] * pad_len)
        cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv), backend=default_backend())
        enc = cipher.encryptor()
        ct = enc.update(data_padded) + enc.finalize()
        elapsed = time.time() - t
        self.stats['total_encrypted'] += 1
        result = base64.b64encode(iv + ct).decode('utf-8')
        self._log('AES-256-CBC', len(data), len(result), elapsed)
        return {'ciphertext': result, 'method': 'AES-256-CBC', 'iv': base64.b64encode(iv).decode(),
                'input_size': len(data), 'output_size': len(result), 'elapsed_ms': round(elapsed * 1000, 3)}

    def aes_decrypt(self, ciphertext_b64):
        t = time.time()
        raw = base64.b64decode(ciphertext_b64)
        iv, ct = raw[:16], raw[16:]
        cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv), backend=default_backend())
        dec = cipher.decryptor()
        pt_padded = dec.update(ct) + dec.finalize()
        pad_len = pt_padded[-1]
        pt = pt_padded[:-pad_len]
        elapsed = time.time() - t
        self.stats['total_decrypted'] += 1
        self._log('AES-Decrypt', len(ciphertext_b64), len(pt), elapsed)
        return {'plaintext': pt.decode('utf-8'), 'method': 'AES-256-CBC', 'elapsed_ms': round(elapsed * 1000, 3)}

    # ---- HASHING ----
    def hash_data(self, data, algorithm='sha256'):
        t = time.time()
        encoded = data.encode('utf-8') if isinstance(data, str) else data
        if algorithm == 'sha256':
            h = hashlib.sha256(encoded).hexdigest()
        elif algorithm == 'sha512':
            h = hashlib.sha512(encoded).hexdigest()
        elif algorithm == 'md5':
            h = hashlib.md5(encoded).hexdigest()
        else:
            h = hashlib.sha256(encoded).hexdigest()
        elapsed = time.time() - t
        self.stats['total_hashed'] += 1
        self._log(f'Hash-{algorithm}', len(encoded), len(h), elapsed)
        return {'hash': h, 'algorithm': algorithm, 'elapsed_ms': round(elapsed * 1000, 3)}

    # ---- BASE64 ----
    def base64_encode(self, data):
        encoded = data.encode('utf-8') if isinstance(data, str) else data
        return {'encoded': base64.b64encode(encoded).decode('utf-8'), 'method': 'Base64'}

    def base64_decode(self, data):
        return {'decoded': base64.b64decode(data).decode('utf-8'), 'method': 'Base64'}

    def get_stats(self):
        return self.stats

    def get_pipeline_demo(self, text):
        """Run text through the full encryption pipeline."""
        results = []
        # Step 1: Hash original
        results.append({'step': 1, 'name': 'SHA-256 Hash', **self.hash_data(text)})
        # Step 2: Fernet encrypt
        fern = self.fernet_encrypt(text)
        results.append({'step': 2, 'name': 'Fernet Encrypt', **fern})
        # Step 3: AES encrypt
        aes = self.aes_encrypt(text)
        results.append({'step': 3, 'name': 'AES-256-CBC Encrypt', **aes})
        # Step 4: Base64 encode
        b64 = self.base64_encode(text)
        results.append({'step': 4, 'name': 'Base64 Encode', **b64})
        # Step 5: Verify — decrypt Fernet
        dec = self.fernet_decrypt(fern['ciphertext'])
        results.append({'step': 5, 'name': 'Fernet Decrypt (Verify)', 'verified': dec['plaintext'] == text, **dec})
        return results

if __name__ == '__main__':
    eng = EncryptionEngine()
    demo = eng.get_pipeline_demo("Hello, Hack-O-Week! This is sensitive campus data.")
    for step in demo:
        print(f"  Step {step['step']}: {step['name']} — {step.get('elapsed_ms', 'N/A')}ms")
    print(f"\nStats: {eng.get_stats()['total_encrypted']} encrypted, {eng.get_stats()['total_hashed']} hashed")
