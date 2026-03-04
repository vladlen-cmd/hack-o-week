# 🔐 Data Encryption Pipeline
Interactive encryption/decryption dashboard with AES-256, Fernet, hashing, and a full pipeline demo.

## Quick Start
```bash
pip install -r requirements.txt && python app.py
```
Dashboard: http://localhost:5013

## Features
- 🔒 **AES-256-CBC** — Industry-standard symmetric encryption with PKCS7 padding
- 🔑 **Fernet** — Authenticated symmetric encryption (HMAC + AES-128-CBC)
- #️⃣ **SHA-256/512/MD5** — Cryptographic hash generation
- ▶️ **Full Pipeline Demo** — Run data through all encryption stages
- 📊 **Operation History** — Track all encryption operations with performance metrics

## API
- `POST /api/encrypt` · `POST /api/decrypt` · `POST /api/hash` · `POST /api/pipeline` · `GET /api/stats`

## License: MIT — Hack-O-Week
