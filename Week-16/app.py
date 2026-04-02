from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from cryptography.fernet import Fernet, InvalidToken
import sqlite3
import json
import os
from datetime import datetime, timezone

app = Flask(__name__, static_folder="static")
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

DB_PATH = "realtime_alerts.db"
KEY_PATH = "alert_fernet.key"
PORT = 5015

THRESHOLDS = {
    "high_bpm": 110,
    "critical_bpm": 130,
    "low_bpm": 45,
    "low_spo2": 92,
    "high_temp_c": 38.0,
    "jump_bpm": 35,
    "jump_window_seconds": 30,
}



def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()



def load_or_create_key():
    if os.path.exists(KEY_PATH):
        with open(KEY_PATH, "rb") as f:
            key = f.read().strip()
            return key

    key = Fernet.generate_key()
    with open(KEY_PATH, "wb") as f:
        f.write(key)
    return key


FERNET_KEY = load_or_create_key()
fernet = Fernet(FERNET_KEY)



def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn



def init_db():
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            device TEXT,
            bpm REAL NOT NULL,
            spo2 REAL,
            temp_c REAL,
            source TEXT,
            is_anomaly INTEGER DEFAULT 0,
            anomaly_reasons TEXT,
            severity TEXT,
            received_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            severity TEXT NOT NULL,
            reason_summary TEXT NOT NULL,
            encrypted_payload TEXT NOT NULL,
            delivered INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()



def _to_float(value, field_name, required=False):
    if value is None:
        if required:
            raise ValueError(f"Missing required field: {field_name}")
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for {field_name}")



def get_latest_bpm(user_id):
    conn = get_conn()
    row = conn.execute(
        "SELECT bpm, received_at FROM readings WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    ).fetchone()
    conn.close()
    return row



def detect_anomaly(user_id, bpm, spo2=None, temp_c=None):
    reasons = []
    severity = "info"

    if bpm >= THRESHOLDS["critical_bpm"]:
        reasons.append(f"Critical BPM: {bpm:.1f} >= {THRESHOLDS['critical_bpm']}")
        severity = "critical"
    elif bpm >= THRESHOLDS["high_bpm"]:
        reasons.append(f"High BPM: {bpm:.1f} >= {THRESHOLDS['high_bpm']}")
        severity = "warning"

    if bpm <= THRESHOLDS["low_bpm"]:
        reasons.append(f"Low BPM: {bpm:.1f} <= {THRESHOLDS['low_bpm']}")
        if severity != "critical":
            severity = "warning"

    if spo2 is not None and spo2 < THRESHOLDS["low_spo2"]:
        reasons.append(f"Low SpO2: {spo2:.1f}% < {THRESHOLDS['low_spo2']}%")
        if severity != "critical":
            severity = "warning"

    if temp_c is not None and temp_c >= THRESHOLDS["high_temp_c"]:
        reasons.append(f"High temperature: {temp_c:.1f}C >= {THRESHOLDS['high_temp_c']}C")
        if severity != "critical":
            severity = "warning"

    prev = get_latest_bpm(user_id)
    if prev is not None:
        prev_bpm = float(prev["bpm"])
        prev_time = datetime.fromisoformat(prev["received_at"])
        now = datetime.now(timezone.utc)
        delta_seconds = (now - prev_time).total_seconds()
        jump = abs(bpm - prev_bpm)
        if delta_seconds <= THRESHOLDS["jump_window_seconds"] and jump >= THRESHOLDS["jump_bpm"]:
            reasons.append(
                f"Sudden BPM jump: {jump:.1f} within {int(delta_seconds)}s"
            )
            if severity != "critical":
                severity = "warning"

    is_anomaly = len(reasons) > 0
    return is_anomaly, reasons, severity



def encrypt_notification(payload):
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return fernet.encrypt(raw).decode("utf-8")



def decrypt_notification(ciphertext):
    plaintext = fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    return json.loads(plaintext)



def create_notification(user_id, severity, reasons, reading_payload):
    created_at = utc_now_iso()
    payload = {
        "type": "physiological-alert",
        "user_id": user_id,
        "severity": severity,
        "reasons": reasons,
        "reading": reading_payload,
        "created_at": created_at,
    }
    encrypted_payload = encrypt_notification(payload)

    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO notifications (user_id, severity, reason_summary, encrypted_payload, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, severity, "; ".join(reasons), encrypted_payload, created_at),
    )
    notification_id = cur.lastrowid
    conn.commit()
    conn.close()

    return {
        "notification_id": notification_id,
        "user_id": user_id,
        "severity": severity,
        "ciphertext": encrypted_payload,
        "algorithm": "Fernet",
        "created_at": created_at,
    }



def process_stream_event(data, source):
    if not isinstance(data, dict):
        raise ValueError("Payload must be a JSON object")

    user_id = str(data.get("user_id", "anonymous"))
    device = str(data.get("device", "unknown"))
    bpm = _to_float(data.get("bpm"), "bpm", required=True)
    spo2 = _to_float(data.get("spo2"), "spo2")
    temp_c = _to_float(data.get("temp_c"), "temp_c")

    is_anomaly, reasons, severity = detect_anomaly(user_id, bpm, spo2, temp_c)
    received_at = utc_now_iso()

    conn = get_conn()
    conn.execute(
        """
        INSERT INTO readings (
            user_id, device, bpm, spo2, temp_c, source,
            is_anomaly, anomaly_reasons, severity, received_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            user_id,
            device,
            bpm,
            spo2,
            temp_c,
            source,
            1 if is_anomaly else 0,
            json.dumps(reasons),
            severity,
            received_at,
        ),
    )
    conn.commit()
    conn.close()

    reading_payload = {
        "user_id": user_id,
        "device": device,
        "bpm": bpm,
        "spo2": spo2,
        "temp_c": temp_c,
        "source": source,
        "received_at": received_at,
    }

    result = {
        "success": True,
        "is_anomaly": is_anomaly,
        "severity": severity,
        "reasons": reasons,
        "reading": reading_payload,
    }

    if is_anomaly:
        push_msg = create_notification(user_id, severity, reasons, reading_payload)
        result["notification"] = push_msg
        socketio.emit("encrypted_alert", push_msg, broadcast=True)

    socketio.emit(
        "stream_processed",
        {
            "user_id": user_id,
            "bpm": bpm,
            "is_anomaly": is_anomaly,
            "severity": severity,
            "received_at": received_at,
        },
        broadcast=True,
    )

    return result


@app.route("/")
def home():
    return send_from_directory("static", "index.html")


@app.route("/api")
def api_home():
    return jsonify(
        {
            "service": "real-time-alert-system",
            "status": "running",
            "version": "1.0",
            "endpoints": [
                "/api/health",
                "/api/stream",
                "/api/alerts",
                "/api/alerts/<id>/ack",
                "/api/decrypt",
                "/api/stats",
            ],
        }
    )


@app.route("/api/health")
def health():
    return jsonify({"success": True, "time": utc_now_iso(), "encryption": "Fernet"})


@app.route("/api/stream", methods=["POST"])
def ingest_stream():
    try:
        payload = request.get_json(silent=True) or {}
        result = process_stream_event(payload, source="rest")
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"success": False, "error": f"Internal error: {exc}"}), 500


@app.route("/api/alerts")
def list_alerts():
    limit = request.args.get("limit", default=25, type=int)
    limit = min(max(limit, 1), 100)

    conn = get_conn()
    rows = conn.execute(
        """
        SELECT id, user_id, severity, reason_summary, encrypted_payload, delivered, created_at
        FROM notifications ORDER BY id DESC LIMIT ?
        """,
        (limit,),
    ).fetchall()
    conn.close()

    alerts = [dict(r) for r in rows]
    return jsonify({"success": True, "count": len(alerts), "alerts": alerts})


@app.route("/api/alerts/<int:notification_id>/ack", methods=["POST"])
def ack_alert(notification_id):
    conn = get_conn()
    cur = conn.execute(
        "UPDATE notifications SET delivered = 1 WHERE id = ?",
        (notification_id,),
    )
    conn.commit()
    updated = cur.rowcount
    conn.close()

    if updated == 0:
        return jsonify({"success": False, "error": "Notification not found"}), 404

    return jsonify({"success": True, "notification_id": notification_id, "delivered": True})


@app.route("/api/decrypt", methods=["POST"])
def decrypt_api():
    payload = request.get_json(silent=True) or {}
    ciphertext = payload.get("ciphertext", "")
    if not ciphertext:
        return jsonify({"success": False, "error": "ciphertext is required"}), 400

    try:
        decrypted = decrypt_notification(ciphertext)
        return jsonify({"success": True, "payload": decrypted})
    except InvalidToken:
        return jsonify({"success": False, "error": "Invalid ciphertext/token"}), 400


@app.route("/api/stats")
def stats():
    conn = get_conn()
    total_readings = conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    total_alerts = conn.execute("SELECT COUNT(*) FROM notifications").fetchone()[0]
    undelivered = conn.execute(
        "SELECT COUNT(*) FROM notifications WHERE delivered = 0"
    ).fetchone()[0]
    critical = conn.execute(
        "SELECT COUNT(*) FROM notifications WHERE severity = 'critical'"
    ).fetchone()[0]
    conn.close()

    return jsonify(
        {
            "success": True,
            "total_readings": total_readings,
            "total_alerts": total_alerts,
            "undelivered_alerts": undelivered,
            "critical_alerts": critical,
            "thresholds": THRESHOLDS,
            "algorithm": "Rule-based anomaly detection + Fernet encryption",
        }
    )


@socketio.on("connect")
def on_connect():
    emit(
        "status",
        {
            "message": "Connected to real-time alert stream",
            "encryption": "Fernet",
            "timestamp": utc_now_iso(),
        },
    )


@socketio.on("stream_data")
def on_stream_data(data):
    try:
        result = process_stream_event(data, source="websocket")
        emit("ingest_ack", result)
    except ValueError as exc:
        emit("ingest_ack", {"success": False, "error": str(exc)})
    except Exception as exc:
        emit("ingest_ack", {"success": False, "error": f"Internal error: {exc}"})


if __name__ == "__main__":
    init_db()
    print("=" * 60)
    print("Week-16: Real-Time Alert System")
    print("Monitoring streams, detecting anomalies, pushing encrypted alerts")
    print("=" * 60)
    print(f"Server running at: http://localhost:{PORT}")
    socketio.run(app, port=PORT, debug=False, allow_unsafe_werkzeug=True)
