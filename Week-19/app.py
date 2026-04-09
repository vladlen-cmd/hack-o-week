from __future__ import annotations

import json
import os
import secrets
import sqlite3
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from cryptography.fernet import Fernet
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "compliance_panel.db"
KEY_PATH = BASE_DIR / "compliance_fernet.key"
PORT = 5017

app = Flask(__name__, static_folder=str(BASE_DIR / "static"))
CORS(app)

# Demo users for role-based access control.
USERS = {
    "admin": {"password": "admin123", "role": "admin", "display_name": "Campus Admin"},
    "analyst": {"password": "analyst123", "role": "analyst", "display_name": "Risk Analyst"},
    "viewer": {"password": "viewer123", "role": "viewer", "display_name": "Read Only User"},
}
TOKENS: dict[str, dict] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_or_create_key() -> bytes:
    if KEY_PATH.exists():
        return KEY_PATH.read_bytes().strip()
    key = Fernet.generate_key()
    KEY_PATH.write_bytes(key)
    return key


fernet = Fernet(load_or_create_key())


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def encrypt_json(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return fernet.encrypt(raw).decode("utf-8")


def decrypt_json(ciphertext: str) -> dict:
    plaintext = fernet.decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    return json.loads(plaintext)


def init_db() -> None:
    conn = get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            actor TEXT NOT NULL,
            role TEXT NOT NULL,
            action TEXT NOT NULL,
            resource TEXT NOT NULL,
            encrypted_payload TEXT NOT NULL,
            accessed_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS anomaly_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            severity TEXT NOT NULL,
            encrypted_details TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def seed_data() -> None:
    conn = get_conn()
    logs_count = conn.execute("SELECT COUNT(*) AS c FROM audit_logs").fetchone()["c"]
    reports_count = conn.execute("SELECT COUNT(*) AS c FROM anomaly_reports").fetchone()["c"]

    if logs_count == 0:
        sample_logs = [
            ("analyst", "analyst", "VIEW_ENCRYPTED_RECORD", "student_profile:3448", {"ip": "10.2.14.21", "result": "allowed"}),
            ("viewer", "viewer", "DOWNLOAD_REPORT", "anomaly_batch:APR08", {"ip": "10.2.14.90", "result": "denied"}),
            ("admin", "admin", "EXPORT_AUDIT", "audit_logs:Q2", {"ip": "10.2.14.5", "result": "allowed"}),
        ]
        for actor, role, action, resource, payload in sample_logs:
            conn.execute(
                """
                INSERT INTO audit_logs (actor, role, action, resource, encrypted_payload, accessed_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (actor, role, action, resource, encrypt_json(payload), utc_now_iso()),
            )

    if reports_count == 0:
        sample_reports = [
            ("sleep_lstm", "high", {"reason": "Sustained anomaly probability > 0.9", "user_id": "ST-9912"}, "open"),
            ("activity_lstm", "medium", {"reason": "Unexpected inactivity after low sleep", "user_id": "ST-1130"}, "investigating"),
            ("sleep_lstm", "critical", {"reason": "Repeated severe anomaly across 3 windows", "user_id": "ST-6704"}, "open"),
        ]
        for source, severity, details, status in sample_reports:
            conn.execute(
                """
                INSERT INTO anomaly_reports (source, severity, encrypted_details, status, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source, severity, encrypt_json(details), status, utc_now_iso()),
            )

    conn.commit()
    conn.close()


def _token_from_request() -> str | None:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def require_roles(*allowed_roles: str):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            token = _token_from_request()
            if not token or token not in TOKENS:
                return jsonify({"success": False, "error": "Unauthorized"}), 401

            user = TOKENS[token]
            if user["role"] not in allowed_roles:
                return jsonify({"success": False, "error": "Forbidden: insufficient role"}), 403

            request.user = user
            request.token = token
            return fn(*args, **kwargs)

        return wrapper

    return decorator


def write_audit(actor: str, role: str, action: str, resource: str, payload: dict) -> None:
    conn = get_conn()
    conn.execute(
        """
        INSERT INTO audit_logs (actor, role, action, resource, encrypted_payload, accessed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (actor, role, action, resource, encrypt_json(payload), utc_now_iso()),
    )
    conn.commit()
    conn.close()


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "service": "admin-compliance-panel"})


@app.route("/api/login", methods=["POST"])
def login():
    body = request.get_json(silent=True) or {}
    username = str(body.get("username", "")).strip().lower()
    password = str(body.get("password", "")).strip()

    user = USERS.get(username)
    if not user or user["password"] != password:
        return jsonify({"success": False, "error": "Invalid credentials"}), 401

    token = secrets.token_urlsafe(24)
    TOKENS[token] = {
        "username": username,
        "role": user["role"],
        "display_name": user["display_name"],
    }

    write_audit(
        actor=username,
        role=user["role"],
        action="LOGIN",
        resource="admin_panel",
        payload={"result": "success", "ip": request.remote_addr},
    )

    return jsonify(
        {
            "success": True,
            "token": token,
            "user": {
                "username": username,
                "role": user["role"],
                "display_name": user["display_name"],
            },
        }
    )


@app.route("/api/logout", methods=["POST"])
@require_roles("admin", "analyst", "viewer")
def logout():
    user = request.user
    token = request.token
    TOKENS.pop(token, None)

    write_audit(
        actor=user["username"],
        role=user["role"],
        action="LOGOUT",
        resource="admin_panel",
        payload={"result": "success"},
    )

    return jsonify({"success": True})


@app.route("/api/me")
@require_roles("admin", "analyst", "viewer")
def me():
    return jsonify({"success": True, "user": request.user})


@app.route("/api/admin/audit-logs")
@require_roles("admin")
def get_audit_logs():
    limit = min(int(request.args.get("limit", 100)), 300)
    action_filter = request.args.get("action", "").strip()

    conn = get_conn()
    if action_filter:
        rows = conn.execute(
            """
            SELECT * FROM audit_logs
            WHERE action = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (action_filter, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT * FROM audit_logs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    conn.close()

    logs = []
    for row in rows:
        logs.append(
            {
                "id": row["id"],
                "actor": row["actor"],
                "role": row["role"],
                "action": row["action"],
                "resource": row["resource"],
                "payload": decrypt_json(row["encrypted_payload"]),
                "accessed_at": row["accessed_at"],
            }
        )

    write_audit(
        actor=request.user["username"],
        role=request.user["role"],
        action="VIEW_AUDIT_LOGS",
        resource="audit_logs",
        payload={"count": len(logs)},
    )

    return jsonify({"success": True, "logs": logs})


@app.route("/api/admin/anomaly-reports")
@require_roles("admin", "analyst")
def get_anomaly_reports():
    severity_filter = request.args.get("severity", "").strip().lower()

    conn = get_conn()
    rows = conn.execute("SELECT * FROM anomaly_reports ORDER BY id DESC").fetchall()
    conn.close()

    reports = []
    for row in rows:
        severity = row["severity"].lower()
        if severity_filter and severity != severity_filter:
            continue
        reports.append(
            {
                "id": row["id"],
                "source": row["source"],
                "severity": row["severity"],
                "details": decrypt_json(row["encrypted_details"]),
                "status": row["status"],
                "created_at": row["created_at"],
            }
        )

    write_audit(
        actor=request.user["username"],
        role=request.user["role"],
        action="VIEW_ANOMALY_REPORTS",
        resource="anomaly_reports",
        payload={"count": len(reports), "severity_filter": severity_filter or "all"},
    )

    return jsonify({"success": True, "reports": reports})


@app.route("/api/admin/anomaly-reports", methods=["POST"])
@require_roles("admin", "analyst")
def create_anomaly_report():
    body = request.get_json(silent=True) or {}
    source = str(body.get("source", "sleep_lstm")).strip() or "sleep_lstm"
    severity = str(body.get("severity", "medium")).strip().lower()
    reason = str(body.get("reason", "Model flagged abnormal trend.")).strip()
    user_id = str(body.get("user_id", "ST-0000")).strip()
    status = str(body.get("status", "open")).strip().lower()

    if severity not in {"low", "medium", "high", "critical"}:
        return jsonify({"success": False, "error": "Invalid severity"}), 400

    if status not in {"open", "investigating", "resolved"}:
        return jsonify({"success": False, "error": "Invalid status"}), 400

    conn = get_conn()
    cur = conn.execute(
        """
        INSERT INTO anomaly_reports (source, severity, encrypted_details, status, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (source, severity, encrypt_json({"reason": reason, "user_id": user_id}), status, utc_now_iso()),
    )
    report_id = cur.lastrowid
    conn.commit()
    conn.close()

    write_audit(
        actor=request.user["username"],
        role=request.user["role"],
        action="CREATE_ANOMALY_REPORT",
        resource=f"anomaly_report:{report_id}",
        payload={"source": source, "severity": severity},
    )

    return jsonify({"success": True, "report_id": report_id})


if __name__ == "__main__":
    init_db()
    seed_data()
    print("=" * 64)
    print("Week 19 - Admin Compliance Panel")
    print("=" * 64)
    print("Dashboard: http://localhost:5017")
    print("Demo users: admin/admin123, analyst/analyst123, viewer/viewer123")
    app.run(port=PORT, debug=False)
