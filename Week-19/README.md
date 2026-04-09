# Week 19 - Admin Compliance Panel

Professional admin dashboard for encrypted data access audit logs and anomaly report monitoring, with role-based access control.

## Features
- Encrypted payload storage for audit and anomaly records (Fernet)
- Role-based access:
  - `admin`: full access (audit logs + anomaly reports)
  - `analyst`: anomaly reports only
  - `viewer`: no admin report access
- Login/logout session token flow
- Professional web dashboard with KPI cards, filters, and secure tables

## Demo Credentials
- `admin / admin123`
- `analyst / analyst123`
- `viewer / viewer123`

## Run
```bash
pip install -r requirements.txt
python app.py
```

Open: http://localhost:5017

## API Summary
- `POST /api/login`
- `POST /api/logout`
- `GET /api/me`
- `GET /api/admin/audit-logs` (admin only)
- `GET /api/admin/anomaly-reports` (admin/analyst)
- `POST /api/admin/anomaly-reports` (admin/analyst)
