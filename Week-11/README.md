# 👤 User Registration Portal
A full-stack registration and login portal with profile management, built with Flask + SQLite.

## Quick Start
```bash
pip install -r requirements.txt
python app.py  # → http://localhost:5010
```

## Features
- 📝 **Registration** — Username, email, password with validation
- 🔐 **Login** — Session-based authentication (username or email)
- 👤 **Profile Management** — Edit name and department
- 👥 **User Directory** — View all registered users
- 🎨 **Avatar Colors** — Auto-generated based on username
- 🔒 **Security** — SHA-256 password hashing with salts

## Structure
```
Week-11/
├── app.py              # Flask server + SQLite + auth routes
├── requirements.txt    # flask, flask-cors
├── users.db            # Auto-created SQLite database
└── static/
    ├── index.html      # SPA with auth + dashboard views
    ├── style.css       # Violet/purple dark theme
    └── app.js          # Auth flow + profile + user table
```

## API Endpoints
- `POST /api/register` — Create account
- `POST /api/login` — Authenticate
- `GET /api/profile` — Get current user
- `PUT /api/profile` — Update profile
- `POST /api/logout` — End session
- `GET /api/users` — List all users

## License: MIT — Hack-O-Week
