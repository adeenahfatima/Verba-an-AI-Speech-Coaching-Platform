# VERBA — AI Speech Coaching Platform

VERBA is a web app that analyzes recorded or uploaded speech and gives feedback on
clarity, pace, pitch, filler words, and vocabulary. It transcribes audio with
OpenAI Whisper and runs acoustic analysis with librosa and Praat (parselmouth).

---

## Tech Stack

 Backend:              Flask, Flask-SQLAlchemy, Gunicorn 
 
 Database:           PostgreSQL (production) / SQLite (local dev fallback) 
 
 Migrations:         Alembic 
 
 Speech-to-text:     OpenAI Whisper 
 
 Audio analysis:     librosa, parselmouth (Praat), pyAudioAnalysis 
 
 NLP:                NLTK 
 
 Frontend:           Plain HTML/CSS/JS (no build step, served directly by Flask) 
 
 Auth:               Flask-Bcrypt (password hashing) 

---

## Project Structure

```
Verba-prod/
├── backend/
│   └── api.py              # Main Flask app: routes, models, audio analysis logic
├── frontend/
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── upload.html
│   ├── AboutUs.html
│   ├── styles.css
│   └── *.png / *.jpg       # Images/assets
├── migrations/
│   ├── alembic.ini
│   ├── env.py
│   └── versions/           # Auto-generated migration scripts
├── scripts/
│   ├── migrate_uploads.py      # One-off data migration helper
│   ├── serve_frontend_local.py # Standalone static server for local frontend-only dev
│   └── test_hf_api.py          # Manual script to test HuggingFace inference API
├── requirements.txt
├── Procfile                 # Gunicorn start command (used by Render/Railway)
├── render.yaml               # Render deployment blueprint
├── .env.example
└── .gitignore
```

---

## Environment Variables

Create a `.env` file locally based on `.env.example`:

```env
SECRET_KEY=your-random-secret-key
DATABASE_URL=postgresql://user:password@host:5432/verba
HF_API_KEY=                    # optional — enables AI-generated tips, leave blank to disable
```

Variables:
`SECRET_KEY`:     Used for Flask session signing 
`DATABASE_URL`:   Falls back to local SQLite if unset 
`HF_API_KEY`:     If unset, AI tips section is skipped gracefully (`ai_advice: null`) 
`PORT`:           Auto-set by host. Defaults to `5000` locally

---

## Local Development Setup

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/verba_prod.git
cd verba_prod
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: `torch` and `openai-whisper` are large downloads (~2GB+). First install will take a while.

### 4. Set up environment variables

```bash
cp .env.example .env
# then edit .env with your local values
```

For local dev, you can leave `DATABASE_URL` unset to use SQLite automatically.

### 5. Run database migrations (optional — tables also auto-create on first run)

```bash
cd migrations
alembic upgrade head
cd ..
```

### 6. Start the server

```bash
python backend/api.py
```

The app will be available at **http://localhost:5000** — Flask serves both the
API and the frontend HTML pages from this single server.

---

## Running Migrations

This project uses Alembic for schema migrations.

```bash
cd migrations

# Create a new migration after changing models in backend/api.py
alembic revision --autogenerate -m "describe your change"

# Apply migrations
alembic upgrade head

# Roll back one migration
alembic downgrade -1
```

---

## API Reference

All routes are available both with and without the `/api` prefix for backward
compatibility — use `/api/...` for new integrations.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | Health check — confirms Whisper model is loaded |
| POST | `/api/transcribe_upload` | Upload an audio file for transcription + analysis |
| GET | `/api/uploads/<user_id>` | List all past uploads/analyses for a user |
| GET | `/api/profile/<user_id>` | Get user profile info |
| POST | `/api/register` | Create a new user account |
| POST | `/api/login` | Authenticate a user |

### Example: Upload audio for analysis

```bash
curl -X POST http://localhost:5000/api/transcribe_upload \
  -F "audio=@sample.ogg" \
  -F "user_id=1"
```

---

## Deployment

This app is configured for **Railway** (or any host supporting a `Procfile`).
Check it out: verbaprod-production.up.railway.app


## Local-only Helper Scripts

`scripts/serve_frontend_local.py`:  Serves only the `frontend/` folder standalone (useful if testing frontend without running the full Flask backend) 
`scripts/migrate_uploads.py`:       One-off script for migrating old upload records to a new schema 
`scripts/test_hf_api.py`:           Manually test a HuggingFace inference endpoint outside the app 


## Known Limitations

- AI-generated tips (`ai_advice` field) require a HuggingFace API key (`HF_API_KEY`).
  Without it, the app falls back to rule-based tips automatically.
- Free-tier hosting (Railway/Render) has limited CPU — expect slower processing for longer audio files compared to local development on a capable machine.

---
