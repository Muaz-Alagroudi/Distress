# Distress Detection

Flask app that classifies pavement distress (potholes, patching, rutting, ravelling/weathering, cracking) from an uploaded photo, using a Keras CNN (`distressCNN.keras`).

Live demo: https://glyphtechnology.cloud/distress/

## Structure

- `backend.py`: Flask app and `/predict` route (this is what runs in production)
- `images.py`: image preprocessing helpers
- `main.py`: scratch/training script, not used in production
- `templates/`, `static/`: HTML + CSS for the upload form and result page

## Local setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python backend.py
```

Runs at `http://localhost:5000` (Flask dev server).

## Deploy

See [DEPLOY.md](DEPLOY.md) for the live VPS setup (gunicorn under pm2, nginx path routing).
