# Distress Detection

Flask app that classifies pavement distress (potholes, patching, rutting, ravelling/weathering, cracking) from an uploaded photo, using a Keras CNN (`distressCNN.keras`).

Live demo: https://glyphtechnology.cloud/Pavescan/

## Structure

- `backend.py`: Flask app and `/predict` route (this is what runs in production)
- `images.py`: image preprocessing helpers
- `main.py`: scratch/training script, not used in production
- `templates/`, `static/`: HTML + CSS/JS for the upload form and result page, see [DESIGN.md](DESIGN.md)

## Local setup

```bash
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python backend.py
```

Runs at `http://localhost:5000` (Flask dev server).

If you change `static/styles/main.scss`, rebuild the compiled CSS (see [DESIGN.md](DESIGN.md)):

```bash
npm install
npm run build:css
```

## Deploy

See [DEPLOY.md](DEPLOY.md) for the live VPS setup (gunicorn under pm2, nginx path routing).
