# Plan: Embed PaveScan into React portfolio at `/road-distress`

## Context

You have a React portfolio on Hostinger (static hosting only — no Python runtime). PaveScan is a Flask + Keras app (19 MB model, full TF stack). Goal: visitors hit `your-site.com/road-distress`, see a React page styled like the rest of your portfolio, upload an image, and get a prediction back. Since Hostinger only serves static files, the Flask app has to live elsewhere and be called as an API.

The cleanest split:

- **Frontend** (the upload UI + result view) → rebuilt as a React route inside your existing portfolio, deployed to Hostinger like the rest of the site.
- **Backend** (model inference only) → hosted on a free Python-friendly host, exposed as a JSON API the React page calls via `fetch`.

The current Flask HTML UI stays intact as a standalone demo at the HF Space URL; the React route is an additional client that hits a new JSON endpoint.

## Hosting: Hugging Face Spaces (Docker SDK)

- Free, ML-friendly, 16 GB RAM on the free CPU tier — comfortably fits TF + the 19 MB Keras model.
- Stable URL: `https://<user>-pavescan.hf.space` — both a browsable demo (the current Flask HTML UI) and the API endpoint your React portfolio fetches from.
- Cold start after idle: ~20–40 s. Surface a "warming up" loading state in the React UI for the first request.

## Hostinger routing for `/road-distress`

React Router uses client-side routing, but Hostinger's Apache will return 404 on a hard refresh of `/road-distress` unless you rewrite. Add this to `public/.htaccess` in the React project (gets copied to the deploy root):

```
RewriteEngine On
RewriteRule ^index\.html$ - [L]
RewriteCond %{REQUEST_FILENAME} !-f
RewriteCond %{REQUEST_FILENAME} !-d
RewriteRule . /index.html [L]
```

Then in your React app: `<Route path="/road-distress" element={<PaveScan />} />`.

## Backend changes (this repo) — keep HTML, add JSON

The current Flask app stays as-is for the standalone demo. We add a parallel JSON endpoint that the React portfolio calls.

- Existing form-post `/predict` keeps rendering `result.html` (browsable demo on HF).
- New `POST /api/predict` returns JSON: `{"predicted_class": "...", "severity": "low"|"med"|"high"|"neutral"}`.

Critical files to modify:

- **[backend.py](backend.py)**:
  1. Add `flask-cors`, scope it to the API route only: `CORS(app, resources={r"/api/*": {"origins": ["https://your-portfolio.com"]}})`. The HTML routes stay same-origin.
  2. Add `/api/predict` that runs the same `preprocess_image` → `model.predict` pipeline and returns JSON. Move the severity-derivation logic out of [templates/result.html:37-46](templates/result.html#L37-L46) into a small helper in `backend.py` so both the template and the JSON route use it.
  3. Skip the `img.save(...)` on the JSON path (React previews locally). Keep it on the HTML path so the existing demo still works.
- **[requirements.txt](requirements.txt)** — add `flask-cors` and `gunicorn`.
- **New `Dockerfile`** at repo root for HF Spaces: `python:3.11-slim`, install requirements, `CMD gunicorn backend:app --bind 0.0.0.0:7860 --timeout 120`. HF Spaces expects port 7860; longer timeout covers the first-request model warmup.
- **New `README.md` frontmatter** prepended to README for HF Space config:
  ```
  ---
  title: PaveScan
  sdk: docker
  app_port: 7860
  ---
  ```

Reuse as-is: `preprocess_image` in [backend.py:23](backend.py#L23), `resize_grayscale` in [images.py](images.py), `class_names` in [backend.py:15](backend.py#L15), and the entire `templates/` + `static/styles/` tree.

## Frontend changes (separate React portfolio repo)

This work happens in your portfolio repo, not here. Tell me the path when you're ready and I'll switch over.

- New route `<Route path="/road-distress" element={<PaveScan />} />`.
- Single `PaveScan` component:
  - Drag-drop + file picker (port the JS from [templates/index.html:92-126](templates/index.html#L92-L126) — ~30 lines).
  - On submit: `FormData` with the file, `fetch('https://<user>-pavescan.hf.space/api/predict', { method: 'POST', body: formData })`, parse JSON.
  - Result view inline (no second page) — visual layout adapted from [templates/result.html:28-82](templates/result.html#L28-L82), restyled to match your portfolio's design system rather than copying `main.css` wholesale.
  - Loading states: idle → uploading → "warming up the model" (after 5 s no response) → result | error.
- `public/.htaccess` for SPA routing (so hard-refresh on `/road-distress` doesn't 404) — see snippet above.

## Verification

1. **Backend locally — HTML path still works**: `python backend.py` → open `http://127.0.0.1:5000/`, upload a sample, confirm `result.html` renders as before. (No regression on the existing demo.)
2. **Backend locally — new JSON path**: `curl -F "file=@image\ samples/<some>.jpg" http://127.0.0.1:5000/api/predict` → expect `{"predicted_class": "...", "severity": "..."}`.
3. **Backend on HF Space**: push, wait for Docker build, repeat both checks against the `.hf.space` URL.
4. **CORS**: from your portfolio dev server, fetch `/api/predict` — confirm no CORS error and that an unrelated origin is rejected.
5. **Production routing**: deploy React to Hostinger, hard-refresh `your-site.com/road-distress` — must load, not 404. If it 404s, `.htaccess` didn't ship with the build output.
6. **End-to-end**: upload one image of each distress type from `image samples/`; confirm React result matches the standalone HF demo for the same input.

## Scope of next turn (after you approve)

Backend-only in this repo: `backend.py`, `requirements.txt`, new `Dockerfile`, README frontmatter. React portfolio changes happen in a follow-up turn once you point me at that repo's path.
