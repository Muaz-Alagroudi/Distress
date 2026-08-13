# Deploy

Live demo: https://glyphtechnology.cloud/distress/

## What this is

Flask app (`backend.py`) serving an upload form that runs a Keras CNN (`distressCNN.keras`) to classify pavement distress in an uploaded image.

## VPS setup

- App root: `/var/www/Distress`
- Python venv: `/var/www/Distress/venv` (flask, gunicorn, tensorflow, keras, pillow, numpy, matplotlib)
- Run via gunicorn under pm2, process name `distress-demo-api`, listening on `127.0.0.1:4101`
- `APP_PREFIX=/distress` env var tells the app it's mounted under a path prefix (see `PrefixMiddleware` in `backend.py`), so `url_for()` generates correct `/distress/...` links
- nginx: `location /distress/` in `/var/www/glyphtechnology.cloud/nginx/glyphtechnology.cloud.conf` proxies to port 4101 (full path forwarded, not stripped)
- Port registered in `/var/www/glyphtechnology.cloud/PORTS.md`

## Redeploy after changes

```bash
cd /var/www/Distress
git pull   # if/when this becomes a git remote-backed repo
venv/bin/pip install -r requirements.txt   # if deps changed
pm2 restart distress-demo-api
```

## Notes

- Templates (`templates/index.html`, `templates/result.html`) use `url_for()` for the predict/index form actions specifically so they resolve correctly under the `/distress` prefix; don't replace those with hardcoded absolute paths.
