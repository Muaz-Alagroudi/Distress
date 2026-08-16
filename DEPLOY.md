# Deploy

Live demo: https://glyphtechnology.cloud/Pavescan/

## What this is

Flask app (`backend.py`) serving an upload form that runs a Keras CNN (`distressCNN.keras`) to classify pavement distress in an uploaded image.

## VPS setup

- App root: `/var/www/Distress`
- Python venv: `/var/www/Distress/venv` (flask, gunicorn, tensorflow, keras, pillow, numpy, matplotlib)
- Run via gunicorn under pm2, process name `distress-demo-api`, listening on `127.0.0.1:4101`
- `APP_PREFIX=/Pavescan` env var tells the app it's mounted under a path prefix (see `PrefixMiddleware` in `backend.py`), so `url_for()` generates correct `/Pavescan/...` links. Set inline on the pm2 start command (no ecosystem file); re-set it if the process is ever deleted and recreated.
- nginx: `location /Pavescan/` in `/var/www/glyphtechnology.cloud/nginx/glyphtechnology.cloud.conf` proxies to port 4101 (full path forwarded, not stripped). The old `/distress/` path 301-redirects to `/Pavescan/` for existing links.
- Port registered in `/var/www/glyphtechnology.cloud/PORTS.md`

## Redeploy after changes

```bash
cd /var/www/Distress
git pull   # if/when this becomes a git remote-backed repo
venv/bin/pip install -r requirements.txt   # if deps changed
npm run build:css   # if static/styles/main.scss changed, see DESIGN.md; skip if only main.css changed
pm2 restart distress-demo-api
```

## Notes

- Templates (`templates/index.html`, `templates/result.html`) use `url_for()` for the predict/index form actions and static assets so they resolve correctly under the `/Pavescan` prefix; don't replace those with hardcoded absolute paths.
- `static/styles/main.css` is compiled from `static/styles/main.scss` and committed, so the VPS doesn't need Node installed to serve the site as-is. Only run `npm run build:css` there if you edited the `.scss` source directly on the box instead of building locally. See [DESIGN.md](DESIGN.md).
