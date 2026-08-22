#!/usr/bin/env bash
# Redeploy Distress (PaveScan) after a git pull or local edit.
# Usage: ./restart.sh -s (server only) | -c (client/CSS only) | -a (both)
set -euo pipefail
cd "$(dirname "$0")"

FLAG="${1:--a}"

build_client() {
    echo "==> Installing client deps (if changed) and building CSS"
    npm install
    npm run build:css
}

restart_server() {
    echo "==> Installing Python deps (if changed)"
    venv/bin/pip install -r requirements.txt
    echo "==> Restarting pm2 process distress-demo-api"
    pm2 restart distress-demo-api
}

case "$FLAG" in
    -c)
        build_client
        ;;
    -s)
        restart_server
        ;;
    -a)
        build_client
        restart_server
        ;;
    *)
        echo "Usage: $0 [-s|-c|-a]" >&2
        exit 1
        ;;
esac

echo "==> Done"
pm2 status distress-demo-api
