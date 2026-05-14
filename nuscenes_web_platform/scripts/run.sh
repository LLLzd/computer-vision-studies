#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}"
# Optional: load .env if present (export all vars)
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ROOT}/.env"
  set +a
fi
export NUSCENES_DATAROOT="${NUSCENES_DATAROOT:-${ROOT}/../demo_by_nuscenes/data/nuscenes}"
export NUSCENES_VERSION="${NUSCENES_VERSION:-v1.0-mini}"
export NUSCENES_WEB_HOST="${NUSCENES_WEB_HOST:-127.0.0.1}"
export NUSCENES_WEB_PORT="${NUSCENES_WEB_PORT:-8765}"
exec python3 -m uvicorn backend.main:app --host "${NUSCENES_WEB_HOST}" --port "${NUSCENES_WEB_PORT}" "$@"
