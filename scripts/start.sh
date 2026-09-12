#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ ! -x .venv/bin/python || ! -d dist ]]; then
  echo 'Run bash scripts/setup.sh first.'
  exit 1
fi
echo '对照工作台：http://127.0.0.1:8765/'
exec .venv/bin/python -m uvicorn server.app:app --host 127.0.0.1 --port 8765
