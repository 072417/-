#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
if [[ "$(uname)" == "Darwin" ]] && command -v swiftc >/dev/null; then
  swiftc server/ocr.swift -o server/ocr-helper
else
  echo 'No Apple Vision runtime. Geometry/color analysis works; text detectors will report unavailable.'
fi
npm run build
