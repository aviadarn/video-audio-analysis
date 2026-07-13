#!/usr/bin/env bash
set -euo pipefail
role="${1:-api}"
case "$role" in
  api)         exec uvicorn celebvision.api.app:app --host 0.0.0.0 --port 8000 ;;
  coordinator) exec python -m celebvision.coordinator_main ;;
  worker)      exec python -m celebvision.workers "${2}" ;;
  init)        exec python scripts/init_stack.py ;;
  *) echo "unknown role: $role"; exit 1 ;;
esac
