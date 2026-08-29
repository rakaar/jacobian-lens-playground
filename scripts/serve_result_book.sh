#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RESULT_BOOK_PORT="${RESULT_BOOK_PORT:-40884}"

cd "$REPO_ROOT"
exec .venv/bin/python -m mkdocs serve \
  --dev-addr "127.0.0.1:$RESULT_BOOK_PORT"
