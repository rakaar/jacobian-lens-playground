#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
PRELOAD_MODEL="${PRELOAD_MODEL:-1}"
KERNEL_NAME="${KERNEL_NAME:-jacobian-lens}"
WORKSPACE_CACHE="${WORKSPACE_CACHE:-/workspace/.cache}"

export HF_HOME="${HF_HOME:-$WORKSPACE_CACHE/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WORKSPACE_CACHE/pip}"

mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$PIP_CACHE_DIR"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating $VENV_DIR with the image's preinstalled PyTorch..."
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
REQUIREMENTS="$REPO_ROOT/requirements.txt"
REQUIREMENTS_HASH="$(sha256sum "$REQUIREMENTS" | awk '{print $1}')"
INSTALL_STAMP="$VENV_DIR/.requirements-$REQUIREMENTS_HASH"

if [[ ! -f "$INSTALL_STAMP" || "${FORCE_INSTALL:-0}" == "1" ]]; then
  echo "Installing Jacobian Lens notebook dependencies..."
  "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel
  "$VENV_PYTHON" -m pip install -r "$REQUIREMENTS"
  find "$VENV_DIR" -maxdepth 1 -name '.requirements-*' -delete
  touch "$INSTALL_STAMP"
else
  echo "Dependencies already match requirements.txt."
fi

mkdir -p "$HOME/.cache"
if [[ ! -e "$HOME/.cache/huggingface" ]]; then
  ln -s "$HF_HOME" "$HOME/.cache/huggingface"
fi

"$VENV_PYTHON" -m ipykernel install \
  --user \
  --name "$KERNEL_NAME" \
  --display-name "Python (jacobian-lens)"

KERNEL_JSON="$HOME/.local/share/jupyter/kernels/$KERNEL_NAME/kernel.json"
"$VENV_PYTHON" - "$KERNEL_JSON" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" <<'PY'
import json
import pathlib
import sys

kernel_path = pathlib.Path(sys.argv[1])
data = json.loads(kernel_path.read_text())
kernel_env = data.setdefault("env", {})
kernel_env["HF_HOME"] = sys.argv[2]
kernel_env["HUGGINGFACE_HUB_CACHE"] = sys.argv[3]
kernel_path.write_text(json.dumps(data, indent=2) + "\n")
PY

if [[ "$PRELOAD_MODEL" == "1" ]]; then
  echo "Caching $MODEL_ID in $HUGGINGFACE_HUB_CACHE..."
  MODEL_ID="$MODEL_ID" "$VENV_PYTHON" - <<'PY'
import os

from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    token=os.environ.get("HF_TOKEN") or None,
)
print(f"Model cached at: {path}")
PY
else
  echo "Skipping model preload because PRELOAD_MODEL=$PRELOAD_MODEL."
fi

cat <<EOF

Remote environment is ready.
Repository: $REPO_ROOT
Python:     $VENV_PYTHON
Kernel:     Python (jacobian-lens)
Model:      $MODEL_ID
EOF
