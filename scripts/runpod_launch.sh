#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STATE_DIR="$REPO_ROOT/.runpod"

RUNPODCTL="${RUNPODCTL:-$HOME/.local/bin/runpodctl}"
GPU_ID="${1:-NVIDIA GeForce RTX 3090}"
CLOUD_TYPE="${CLOUD_TYPE:-COMMUNITY}"
POD_NAME="${POD_NAME:-jlens-qwen3}"
RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404}"
CONTAINER_DISK_GB="${CONTAINER_DISK_GB:-20}"
VOLUME_GB="${VOLUME_GB:-30}"
RUNPOD_NETWORK_VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-}"
RUNPOD_WAIT_SECONDS="${RUNPOD_WAIT_SECONDS:-600}"
SSH_ALIAS="${SSH_ALIAS:-jlens-runpod}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B}"
PRELOAD_MODEL="${PRELOAD_MODEL:-1}"
OPEN_VSCODE="${OPEN_VSCODE:-0}"
REPO_URL="${REPO_URL:-https://github.com/rakaar/jacobian-lens-playground.git}"
REMOTE_REPO="${REMOTE_REPO:-/workspace/jacobian-lens-playground}"

fail() {
  echo "Error: $*" >&2
  exit 1
}

for command_name in jq ssh git; do
  command -v "$command_name" >/dev/null ||
    fail "Required local command is missing: $command_name"
done

[[ -x "$RUNPODCTL" ]] || fail "runpodctl was not found at $RUNPODCTL"
[[ "$CLOUD_TYPE" == "COMMUNITY" || "$CLOUD_TYPE" == "SECURE" ]] ||
  fail "CLOUD_TYPE must be COMMUNITY or SECURE"

mkdir -p "$STATE_DIR"

if [[ -s "$STATE_DIR/pod-id" ]]; then
  previous_pod="$(cat "$STATE_DIR/pod-id")"
  if "$RUNPODCTL" pod get "$previous_pod" >/dev/null 2>&1; then
    fail "Pod $previous_pod still exists. Delete it before creating another."
  fi
  rm -f "$STATE_DIR/pod-id"
fi

create_args=(
  pod create
  --name "$POD_NAME"
  --gpu-id "$GPU_ID"
  --gpu-count 1
  --cloud-type "$CLOUD_TYPE"
  --image "$RUNPOD_IMAGE"
  --container-disk-in-gb "$CONTAINER_DISK_GB"
  --volume-mount-path /workspace
  --ports "22/tcp,8888/http"
  --ssh
)

if [[ -n "$RUNPOD_NETWORK_VOLUME_ID" ]]; then
  create_args+=(--network-volume-id "$RUNPOD_NETWORK_VOLUME_ID")
else
  create_args+=(--volume-in-gb "$VOLUME_GB")
fi

if [[ "$CLOUD_TYPE" == "COMMUNITY" ]]; then
  create_args+=(--public-ip)
fi

if [[ -n "${RUNPOD_STOP_AFTER:-}" ]]; then
  create_args+=(--stop-after "$RUNPOD_STOP_AFTER")
fi

if [[ -n "${RUNPOD_TERMINATE_AFTER:-}" ]]; then
  create_args+=(--terminate-after "$RUNPOD_TERMINATE_AFTER")
fi

echo "Creating one $GPU_ID pod in $CLOUD_TYPE cloud..."
create_json="$("$RUNPODCTL" "${create_args[@]}" -o json)"
printf '%s\n' "$create_json" >"$STATE_DIR/create.json"

pod_id="$(
  jq -r '.id // .pod.id // .data.id // .podId // empty' \
    <<<"$create_json"
)"
[[ -n "$pod_id" ]] || fail "RunPod created no pod ID: $create_json"
printf '%s\n' "$pod_id" >"$STATE_DIR/pod-id"

cost_per_hour="$(
  jq -r '.adjustedCostPerHr // .costPerHr // .pod.costPerHr // empty' \
    <<<"$create_json"
)"

echo "Created pod: $pod_id"
if [[ -n "$cost_per_hour" ]]; then
  echo "Reported compute rate: \$$cost_per_hour/hour"
fi
echo "Billing has started. If setup fails, delete it with:"
echo "  $RUNPODCTL pod delete $pod_id"

deadline=$((SECONDS + RUNPOD_WAIT_SECONDS))
pod_json=""
ssh_ip=""
ssh_port=""
ssh_key=""

echo "Waiting for direct SSH..."
while ((SECONDS < deadline)); do
  if pod_json="$("$RUNPODCTL" pod get "$pod_id" -o json 2>/dev/null)"; then
    status="$(jq -r '.desiredStatus // "UNKNOWN"' <<<"$pod_json")"
    case "$status" in
      EXITED | TERMINATED | ERROR | UNKNOWN | OFFLINE)
        fail "Pod entered failure state: $status"
        ;;
    esac

    ssh_ip="$(jq -r '.ssh.ip // empty' <<<"$pod_json")"
    ssh_port="$(jq -r '.ssh.port // empty' <<<"$pod_json")"
    ssh_key="$(jq -r '.ssh.ssh_key.path // empty' <<<"$pod_json")"
    if [[ -n "$ssh_ip" && -n "$ssh_port" && -f "$ssh_key" ]]; then
      break
    fi
  fi
  sleep 8
done

[[ -n "$ssh_ip" && -n "$ssh_port" && -f "$ssh_key" ]] ||
  fail "Timed out waiting for SSH. Pod $pod_id is still billable."

ssh_args=(
  -i "$ssh_key"
  -p "$ssh_port"
  -o BatchMode=yes
  -o ConnectTimeout=10
  -o StrictHostKeyChecking=accept-new
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)

ssh_ready=0
for _ in $(seq 1 30); do
  if ssh "${ssh_args[@]}" "root@$ssh_ip" true 2>/dev/null; then
    ssh_ready=1
    break
  fi
  sleep 8
done
[[ "$ssh_ready" == "1" ]] ||
  fail "SSH endpoint was allocated but did not become reachable."

ssh_config="$HOME/.ssh/config"
mkdir -p "$HOME/.ssh"
touch "$ssh_config"
chmod 600 "$ssh_config"

managed_begin="# BEGIN managed jacobian-lens RunPod"
managed_end="# END managed jacobian-lens RunPod"
ssh_config_tmp="$(mktemp)"

awk -v begin="$managed_begin" -v end="$managed_end" '
  $0 == begin { skipping = 1; next }
  $0 == end { skipping = 0; next }
  !skipping { print }
' "$ssh_config" >"$ssh_config_tmp"

cat >>"$ssh_config_tmp" <<EOF

$managed_begin
Host $SSH_ALIAS
  HostName $ssh_ip
  User root
  Port $ssh_port
  IdentityFile $ssh_key
  IdentitiesOnly yes
  ServerAliveInterval 30
  ServerAliveCountMax 4
  StrictHostKeyChecking accept-new
$managed_end
EOF

chmod 600 "$ssh_config_tmp"
mv "$ssh_config_tmp" "$ssh_config"

printf -v repo_url_q '%q' "$REPO_URL"
printf -v remote_repo_q '%q' "$REMOTE_REPO"
printf -v model_id_q '%q' "$MODEL_ID"
printf -v preload_model_q '%q' "$PRELOAD_MODEL"

remote_setup="
set -Eeuo pipefail
if [[ -e $remote_repo_q && ! -d $remote_repo_q/.git ]]; then
  echo 'Remote repository path exists but is not a Git checkout.' >&2
  exit 1
fi
if [[ ! -d $remote_repo_q/.git ]]; then
  git clone $repo_url_q $remote_repo_q
fi
cd $remote_repo_q
MODEL_ID=$model_id_q PRELOAD_MODEL=$preload_model_q \
  bash scripts/bootstrap_runpod.sh
"

printf -v remote_setup_q '%q' "$remote_setup"
ssh "${ssh_args[@]}" "root@$ssh_ip" "bash -lc $remote_setup_q"

cat <<EOF

RunPod workspace is ready.
Pod ID:    $pod_id
SSH:       ssh $SSH_ALIAS
VS Code:   code --remote ssh-remote+$SSH_ALIAS $REMOTE_REPO
Notebook:  $REMOTE_REPO/j_lens_multi_hop.ipynb
Kernel:    Python (jacobian-lens)

Do not stop this pod if you need guaranteed immediate access again.
When finished, preserve your notebook and delete the pod:
  $RUNPODCTL pod delete $pod_id
EOF

if [[ "$OPEN_VSCODE" == "1" ]]; then
  if command -v code >/dev/null; then
    code --remote "ssh-remote+$SSH_ALIAS" "$REMOTE_REPO" >/dev/null 2>&1 &
  else
    echo "VS Code CLI is unavailable; use the command printed above."
  fi
fi
