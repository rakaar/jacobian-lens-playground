# Repository Agent Notes

## Vast.ai CLI

- Start Vast.ai documentation discovery with `https://docs.vast.ai/llms.txt`.
- The CLI is installed for this user as `~/.local/bin/vastai`.
- `VAST_AI_KEY` is defined in `~/.zshrc`. Use it to authenticate with
  `vastai set api-key "$VAST_AI_KEY"`.
- Never print, log, commit, or paste the value of `VAST_AI_KEY`.
- The public key `~/.ssh/vastai_ed25519.pub` is already registered in the
  Vast.ai console. Do not create or upload another key unless the user asks.
- Prefer direct SSH instances when appropriate. Get connection details with
  `vastai ssh-url INSTANCE_ID`.
- Poll `vastai show instance INSTANCE_ID` with a timeout. Treat `exited`,
  `unknown`, and `offline` as failures rather than waiting forever.
- Destroy completed instances with `vastai destroy instance INSTANCE_ID`.
  Stopping an instance ends compute billing, but disk storage charges continue.

## RunPod quick setup

- Check live inventory with `runpodctl gpu list -o json`; do not infer
  deployability from a pricing page.
- Launch an explicitly chosen GPU with
  `bash scripts/runpod_launch.sh "RUNPOD_GPU_ID"`.
- Set `OPEN_VSCODE=1` to open the configured `jlens-runpod` remote after setup.
- The launcher must not silently fall back to a different or more expensive
  GPU. It stores the created pod ID in `.runpod/pod-id`.
- `scripts/bootstrap_runpod.sh` is the idempotent remote environment setup. It
  creates `.venv`, installs the repository dependencies, registers the
  `jacobian-lens` Jupyter kernel, and optionally caches `Qwen/Qwen3-8B`.
- Stopping a RunPod pod releases its GPU. For short breaks, leave it running;
  for durable recreation, use a network volume or preserve work in Git before
  deleting the pod.
