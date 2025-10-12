bash -lc '
set -euo pipefail
export HF_HUB_ENABLE_HF_TRANSFER=1        # faster model pulls
export PIP_ROOT_USER_ACTION=ignore        # silence “pip as root” warning

# ── 1 · Update system packages and install nano ─────────────────────────────
apt-get update && apt-get install -y nano git ninja-build && apt install -y libnuma1 libnuma-dev

# ── 2 · Configure git with environment variables ────────────────────────────
[ -n "${GIT_USER_NAME:-}"  ]  && git config --global user.name  "$GIT_USER_NAME"
[ -n "${GIT_USER_EMAIL:-}" ]  && git config --global user.email "$GIT_USER_EMAIL"

# ── 3 · Bring your project up-to-date ───────────────────────────────────────
cd /workspace
if [ -d arc-diffusion/.git ]; then
    git -C arc-diffusion pull --ff-only
else
    git clone https://$GITHUB_PAT@github.com/TrelisResearch/arc-diffusion.git arc-diffusion
fi
cd arc-diffusion

# ── 4 · Install deps from pyproject.toml using uv ───────────────────────────
python -m pip install --upgrade --no-cache-dir uv
uv sync --all-groups

# ── 5 · Hand off to RunPod’s standard entrypoint ────────────────────────────
exec /start.sh
'
