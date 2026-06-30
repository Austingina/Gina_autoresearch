#!/usr/bin/env bash
# One-shot remote setup: create env_isaaclab via IsaacLab and install AutoResearch deps.
set -euo pipefail

ISAACLAB_ROOT="${ISAACLAB_ROOT:-$HOME/code/IsaacLab}"
AUTORESEARCH_ROOT="${AUTORESEARCH_ROOT:-$HOME/AutoResearch}"
CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
LOG="${LOG:-$AUTORESEARCH_ROOT/runs/remote_env_setup.log}"

mkdir -p "$(dirname "${LOG}")"
exec > >(tee -a "${LOG}") 2>&1

echo "=== remote_setup_env_isaaclab $(date -Is) ==="
echo "  ISAACLAB_ROOT=${ISAACLAB_ROOT}"
echo "  AUTORESEARCH_ROOT=${AUTORESEARCH_ROOT}"
echo "  CONDA_ENV=${CONDA_ENV}"

export PATH="${HOME}/miniconda3/bin:${PATH}"
source "${HOME}/miniconda3/etc/profile.d/conda.sh"

cd "${ISAACLAB_ROOT}"
chmod +x isaaclab.sh

if ! conda env list | grep -qw "${CONDA_ENV}"; then
  echo "[1/5] Creating conda env ${CONDA_ENV}..."
  ./isaaclab.sh --conda "${CONDA_ENV}"
else
  echo "[1/5] Conda env ${CONDA_ENV} already exists."
fi

conda activate "${CONDA_ENV}"
export TERM="${TERM:-xterm-256color}"
export OMNI_KIT_ACCEPT_EULA=YES
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

if ! "${CONDA_PREFIX}/bin/python" -m pip --version >/dev/null 2>&1; then
  echo "[1b/5] Bootstrapping pip..."
  conda install -y pip
fi
if ! command -v cmake >/dev/null 2>&1; then
  echo "[1c/5] Installing cmake via conda..."
  conda install -y -c conda-forge cmake make ninja
fi

echo "[2/5] Pinning torch 2.7.0+cu128..."
pip install -U "setuptools<82.0.0"
pip install -U torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

if ! python -c "import isaacsim" >/dev/null 2>&1; then
  echo "[3/5] Installing Isaac Sim 5.1.0 pip packages (large download)..."
  pip install "isaacsim[all,extscache]==5.1.0.0" --extra-index-url https://pypi.nvidia.com
else
  echo "[3/5] Isaac Sim already importable."
fi

echo "[4/5] Installing IsaacLab extensions + rl_games (skip mimic/egl_probe)..."
cd "${ISAACLAB_ROOT}"
for ext in isaaclab isaaclab_assets isaaclab_contrib isaaclab_tasks isaaclab_rl; do
  pip install -e "${ISAACLAB_ROOT}/source/${ext}"
done
pip install -e "${ISAACLAB_ROOT}/source/isaaclab_rl[rl-games]"

echo "[5/5] Installing AutoResearch Python deps..."
cd "${AUTORESEARCH_ROOT}"
pip install -U pip
pip install -r requirements.txt

python - <<'PY'
import os
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
import torch
import isaacsim
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "devices", torch.cuda.device_count())
print("isaacsim", isaacsim.__file__)
PY

echo "[OK] env_isaaclab ready. Log: ${LOG}"
