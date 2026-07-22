# Remote H200 Setup

**Recovery after a wipe:** start with [BOOTSTRAP.md](BOOTSTRAP.md), then this page.

- GitHub: https://github.com/wukui-muc/AutoResearch.git
- SSH: `unrealzoo-remote` → `wukui@34.52.175.145`
- Canonical H200 root: `/mnt/nvme33n1/wuk_code/AutoResearch`
- CALVIN workspace: `/mnt/nvme33n1/wuk_code/calvin_g`
- Large artifacts: `data/`, `runs/`, `checkpoints/`, `experiments/**/*.pt`, CALVIN zips

Path map: [HOSTS.md](HOSTS.md) · Env pins: [ENVIRONMENTS.md](ENVIRONMENTS.md).

## 1. Clone and Install

```bash
ssh unrealzoo-remote
git clone https://github.com/wukui-muc/AutoResearch.git /mnt/nvme33n1/wuk_code/AutoResearch
cd /mnt/nvme33n1/wuk_code/AutoResearch
export AUTORESEARCH_ROOT="$PWD"
source simulators/resolve_paths.sh

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Optional:

```bash
pip install -e ".[maniskill]"
pip install -e ".[visual]"
```

## 2. Simulator backends

### Isaac Lab

Isaac Lab sources ship inside this repo at `simulators/isaaclab/`.
Install Isaac Sim into conda (not into Git):

```bash
bash scripts/remote_setup_env_isaaclab.sh
```

Or point at another tree:

```bash
export ISAACLAB_ROOT=$HOME/code/IsaacLab
```

Verified remote stack: Python 3.11, torch 2.7.0+cu128, isaacsim 5.1, isaaclab 0.54.x.
Freeze: `shared/envs/env_isaaclab.pip.freeze.txt`.

**Headless rendering:**

```bash
sudo apt-get install -y libglu1-mesa libgl1-mesa-glx xvfb
export CUDA_VISIBLE_DEVICES=3
export OMNI_KIT_ACCEPT_EULA=YES
xvfb-run -a bash scripts/remote_smoke_b7.sh
```

### CALVIN

```bash
export CALVIN_G_ROOT=/mnt/nvme33n1/wuk_code/calvin_g
ln -sfn "$CALVIN_G_ROOT" phase1_calvin/calvin_g
ln -sfn "$CALVIN_G_ROOT/code/calvin" phase1_calvin/calvin_code
mkdir -p phase1_calvin/envs phase1_calvin/data
ln -sfn "$CALVIN_G_ROOT/envs/calvin_h200" phase1_calvin/envs/calvin_h200
ln -sfn "$CALVIN_G_ROOT/data" phase1_calvin/data

framework/scripts/check_calvin_ready.sh
GPU=2 framework/scripts/run_calvin_dualcam.sh smoke
```

Use **`calvin_h200`** (torch 2.3.1+cu121, sm_90), not the official cu117 env for training.

## 3. Initialize project storage

```bash
./scripts/init_workspace.sh
python scripts/check_flywheel.py
```

Only small provenance under `data/manifests/` is tracked. Demos, checkpoints, and videos stay on disk.

## 4. Transfer artifacts

```bash
rsync -av --progress data/manifests/ unrealzoo-remote:/mnt/nvme33n1/wuk_code/AutoResearch/data/manifests/
rsync -av --progress data/lift/s1_random_cube/demos_clean240/ \
  unrealzoo-remote:/mnt/nvme33n1/wuk_code/AutoResearch/data/lift/s1_random_cube/demos_clean240/

# Pull lightweight summaries back
rsync -av --include='*/' --include='*.json' --include='*.csv' --include='*.png' \
  --include='*.md' --include='*.mp4' --exclude='*' \
  unrealzoo-remote:/mnt/nvme33n1/wuk_code/AutoResearch/experiments/ experiments/remote/
```

## 5. Sanity checks

```bash
nvidia-smi
source simulators/resolve_paths.sh
echo "ISAACLAB_ROOT=$ISAACLAB_ROOT"
echo "CALVIN_ENV=$CALVIN_ENV"
```
