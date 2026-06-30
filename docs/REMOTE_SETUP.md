# Remote H200 Setup

This project is designed to keep code in Git and keep large artifacts out of Git.

- GitHub: https://github.com/wukui-muc/AutoResearch.git
- Remote server: `wukui@34.52.175.145`
- Large local artifacts: `data/`, `runs/`, `checkpoints/`, `outputs/`

## 1. Clone and Install

```bash
ssh wukui@34.52.175.145
git clone https://github.com/wukui-muc/AutoResearch.git
cd AutoResearch

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For ManiSkill smoke tests:

```bash
pip install -e ".[maniskill]"
```

For local visualization:

```bash
pip install -e ".[visual]"
```

## 2. External Simulator Dependencies

IsaacLab and Arena packages are external dependencies and are not installed by this repository.
Set these paths on the remote if those repos are available there:

```bash
export ISAACLAB_ROOT=$HOME/code/IsaacLab
export HIEVT_ROOT=/path/to/Hierarchical-Instruction-aware-Embodied-Visual-Tracking-master
export EVT_ROOT=/path/to/Offline_RL_Active_Tracking
export DUALGF_ROOT=/path/to/DualGF
export TARGF_ROOT=/path/to/TarGF
export AUTORESEARCH_ROOT=$PWD
export OMNI_KIT_ACCEPT_EULA=YES
```

The current local defaults are documented in `configs/default.yaml`.

### IsaacLab on H200 (env_isaaclab)

IsaacLab code lives at `~/code/IsaacLab` (rsync from local; excludes `runs/`, `datasets/`).

Create the conda env and install Isaac Sim 5.1 + extensions:

```bash
bash scripts/remote_setup_env_isaaclab.sh
# log: runs/remote_env_setup.log
```

Verified remote stack: Python 3.11, torch 2.7.0+cu128, isaacsim 5.1.0.0, isaaclab 0.54.4.

**Headless rendering prerequisites** (mask+depth collect + B7 need RTX cameras):

```bash
sudo apt-get install -y libglu1-mesa libgl1-mesa-glx xvfb
```

Without `libGLU.so.1`, Isaac Sim headless RTX may crash in `libcarb.glinterop.plugin.so`.
GLFW also needs a virtual display — use `xvfb-run`:

```bash
export CUDA_VISIBLE_DEVICES=3   # mask to a free GPU; activeGpu=0 inside Isaac
export OMNI_KIT_ACCEPT_EULA=YES
xvfb-run -a bash scripts/remote_smoke_b7.sh
```

Do **not** pass `--device cuda:3` while `CUDA_VISIBLE_DEVICES` is unset (activeGpu>0 can crash on this host).

G-only training (no sim) works once demos exist under `data/lift/s1_random_cube/demos/`.

## 3. Initialize Project Storage

```bash
./scripts/init_workspace.sh
python scripts/check_flywheel.py
```

Only small provenance files under `data/manifests/` are tracked by Git. Demos, checkpoints,
logs, videos, and tensorboard events should stay on local or remote disks.

## 4. Transfer Artifacts

Use `rsync` for selected datasets/checkpoints instead of Git:

```bash
rsync -av --progress data/manifests/ wukui@34.52.175.145:~/AutoResearch/data/manifests/
rsync -av --progress data/lift/s1_random_cube/demos_clean240/ \
  wukui@34.52.175.145:~/AutoResearch/data/lift/s1_random_cube/demos_clean240/
rsync -av --progress checkpoints/lift/ \
  wukui@34.52.175.145:~/AutoResearch/checkpoints/lift/
```

Sync back only lightweight visual summaries when reviewing results locally:

```bash
rsync -av --include='*/' --include='*.json' --include='*.csv' --include='*.png' \
  --include='*.mp4' --exclude='*' \
  wukui@34.52.175.145:~/AutoResearch/runs/ runs/remote/
```

## 5. Remote Sanity Checks

Before launching multi-GPU work:

```bash
nvidia-smi
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

Start with one GPU and a short smoke run before using all 8 H200s. If GPU utilization is low,
check dataloader workers, storage throughput, and whether datasets are on local NVMe.
