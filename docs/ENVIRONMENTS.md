# Environment recipes

Pin dumps live under `shared/envs/`. Prefer creating envs from the notes below, then `pip install -r` / selective freeze install.

## `env_isaaclab` (Phase 0)

- **Python:** 3.11
- **Torch:** 2.7.0+cu128 (H200 verified)
- **Isaac Sim:** 5.1.0 · **Isaac Lab:** ~0.54.x
- **Conda name:** `env_isaaclab`
- **Setup script:** `scripts/remote_setup_env_isaaclab.sh`
- **Freeze:** `shared/envs/env_isaaclab.pip.freeze.txt`
- **Headless deps:** `libglu1-mesa`, `libgl1-mesa-glx`, `xvfb` (see [REMOTE_SETUP.md](REMOTE_SETUP.md))

```bash
conda activate env_isaaclab
export OMNI_KIT_ACCEPT_EULA=YES
source simulators/resolve_paths.sh
cd "$ISAACLAB_ROOT" && ./isaaclab.sh -p -c "import isaaclab; print('ok')"
```

## `calvin_h200` (Phase 1 training)

- **Python:** 3.8.20 in the audited H200 environment
- **Torch:** **2.3.1+cu121** with **sm_90** (required on H200)
- **Do not** train on the official `calvin` env (Torch 1.13.1+cu117) on H200
- **Freeze:** `shared/envs/calvin_h200.pip.freeze.txt`
- Editable installs (from freeze):
  - `calvin` @ `fa03f01` (`calvin_models`)
  - `calvin_env` @ `1431a46` (EGL build patched for headless)
  - `tacto` (submodule of calvin_env)

```bash
# After cloning calvin under $CALVIN_G_ROOT/code/calvin
conda create -p "$CALVIN_G_ROOT/envs/calvin_h200" python=3.8.20 -y
conda activate "$CALVIN_G_ROOT/envs/calvin_h200"
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
pip install -r shared/envs/calvin_h200.requirements.txt   # curated subset
# then follow official CALVIN install for calvin_models / calvin_env
```

Curated requirements (high-signal pins): see `shared/envs/calvin_h200.requirements.txt`.

## AutoResearch core (no simulator)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
# optional: pip install -e ".[maniskill]" ".[visual]"
```

## Sanity checks

```bash
python scripts/check_flywheel.py          # Phase 0 flywheel
framework/scripts/check_calvin_ready.sh   # Phase 1 paths + sm_90
nvidia-smi
```
