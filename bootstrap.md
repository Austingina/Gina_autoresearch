# Bootstrap AutoResearch from an empty machine

This is the canonical recovery procedure for the repository. It restores the
active CALVIN ABC-to-D visual experiment from public sources and records how to
regenerate the legacy Isaac Lab demonstrations. Read `dataset.md` first for
data provenance and checksums.

Last validated against the BNU V100 migration workspace on 2026-08-14.

## Recovery contract

Git contains the source code, simulator modifications, environment freezes,
oracle geometry, training/evaluation launchers, run metadata, and lightweight
metrics. Git intentionally does not contain 500+ GB archives, HDF5/NPZ data,
feature caches, videos, or `policy.pt` files.

Therefore there are two recovery levels:

1. **Scientific recovery:** download the recorded public inputs, regenerate
   derived data, retrain with the recorded seeds, and reproduce the same
   comparisons and qualitative behavior.
2. **Bit-identical checkpoint recovery:** additionally copy the two current
   `policy.pt` files before the old server expires. CUDA training is not
   guaranteed to regenerate byte-identical weights even with fixed seeds.

## 1. Machine and disk requirements

The active visual gate was validated on Linux with NVIDIA H200 and V100 GPUs.
The training stack is CUDA 12.1 / PyTorch 2.3.1 and must include the target
architecture (`sm_90` for H200 or `sm_70` for V100).

Recommended capacity:

| Resource | Minimum for current gate |
| --- | --- |
| Disk | 800 GB if `task_ABC_D.zip` is kept compressed and D is extracted; 1 TB+ recommended |
| GPU | one CUDA GPU for training; a second GPU is convenient for parallel evaluation |
| RAM | 64 GB recommended |
| Renderer | NVIDIA EGL preferred; llvmpipe is valid only for matched diagnostics |

Install system dependencies:

```bash
sudo apt-get update
sudo apt-get install -y \
  ffmpeg \
  git-lfs \
  libgl1-mesa-glx \
  libglu1-mesa \
  libosmesa6 \
  unzip \
  wget \
  xvfb
git lfs install
```

## 2. Clone the recoverable repository

```bash
mkdir -p /mnt/nvme33n1/wuk_code
cd /mnt/nvme33n1/wuk_code
git clone https://github.com/wukui-muc/AutoResearch.git
cd AutoResearch

export AUTORESEARCH_ROOT="$PWD"
export CALVIN_G_ROOT="/mnt/nvme33n1/wuk_code/calvin_g"
export CALVIN_CODE="$CALVIN_G_ROOT/code/calvin"
export CALVIN_DATA="$CALVIN_G_ROOT/data"
export CALVIN_ENV="$CALVIN_G_ROOT/envs/calvin_h200"
export CALVIN_PYTHON="$CALVIN_ENV/bin/python"
mkdir -p "$CALVIN_G_ROOT/code" "$CALVIN_DATA" "$CALVIN_G_ROOT/envs"
```

The absolute root may change, but keep all variables internally consistent.
The active launchers accept these variables and retain the audited H200 paths
as fallbacks.

## 3. Restore the pinned CALVIN source

```bash
git clone --recurse-submodules https://github.com/mees/calvin.git "$CALVIN_CODE"
git -C "$CALVIN_CODE" checkout fa03f01f19c65920e18cf37398a9ce859274af76
git -C "$CALVIN_CODE" submodule update --init --recursive

test "$(git -C "$CALVIN_CODE" rev-parse HEAD)" = \
  fa03f01f19c65920e18cf37398a9ce859274af76
test "$(git -C "$CALVIN_CODE/calvin_env" rev-parse HEAD)" = \
  1431a46bd36bde5903fb6345e68b5ccc30def666
```

Do not silently move CALVIN to a newer commit during reproduction. Create a
new experiment record if the simulator commit changes.

## 4. Restore the CALVIN H200 environment

Install Miniconda first if the machine does not provide conda. Then:

```bash
conda create -p "$CALVIN_ENV" python=3.8.20 -y
conda activate "$CALVIN_ENV"

python -m pip install --upgrade pip wheel "setuptools<58"
python -m pip install \
  torch==2.3.1+cu121 \
  torchvision==0.18.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r "$AUTORESEARCH_ROOT/shared/envs/calvin_h200.requirements.txt"

python -m pip install -e "$CALVIN_CODE/calvin_env/tacto"
python -m pip install -e "$CALVIN_CODE/calvin_env"
python -m pip install -e "$CALVIN_CODE/calvin_models"
```

The complete package snapshot is
`shared/envs/calvin_h200.pip.freeze.txt`. The curated requirements file is the
preferred rebuild input because a raw freeze may contain old absolute editable
paths.

Verify the important versions:

```bash
"$CALVIN_PYTHON" - <<'PY'
import numpy
import torch
import transformers

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("numpy", numpy.__version__)
print("transformers", transformers.__version__)
assert torch.cuda.is_available()
capability = torch.cuda.get_device_capability()
required_arch = f"sm_{capability[0]}{capability[1]}"
assert required_arch in torch.cuda.get_arch_list(), (required_arch, torch.cuda.get_arch_list())
PY
```

Expected core versions are Python `3.8.20`, PyTorch `2.3.1+cu121`, NumPy
`1.24.4`, Transformers `4.46.3`, and PyArrow `17.0.0`. On restricted networks,
the curated non-Torch packages can be installed from the Tsinghua PyPI mirror.

For a V100, feature caches use float16. Do not select bfloat16 merely because
PyTorch exposes the dtype; native BF16 requires Ampere or newer hardware.

Create compatibility links used by older experiment scripts:

```bash
cd "$AUTORESEARCH_ROOT"
mkdir -p phase1_calvin
ln -sfn "$CALVIN_G_ROOT" phase1_calvin/calvin_g
ln -sfn "$CALVIN_CODE" phase1_calvin/calvin_code
ln -sfn "$CALVIN_ENV" phase1_calvin/envs/calvin_h200
ln -sfn "$CALVIN_DATA" phase1_calvin/data
```

## 5. Download the required data

The exact URLs, checksums, repairs, and semantic split are documented in
`dataset.md`. The active frontier requires:

```bash
cd "$CALVIN_DATA"

# D data: extract for simulator configuration and held-out evaluation.
wget -c http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip
unzip task_D_D.zip
cd task_D_D
wget -c http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip
unzip -o task_D_D_scene_info.zip
cd ..
test -f task_D_D/training/lang_annotations/auto_lang_ann.npy
test -f task_D_D/validation/.hydra/merged_config.yaml
rm task_D_D.zip

# A/B/C source data: keep the full archive compressed.
wget -c http://calvin.cs.uni-freiburg.de/dataset/task_ABC_D.zip
echo "c2036c67eb4c06966af1d1e1665bdb572c69e1404f5e77ffd46b384ff2b79f74  task_ABC_D.zip" \
  | sha256sum --check

# Optional but useful for smoke tests.
wget -c http://calvin.cs.uni-freiburg.de/dataset/calvin_debug_dataset.zip
unzip calvin_debug_dataset.zip
```

The commands remove the D zip after basic extraction checks to keep peak disk
use below 800 GB. Keep `task_ABC_D.zip`: both compact extraction
and visual feature generation read directly from it.

For a short diagnostic allocation where the official archive cannot finish,
use the integrity-checked ModelScope mirror described in `dataset.md`. Keep its
results in a separately named experiment family and do not report them as
official ABC-to-D numbers.

The BNU behavior-G method gate uses the complete six-task diagnostic
conversion and its frozen dual-camera DINOv2 cache. After conversion and
feature extraction, build the packed numeric cache once:

```bash
"$CALVIN_PYTHON" \
  framework/scripts/data/pack_calvin_state_arrays.py \
  --data-root "$CALVIN_DATA/lerobot_blocks_behavior_g_semantic" \
  --workers 8
```

Validate `packed_state_manifest.json`: it must report `199292` frames and
`complete: true`. This avoids reopening nearly 200,000 NPZ files for every
training seed.

## 6. Regenerate the active derived datasets

### Fast interaction-volume method gate

Before rebuilding the full visual stack, the current method frontier can be
restored with about 1.7 GB of official ABC data:

```bash
"$CALVIN_PYTHON" -m pip install remotezip requests

"$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/extract_calvin_remote_calibration_subset.py" \
  --output-dir "$CALVIN_DATA/task_ABC_D_volume_g_subset"

export CALVIN_VOLUME_G_SUBSET_ROOT="$CALVIN_DATA/task_ABC_D_volume_g_subset"
export CALVIN_SIM_DATA_ROOT="$CALVIN_DATA/calvin_debug_dataset"
bash "$AUTORESEARCH_ROOT/framework/scripts/run_stage3_interaction_volume_matched_gate.sh" 0 0
```

The launcher performs source-only calibration and matched G relabeling when
their outputs are absent. It then trains the proprio-only no-G baseline and the
proprio+volume-G token policies from scratch, evaluates source combinations in
D, and evaluates held-out recombinations with normal/zero/sign-flipped G. No
scene state, direct language, or task one-hot reaches either executor.

On a two-GPU allocation, first finish training and offline audits without
starting the simulator, then launch the two disjoint online families:

```bash
VOLUME_GATE_EVAL_FAMILY=none \
  bash "$AUTORESEARCH_ROOT/framework/scripts/run_stage3_interaction_volume_matched_gate.sh" 0 0

VOLUME_GATE_EVAL_FAMILY=core \
  bash "$AUTORESEARCH_ROOT/framework/scripts/run_stage3_interaction_volume_matched_gate.sh" 0 0 &
VOLUME_GATE_EVAL_FAMILY=factorized \
  bash "$AUTORESEARCH_ROOT/framework/scripts/run_stage3_interaction_volume_matched_gate.sh" 0 1 &
wait
```

The `core` shard contains no-G and unrestricted token-G; `factorized` contains
the translational-G-only executor. Their task counts, initial-state protocol,
and checkpoints are identical to an unsplit `all` run. Use the full visual
reconstruction below only after this representation gate passes.

The launcher also supports an objective-alignment diagnostic without changing
the G labels. `token` skips repeated no-G/factorized rollouts, while the run
label keeps checkpoints and progress files isolated:

```bash
VOLUME_GATE_RUN_LABEL=M3_objective_h1 \
VOLUME_GATE_EVAL_FAMILY=token \
VOLUME_GATE_INCLUDE_FACTORIZED=0 \
VOLUME_GATE_CHUNK_SIZE=1 \
VOLUME_GATE_FIRST_ACTION_LOSS_WEIGHT=1 \
  bash "$AUTORESEARCH_ROOT/framework/scripts/run_stage3_interaction_volume_matched_gate.sh" 0 0

VOLUME_GATE_RUN_LABEL=M3_objective_h16_w16 \
VOLUME_GATE_EVAL_FAMILY=token \
VOLUME_GATE_INCLUDE_FACTORIZED=0 \
VOLUME_GATE_CHUNK_SIZE=16 \
VOLUME_GATE_FIRST_ACTION_LOSS_WEIGHT=16 \
  bash "$AUTORESEARCH_ROOT/framework/scripts/run_stage3_interaction_volume_matched_gate.sh" 0 1
```

Both variants replan after one action by default. Set
`VOLUME_GATE_ACTION_EXECUTION_HORIZON` only for a separate execution-horizon
ablation; it must not exceed the trained chunk size.

### Full visual ABC-to-D stack

Build the 2,101-segment state/action/language subset:

```bash
"$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/build_calvin_compact_state_dataset_from_zip.py" \
  --zip "$CALVIN_DATA/task_ABC_D.zip" \
  --out-dir "$CALVIN_DATA/task_ABC_D_compact_blocks" \
  --include-tasks \
    push_red_block_left \
    push_blue_block_right \
    push_pink_block_left \
    push_pink_block_right
```

Pre-download the exact DINOv2 snapshot used by the existing experiments:

```bash
"$CALVIN_PYTHON" - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="facebook/dinov2-small",
    revision="ed25f3a31f01632728cabb09d1542f84ab7b0056",
)
PY
```

Generate static-camera and gripper-camera features:

```bash
CUDA_VISIBLE_DEVICES=0 "$CALVIN_PYTHON" \
  "$AUTORESEARCH_ROOT/framework/scripts/data/precompute_calvin_dinov2_features_from_zip.py" \
  --source-zip "$CALVIN_DATA/task_ABC_D.zip" \
  --compact-data-root "$CALVIN_DATA/task_ABC_D_compact_blocks" \
  --output-dir "$CALVIN_DATA/task_ABC_D_compact_blocks_dinov2_small" \
  --model-name facebook/dinov2-small \
  --batch-size 128 \
  --shard-size 2048 \
  --device cuda:0
```

Run the manifest checks at the end of `dataset.md`, then:

```bash
cd "$AUTORESEARCH_ROOT"
source simulators/resolve_paths.sh
framework/scripts/check_calvin_ready.sh
```

## 7. Retrain the current visual policies

The active policy is chunked deterministic BC with H16 targets, H1 closed-loop
replanning, a frozen dual-camera DINOv2 encoder, and compact
`G=[attraction(3), object_flow(3)]` tokens. The source combinations are fixed
inside the launcher.

Train visual plus oracle G, without direct executor language:

```bash
cd "$AUTORESEARCH_ROOT"
bash framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_train.sh \
  0 0 visual with_g none
```

Train the matched visual plus official MiniLM language-only baseline:

```bash
bash framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_train.sh \
  0 1 visual language_only minilm
```

Expected checkpoints:

```text
experiments/phase1_calvin/2026-07-19_M1_abc_to_d_oracle_g_none_visual_seed0/with_g/chunk_bc_minimal_g_tokens/policy.pt
experiments/phase1_calvin/2026-07-19_M1_abc_to_d_oracle_g_minilm_visual_seed0/language_only/chunk_bc_minimal_g_tokens/policy.pt
```

The original checkpoints were approximately 69 MB and 70 MB. Their SHA-256
values are recorded so an out-of-band backup can be verified:

```text
5793e105c36f0e8558d7f5506d72aa9e350ca6b110ef323b54f74723753aa830  visual+oracle-G policy.pt
3bb32242fb8ba522a2293560e97e46f3e7d018f3b718bec8545c3e59e5a33e89  visual+MiniLM/no-G policy.pt
```

The original offline validation targets were:

| Policy | Chunk MSE | First-action MSE |
| --- | ---: | ---: |
| visual + oracle G | 0.04217 | 0.02312 |
| visual + MiniLM, no G | 0.05548 | 0.02734 |

Large deviation here usually means the compact manifest, feature cache,
normalization, or dependency versions do not match.

## 8. Run the matched D scene gate

CALVIN rendering has one critical device rule on a multi-GPU server:

```text
Do not remap CUDA_VISIBLE_DEVICES and then pass a physical EGL id.
Leave CUDA_VISIBLE_DEVICES unset, use policy device cuda:N, and set
EGL_VISIBLE_DEVICES to the matching physical NVIDIA device N.
```

The evaluator launcher implements this rule and removes the unused tactile
camera that previously caused a PyBullet render abort.

```bash
cd "$AUTORESEARCH_ROOT"

# Oracle G and its paired causal controls, all on identical D starts.
EGL_RENDER_DEVICE=0 bash framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval.sh \
  0 0 visual with_g normal normal none nvidia_egl
EGL_RENDER_DEVICE=0 bash framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval.sh \
  0 0 visual with_g zero normal none nvidia_egl
EGL_RENDER_DEVICE=0 bash framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval.sh \
  0 0 visual with_g sign_flip normal none nvidia_egl

# Matched direct-language/no-G baseline.
EGL_RENDER_DEVICE=1 bash framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval.sh \
  0 1 visual language_only zero normal minilm nvidia_egl
```

Each row evaluates 100 identical initial states for each held-out task,
`push_blue_block_left` and `push_red_block_right`, with 160 simulator steps and
H1 replanning.

The corrected-NVIDIA recovery targets already committed under
`experiments/phase1_calvin/` are:

| Executor | Blue left | Red right | Total |
| --- | ---: | ---: | ---: |
| visual + oracle G | 14/100 | 18/100 | 32/200 |
| visual + zero G | 1/100 | 0/100 | 1/200 |
| visual + sign-flipped G | 1/100 | 0/100 | 1/200 |
| visual + MiniLM, no G | 4/100 | 2/100 | 6/200 |

Use the exact per-rollout arrays in each `state_bc_summary.json` for paired
tests. Do not compare unmatched random starts.

## 9. Reproduce source-scene health

The source-support diagnostic uses first states from the compact A/B/C
demonstrations. Scene A currently has validated reset fidelity; B/C require
additional configuration recovery and must not be mixed into the accepted row.

```bash
unset CUDA_VISIBLE_DEVICES
export EGL_VISIBLE_DEVICES=0

"$CALVIN_PYTHON" \
  framework/scripts/policy_state_bc/eval_calvin_abc_source_scene_health.py \
  --calvin-code "$CALVIN_CODE" \
  --source-data-root "$CALVIN_DATA/task_ABC_D_compact_blocks" \
  --source-archive "$CALVIN_DATA/task_ABC_D.zip" \
  --eval-data-root "$CALVIN_DATA/task_D_D" \
  --checkpoint experiments/phase1_calvin/2026-07-19_M1_abc_to_d_oracle_g_none_visual_seed0/with_g/chunk_bc_minimal_g_tokens/policy.pt \
  --run-meta experiments/phase1_calvin/2026-07-19_M1_abc_to_d_oracle_g_none_visual_seed0/with_g/run_meta.json \
  --out-dir experiments/phase1_calvin/2026-07-21_M2_visual_source_scene_A_health/with_g \
  --scenes A \
  --num-per-task-scene 25 \
  --ep-len 160 \
  --device cuda:0 \
  --audit-reset-visuals
```

Repeat with the MiniLM/no-G checkpoint and corresponding `run_meta.json`.
Accepted Scene-A totals are 72/100 for visual+G and 62/100 for
visual+MiniLM/no-G. Median reset-frame DINO cosine was 0.99981 for the static
camera and 0.99961 for the gripper camera.

## 10. Restore Phase-0 Isaac data only when needed

Isaac Lab source and project modifications are vendored at
`simulators/isaaclab/`. The simulator packages and demonstrations are not.

```bash
cd "$AUTORESEARCH_ROOT"
export ISAACLAB_ROOT="$AUTORESEARCH_ROOT/simulators/isaaclab"
export OMNI_KIT_ACCEPT_EULA=YES
bash scripts/remote_setup_env_isaaclab.sh

STAGE=s1_random_cube \
RANDOMIZE_CUBE=1 \
CLEAN_TEACHER=1 \
CAMERA_SIZE=240 \
TOPDOWN_CLOSE=1 \
NUM_DEMOS=100 \
bash scripts/collect_lift_demos.sh
```

Use `data/manifests/` and `docs/progress_phase0_isaac.md` to select a historical
collection setting. Phase-0 data is not needed for the active CALVIN gate.

## 11. Resume the research state

Read in this order:

1. `PROGRESS.md` for the project-level frontier.
2. `docs/progress_phase1_calvin.md` for the chronological claim ladder.
3. `knowledge_base/03_experiment_evidence/phase1_calvin_abc_d_scene_gate_visual_transition_2026-07-19.md` for the current scene gate.
4. `experiments/phase1_calvin/**/run_meta.json` and
   `state_bc_summary.json` for machine-readable settings and outcomes.

The next methodological step is not a larger data download. The current gate
shows healthy source-scene execution but weak A-to-D visual transfer. Before
scaling learned G, improve the visual policy's scene/composition generalization
while preserving the normal/zero/sign-flip causal comparison.

## 12. What to back up before a server expires

Git push is sufficient for source, metadata, and lightweight evidence. For
exact checkpoint replay, additionally archive:

```text
...oracle_g_none_visual_seed0/.../policy.pt
...oracle_g_minilm_visual_seed0/.../policy.pt
```

The 518 GB CALVIN archive and 77 MB DINO cache can be regenerated from the
recorded public inputs. Logs and videos are useful for diagnosis but are not
required to reconstruct the numerical gate.
