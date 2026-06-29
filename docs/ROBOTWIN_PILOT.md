# RoboTwin Open Microwave Pilot

RoboTwin is evaluated as an external teacher-demo backend for articulated-object tasks.
Keep it outside this repository and pin the external commit in notes/manifests.

## External Repo

- Repo: https://github.com/RoboTwin-Platform/RoboTwin
- Remote location: `~/RoboTwin`
- Initial pilot commit: `c3ddfa8`
- Task: `open_microwave`
- Official task doc: https://robotwin-platform.github.io/doc/tasks/open_microwave.html

RoboTwin documentation reports `open_microwave` data generation support for several
embodiments, including `franka-panda` with a lower success rate than `aloha-agilex`.

## Remote Pilot Config

Create a small smoke config in the RoboTwin repo, not in AutoResearch:

```bash
cd ~/RoboTwin
cp task_config/demo_randomized.yml task_config/autoresearch_franka_open_microwave.yml
```

Set the following differences from `demo_randomized.yml`:

```yaml
episode_num: 5
embodiment: [franka-panda]
language_num: 1
domain_randomization:
  random_background: false
  cluttered_table: false
  random_table_height: 0
  random_light: false
data_type:
  rgb: true
  depth: true
  endpose: true
  qpos: true
  actor_segmentation: true
```

The first goal is to get one successful HDF5 episode, not to collect a large dataset.

## Local Pilot Result

Local RoboTwin was deployed outside this repo:

```text
/home/l/code/RoboTwin
```

For the local smoke test, `env_isaaclab` was reused instead of creating a new RoboTwin
environment because it already contains `torch 2.7.0+cu128` for the RTX 5090.

Extra packages installed into that environment for the pilot:

```text
sapien==3.0.0b1
mplib==0.2.1
nvidia_curobo v0.7.8 checkout under /home/l/code/RoboTwin/envs/curobo
warp-lang==1.12.0
```

Notes:

- RoboTwin imports `open3d` unconditionally, but this pilot has pointcloud disabled.
  A local `/home/l/code/RoboTwin/open3d.py` stub was used to bypass the import.
- `assets/embodiments/franka-panda/curobo.yml` was copied to `curobo_left.yml` and
  `curobo_right.yml` because the collection path expects side-specific planner configs.
- Instead of downloading the full `objects.zip` locally, only the required
  `objects/044_microwave/`, `objects/objaverse/list.json`, and `objects/same.json`
  subtree was extracted from the remote zip and copied locally.

Successful smoke output:

```text
/home/l/code/RoboTwin/data/open_microwave/autoresearch_franka_open_microwave/data/episode0.hdf5
/home/l/code/RoboTwin/data/open_microwave/autoresearch_franka_open_microwave/video/episode0.mp4
/home/l/code/RoboTwin/data/open_microwave/autoresearch_franka_open_microwave/instructions/episode0.json
/home/l/code/RoboTwin/data/open_microwave/autoresearch_franka_open_microwave/scene_info.json
```

The generated HDF5 contains 483 frames with:

```text
observation/head_camera/{rgb, depth, actor_segmentation}
observation/left_camera/{rgb, depth, actor_segmentation}
observation/right_camera/{rgb, depth, actor_segmentation}
endpose/{left_endpose, left_gripper, right_endpose, right_gripper}
joint_action/{left_arm, left_gripper, right_arm, right_gripper, vector}
```

## Run Command

```bash
cd ~/RoboTwin
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
bash collect_data.sh open_microwave autoresearch_franka_open_microwave 0
```

RoboTwin docs warn that A/H/V series GPUs may be problematic for collection/evaluation.
The H200 server is still useful for policy training, but RoboTwin data generation must be
validated empirically.

## Expected RoboTwin Output

RoboTwin stores outputs under:

```text
~/RoboTwin/data/open_microwave/autoresearch_franka_open_microwave/
├── data/episode*.hdf5
├── seed.txt
├── scene_info.json
└── videos/
```

Use the local inspector before writing a hard-coded importer:

```bash
python scripts/inspect_robotwin_hdf5.py \
  /path/to/episode0.hdf5 --decode-images
```

## AutoResearch Mapping Target

Current field-learning code expects:

```text
data/<demo>/attrs["success"]
data/<demo>/obs/ee_mask
data/<demo>/obs/ee_depth
data/<demo>/obs/third_person_mask
data/<demo>/obs/third_person_depth
data/<demo>/obs/student_state
data/<demo>/goals/phase
```

RoboTwin HDF5 needs an adapter because it stores observations as camera groups and image
bitstreams. `student_state` and `phase` must be derived from RoboTwin `endpose`,
`joint_action`, scene metadata, and task-specific articulation state if available.
