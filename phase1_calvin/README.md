# Phase 1: CALVIN Dual-Camera

CALVIN reconstruction of the same research question: compact interaction G
(`G=[A(3),F(3)]`) with dual cameras (`rgb_static` + `rgb_gripper`).

## Links (create after install)

```text
phase1_calvin/
  calvin_g/      → $CALVIN_G_ROOT          # code + data + envs
  calvin_code/   → $CALVIN_G_ROOT/code/calvin
  envs/calvin_h200 → $CALVIN_G_ROOT/envs/calvin_h200
  data/          → $CALVIN_G_ROOT/data
```

H200 default: `CALVIN_G_ROOT=/mnt/nvme33n1/wuk_code/calvin_g`.

## Entry points

```bash
source simulators/resolve_paths.sh
framework/scripts/check_calvin_ready.sh
GPU=0 framework/scripts/run_calvin_dualcam.sh smoke
```

Trainers / gates: `framework/scripts/`  
Protocol docs: `framework/docs/`, `docs/phase1_calvin/`  
Journal: [docs/progress_phase1_calvin.md](../docs/progress_phase1_calvin.md)  
Bootstrap: [docs/BOOTSTRAP.md](../docs/BOOTSTRAP.md)
