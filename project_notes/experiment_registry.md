# Experiment Registry

## Phase 0: IsaacLab / AutoResearch

Important existing locations:

- Code and records: `phase0_isaac/autoresearch`
- Historical runs: `phase0_isaac/runs`
- Checkpoints: `phase0_isaac/checkpoints`
- Figures: `phase0_isaac/autoresearch/outputs/figures`
- Docs: `phase0_isaac/docs`

Known run folders at migration time:

- `s1_B1_visible_ee_forward_b1_linearbeta_gpu2_20260630_072911`
- `s1_B1_visible_ee_forward_b1_linearbeta_gpu2_20260630_073047`
- `debug_b1_cv2_nvicd_20260630_074355`
- `debug_b1_direct_cuda2_20260630_073541`
- `debug_b1_direct_cuda2_nvicd_20260630_074221`

## Phase 1: CALVIN Dual Camera

Important existing locations:

- Workspace: `phase1_calvin/calvin_g`
- Code: `phase1_calvin/calvin_code`
- Data: `phase1_calvin/data`
- Envs: `phase1_calvin/envs`
- Logs: `phase1_calvin/logs`

Verified smoke logs:

- `phase1_calvin/logs/runs/2026-07-02/05-06-49/training.log`
- `phase1_calvin/logs/runs/2026-07-02/05-19-10/training.log`
- `phase1_calvin/logs/runs/2026-07-02/05-20-26/training.log`
- `phase1_calvin/logs/runs/2026-07-02/05-23-05/training.log`

## New Run Policy

New records should be created under:

- `experiments/phase0_isaac/`
- `experiments/phase1_calvin/`

Do not rely only on framework-generated log folders. Each run needs a human-readable `notes.md` and exact `command.sh`.

## Canonical CALVIN Framework Runs

- `experiments/phase1_calvin/2026-07-02_064653_calvin_dualcam_smoke`: framework smoke, completed.
- `experiments/phase1_calvin/2026-07-02_064810_calvin_dualcam_debug_train`: framework debug train, completed.
