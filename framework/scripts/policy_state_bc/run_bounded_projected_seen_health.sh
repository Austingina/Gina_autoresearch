#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
SEED="${SEED:-0}"
EXPERIMENT="${EXPERIMENT:-${ROOT}/experiments/phase1_calvin/2026-07-13_minimal_g_bounded_projected_rho2_e50_n100_h160_seed${SEED}}"
POLICY="${POLICY:-${EXPERIMENT}/chunk_bc_minimal_g_bounded_projected/policy.pt}"
OUT_DIR="${OUT_DIR:-${EXPERIMENT}/axis_diag_seen_open_slider_n25}"

mkdir -p "${OUT_DIR}"
cd "${ROOT}"

"${PYTHON}" framework/scripts/policy_state_bc/diagnose_chunk_flow_axis_contact.py \
  --calvin-code phase1_calvin/calvin_code \
  --data-root phase1_calvin/data/task_D_D \
  --articulation-geometry-json \
    experiments/phase1_calvin/2026-07-04_oracle_g_v2_articulation_geometry_stability/oracle_g_v2_articulation_geometry_full.json \
  --out-dir "${OUT_DIR}" \
  --policies "seed${SEED}=${POLICY}" \
  --tasks open_drawer move_slider_left move_slider_right \
  --num-sequences 25 \
  --scan-sequences 50000 \
  --ep-len 160 \
  --device "${DEVICE:-cuda:0}"
