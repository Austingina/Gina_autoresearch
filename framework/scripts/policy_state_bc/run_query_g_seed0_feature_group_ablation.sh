#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
POLICY="${POLICY:-${ROOT}/experiments/phase1_calvin/2026-07-10_scope_primitive_zero_shot_close_paired_e50_n100_h160_seed0/chunk_bc_lang_query_g/policy.pt}"
OUT_DIR="${OUT_DIR:-${ROOT}/experiments/phase1_calvin/2026-07-11_query_g_seed0_feature_group_ablation_language_zero_n100}"

cd "${ROOT}"

"${PYTHON}" framework/scripts/policy_state_bc/eval_chunk_flow_checkpoints.py \
  --calvin-code "${CALVIN_CODE}" \
  --data-root "${DATA_ROOT}" \
  --out-dir "${OUT_DIR}" \
  --policies "seed0_lang_query_g=${POLICY}" \
  --eval-task-filter close_drawer \
  --num-sequences 100 \
  --ep-len 160 \
  --eval-scan-sequences 50000 \
  --eval-sequence-offset 0 \
  --query-g-controls \
    normal \
    mask:token_type \
    mask:rel_eef \
    mask:rel_part \
    mask:field_vec \
    mask:desired_flow \
    mask:dist_norm \
    mask:residual_norm \
    mask:contact_conf \
  --language-controls zero \
  --device "${DEVICE:-cuda:0}"
