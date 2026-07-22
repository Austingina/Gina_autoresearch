#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
ACTION_EXECUTION_HORIZON="${ACTION_EXECUTION_HORIZON:?Set ACTION_EXECUTION_HORIZON, e.g. 4}"
STEM="2026-07-13_minimal_g_bounded_projected_rho2_e50_n100_h160_seed"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"

cd "${ROOT}"
for seed in "${SEED_LIST[@]}"; do
  out_dir="${ROOT}/experiments/phase1_calvin/${STEM}${seed}"
  log_path="${out_dir}/run_execute${ACTION_EXECUTION_HORIZON}.log"
  summary_path="${out_dir}/chunk_bc_minimal_g_bounded_projected/task_eval_execute${ACTION_EXECUTION_HORIZON}/state_bc_summary.json"
  if [[ -f "${summary_path}" ]]; then
    echo "[queue:execute${ACTION_EXECUTION_HORIZON}] skipping completed seed ${seed}"
    continue
  fi
  echo "[queue:execute${ACTION_EXECUTION_HORIZON}] evaluating seed ${seed}" | tee -a "${log_path}"
  VARIANT=bounded_projected_nolang \
  SEED="${seed}" \
  OUT_DIR="${out_dir}" \
  DEVICE="${DEVICE:-cuda:0}" \
  FIRST_ACTION_LOSS_WEIGHT=1 \
  ACTION_EXECUTION_HORIZON="${ACTION_EXECUTION_HORIZON}" \
  EVAL_DIR_SUFFIX="_execute${ACTION_EXECUTION_HORIZON}" \
    bash framework/scripts/policy_state_bc/run_minimal_g_concat_structured_seed0.sh \
      >> "${log_path}" 2>&1
done
