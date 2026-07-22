#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
DECAY="${TEMPORAL_ENSEMBLE_DECAY:-0.25}"
SUFFIX="${EVAL_DIR_SUFFIX:-_ensemble_d025}"
STEM="2026-07-13_minimal_g_bounded_projected_rho2_e50_n100_h160_seed"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"

cd "${ROOT}"
for seed in "${SEED_LIST[@]}"; do
  out_dir="${ROOT}/experiments/phase1_calvin/${STEM}${seed}"
  log_path="${out_dir}/run${SUFFIX}.log"
  summary_path="${out_dir}/chunk_bc_minimal_g_bounded_projected/task_eval${SUFFIX}/state_bc_summary.json"
  if [[ -f "${summary_path}" ]]; then
    echo "[queue:${SUFFIX}] skipping completed seed ${seed}"
    continue
  fi
  echo "[queue:${SUFFIX}] evaluating seed ${seed}" | tee -a "${log_path}"
  VARIANT=bounded_projected_nolang \
  SEED="${seed}" \
  OUT_DIR="${out_dir}" \
  DEVICE="${DEVICE:-cuda:0}" \
  FIRST_ACTION_LOSS_WEIGHT=1 \
  ACTION_EXECUTION_HORIZON=1 \
  TEMPORAL_ENSEMBLE=1 \
  TEMPORAL_ENSEMBLE_DECAY="${DECAY}" \
  EVAL_DIR_SUFFIX="${SUFFIX}" \
    bash framework/scripts/policy_state_bc/run_minimal_g_concat_structured_seed0.sh \
      >> "${log_path}" 2>&1
done
