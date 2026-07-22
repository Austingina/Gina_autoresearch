#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"
STEM="${STEM:-2026-07-14_minimal_g_semantic_h1_e50_n100_h160_seed}"
VARIANT="${VARIANT:-minimal_semantic_nolang}"
MODE_NAME="${MODE_NAME:-chunk_bc_minimal_g_semantic}"

cd "${ROOT}"
for seed in "${SEED_LIST[@]}"; do
  out_dir="${ROOT}/experiments/phase1_calvin/${STEM}${seed}"
  checkpoint="${out_dir}/${MODE_NAME}/policy.pt"
  log_path="${out_dir}/train_only.log"
  if [[ -f "${checkpoint}" ]]; then
    echo "[semantic-train] skip completed seed=${seed}"
    continue
  fi
  mkdir -p "${out_dir}"
  echo "[semantic-train] seed=${seed}" | tee -a "${log_path}"
  VARIANT="${VARIANT}" \
  SEED="${seed}" \
  OUT_DIR="${out_dir}" \
  DEVICE="${DEVICE:-cuda:0}" \
  CHUNK_SIZE=1 \
  ACTION_EXECUTION_HORIZON=1 \
  FIRST_ACTION_LOSS_WEIGHT=1 \
  FIELD_ALIGNMENT_WEIGHT="${FIELD_ALIGNMENT_WEIGHT:-0}" \
  SKIP_EVAL=1 \
    bash framework/scripts/policy_state_bc/run_minimal_g_concat_structured_seed0.sh \
      >> "${log_path}" 2>&1
done
