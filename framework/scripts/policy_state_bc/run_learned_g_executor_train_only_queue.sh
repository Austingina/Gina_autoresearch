#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
HORIZON_MODE="${HORIZON_MODE:?Set HORIZON_MODE to h1 or h16}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"

case "${HORIZON_MODE}" in
  h1)
    CHUNK_SIZE=1
    ACTION_EXECUTION_HORIZON=1
    STEM="2026-07-14_learned_g_executor_chunk1_exec1_e50_n100_h160_seed"
    ;;
  h16)
    CHUNK_SIZE=16
    ACTION_EXECUTION_HORIZON=16
    STEM="2026-07-14_learned_g_executor_chunk16_exec16_e50_n100_h160_seed"
    ;;
  *)
    echo "Unknown HORIZON_MODE=${HORIZON_MODE}; expected h1 or h16" >&2
    exit 2
    ;;
esac

cd "${ROOT}"
for seed in "${SEED_LIST[@]}"; do
  out_dir="${ROOT}/experiments/phase1_calvin/${STEM}${seed}"
  log_path="${out_dir}/train_only.log"
  mkdir -p "${out_dir}"
  complete=1
  for mode in chunk_bc_lang chunk_bc_minimal_g_concat chunk_bc_minimal_g_film; do
    if [[ ! -f "${out_dir}/${mode}/policy.pt" ]]; then
      complete=0
    fi
  done
  if [[ "${complete}" == "1" ]]; then
    echo "[train-only:${HORIZON_MODE}] skipping completed seed ${seed}"
    continue
  fi
  echo "[train-only:${HORIZON_MODE}] seed ${seed}" | tee -a "${log_path}"
  VARIANT=learned_executor_triplet \
  SEED="${seed}" \
  OUT_DIR="${out_dir}" \
  DEVICE="${DEVICE:-cuda:0}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  ACTION_EXECUTION_HORIZON="${ACTION_EXECUTION_HORIZON}" \
  FIRST_ACTION_LOSS_WEIGHT=1 \
  FIELD_ALIGNMENT_WEIGHT=0 \
  TEMPORAL_ENSEMBLE=0 \
  SKIP_EVAL=1 \
    bash framework/scripts/policy_state_bc/run_minimal_g_concat_structured_seed0.sh \
      >> "${log_path}" 2>&1
done
