#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
HORIZON_MODE="${HORIZON_MODE:?Set HORIZON_MODE to h1 or h16}"
EGL_DEVICE="${EGL_DEVICE:?Set EGL_DEVICE to the physical GPU index used for rendering}"
export EGL_VISIBLE_DEVICES="${EGL_DEVICE}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"
read -r -a MODE_LIST <<< "${MODES:-chunk_bc_lang chunk_bc_minimal_g_concat chunk_bc_minimal_g_film}"
MAX_RETRIES="${MAX_RETRIES:-3}"

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
  for mode in "${MODE_LIST[@]}"; do
    summary="${out_dir}/${mode}/task_eval/state_bc_summary.json"
    checkpoint="${out_dir}/${mode}/policy.pt"
    if [[ -f "${summary}" ]]; then
      echo "[eval:${HORIZON_MODE}] skip completed seed=${seed} mode=${mode}"
      continue
    fi
    if [[ ! -f "${checkpoint}" ]]; then
      echo "[eval:${HORIZON_MODE}] missing checkpoint seed=${seed} mode=${mode}: ${checkpoint}" >&2
      exit 3
    fi

    mode_log="${out_dir}/eval_${mode}.log"
    attempt=1
    while (( attempt <= MAX_RETRIES )); do
      echo "[eval:${HORIZON_MODE}] seed=${seed} mode=${mode} attempt=${attempt}/${MAX_RETRIES}" | tee -a "${mode_log}"
      if "${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
        --calvin-code "${CALVIN_CODE}" \
        --data-root "${DATA_ROOT}" \
        --checkpoint "${checkpoint}" \
        --run-meta "${out_dir}/run_meta.json" \
        --eval-dir "${out_dir}/${mode}/task_eval" \
        --num-sequences 100 \
        --ep-len 160 \
        --eval-scan-sequences "${EVAL_SCAN_SEQUENCES:-5000}" \
        --action-execution-horizon "${ACTION_EXECUTION_HORIZON}" \
        --device "${DEVICE:-cuda:0}" \
          >> "${mode_log}" 2>&1; then
        break
      else
        status=$?
      fi
      echo "[eval:${HORIZON_MODE}] failed seed=${seed} mode=${mode} status=${status}" | tee -a "${mode_log}" >&2
      if [[ -f "${summary}" ]]; then
        echo "[eval:${HORIZON_MODE}] summary completed before process exit; accepting seed=${seed} mode=${mode}"
        break
      fi
      attempt=$((attempt + 1))
      sleep 10
    done
    if [[ ! -f "${summary}" ]]; then
      echo "[eval:${HORIZON_MODE}] exhausted retries seed=${seed} mode=${mode}" >&2
      exit 4
    fi
  done
done
