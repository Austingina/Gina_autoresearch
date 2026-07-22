#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
EGL_DEVICE="${EGL_DEVICE:?Set EGL_DEVICE to the physical rendering GPU index}"
export EGL_VISIBLE_DEVICES="${EGL_DEVICE}"
read -r -a SEED_LIST <<< "${SEEDS:-3 4}"
read -r -a CONTROL_LIST <<< "${QUERY_G_CONTROLS:-zero sign_flip task:open_drawer zero_attraction zero_flow sign_flip_attraction}"
MAX_RETRIES="${MAX_RETRIES:-50}"
WAIT_FOR_PID="${WAIT_FOR_PID:-}"

TRAIN_STEM="2026-07-14_learned_g_executor_chunk1_exec1_e50_n100_h160_seed"
OUT_STEM="2026-07-14_learned_g_executor_h1_film_causal_n100_h160_seed"

if [[ -n "${WAIT_FOR_PID}" ]]; then
  echo "[causal] waiting for PID ${WAIT_FOR_PID}"
  while kill -0 "${WAIT_FOR_PID}" 2>/dev/null; do
    sleep 30
  done
fi

cd "${ROOT}"
for seed in "${SEED_LIST[@]}"; do
  train_dir="${ROOT}/experiments/phase1_calvin/${TRAIN_STEM}${seed}"
  seed_out_dir="${ROOT}/experiments/phase1_calvin/${OUT_STEM}${seed}"
  mkdir -p "${seed_out_dir}"
  checkpoint="${train_dir}/chunk_bc_minimal_g_film/policy.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing H1 FiLM checkpoint for seed ${seed}: ${checkpoint}" >&2
    exit 3
  fi

  for control in "${CONTROL_LIST[@]}"; do
    control_label="${control//:/_}"
    eval_dir="${seed_out_dir}/${control_label}"
    summary="${eval_dir}/state_bc_summary.json"
    log_path="${seed_out_dir}/eval_${control_label}.log"
    if [[ -f "${summary}" ]]; then
      echo "[causal] skip completed seed=${seed} control=${control}"
      continue
    fi

    attempt=1
    while (( attempt <= MAX_RETRIES )); do
      echo "[causal] seed=${seed} control=${control} attempt=${attempt}/${MAX_RETRIES}" | tee -a "${log_path}"
      if "${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
        --calvin-code "${CALVIN_CODE}" \
        --data-root "${DATA_ROOT}" \
        --checkpoint "${checkpoint}" \
        --run-meta "${train_dir}/run_meta.json" \
        --eval-dir "${eval_dir}" \
        --num-sequences 100 \
        --ep-len 160 \
        --eval-scan-sequences "${EVAL_SCAN_SEQUENCES:-5000}" \
        --action-execution-horizon 1 \
        --g-scale 1 \
        --query-g-control "${control}" \
        --language-control normal \
        --device "${DEVICE:-cuda:0}" \
          >> "${log_path}" 2>&1; then
        break
      else
        status=$?
      fi
      echo "[causal] failed seed=${seed} control=${control} status=${status}" | tee -a "${log_path}" >&2
      if [[ -f "${summary}" ]]; then
        echo "[causal] summary completed before process exit; accepting seed=${seed} control=${control}"
        break
      fi
      attempt=$((attempt + 1))
      sleep 10
    done
    if [[ ! -f "${summary}" ]]; then
      echo "[causal] exhausted retries seed=${seed} control=${control}" >&2
      exit 4
    fi
  done
done
