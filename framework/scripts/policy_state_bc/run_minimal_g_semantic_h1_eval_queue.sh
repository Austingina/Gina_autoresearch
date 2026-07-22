#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
EGL_DEVICE="${EGL_DEVICE:?Set EGL_DEVICE to the physical rendering GPU index}"
export EGL_VISIBLE_DEVICES="${EGL_DEVICE}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"
read -r -a CONTROL_LIST <<< "${QUERY_G_CONTROLS:-normal zero sign_flip task:open_drawer}"
read -r -a WAIT_PIDS <<< "${WAIT_FOR_PIDS:-}"
MAX_RETRIES="${MAX_RETRIES:-50}"
RUN_SEEN_TASKS="${RUN_SEEN_TASKS:-1}"
STEM="${STEM:-2026-07-14_minimal_g_semantic_h1_e50_n100_h160_seed}"
MODE_NAME="${MODE_NAME:-chunk_bc_minimal_g_semantic}"

for pid in "${WAIT_PIDS[@]}"; do
  [[ -z "${pid}" ]] && continue
  echo "[semantic-eval] waiting for PID ${pid}"
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 30
  done
done

cd "${ROOT}"
for seed in "${SEED_LIST[@]}"; do
  out_dir="${ROOT}/experiments/phase1_calvin/${STEM}${seed}"
  checkpoint="${out_dir}/${MODE_NAME}/policy.pt"
  [[ -f "${checkpoint}" ]] || { echo "Missing checkpoint: ${checkpoint}" >&2; exit 3; }

  for control in "${CONTROL_LIST[@]}"; do
    label="${control//:/_}"
    eval_dir="${out_dir}/${MODE_NAME}/eval_${label}"
    summary="${eval_dir}/state_bc_summary.json"
    log_path="${out_dir}/eval_${label}.log"
    [[ -f "${summary}" ]] && { echo "[semantic-eval] skip seed=${seed} control=${control}"; continue; }
    attempt=1
    while (( attempt <= MAX_RETRIES )); do
      echo "[semantic-eval] seed=${seed} control=${control} attempt=${attempt}" | tee -a "${log_path}"
      if "${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
        --calvin-code "${CALVIN_CODE}" \
        --data-root "${DATA_ROOT}" \
        --checkpoint "${checkpoint}" \
        --run-meta "${out_dir}/run_meta.json" \
        --eval-dir "${eval_dir}" \
        --num-sequences 100 \
        --ep-len 160 \
        --eval-scan-sequences 5000 \
        --action-execution-horizon 1 \
        --query-g-control "${control}" \
        --device "${DEVICE:-cuda:0}" \
          >> "${log_path}" 2>&1; then
        break
      fi
      [[ -f "${summary}" ]] && break
      attempt=$((attempt + 1))
      sleep 10
    done
    [[ -f "${summary}" ]] || { echo "Evaluation failed: seed=${seed} control=${control}" >&2; exit 4; }
  done

  seen_dir="${out_dir}/${MODE_NAME}/eval_seen_tasks"
  seen_summary="${seen_dir}/state_bc_summary.json"
  if [[ "${RUN_SEEN_TASKS}" == "1" && ! -f "${seen_summary}" ]]; then
    "${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
      --calvin-code "${CALVIN_CODE}" \
      --data-root "${DATA_ROOT}" \
      --checkpoint "${checkpoint}" \
      --run-meta "${out_dir}/run_meta.json" \
      --eval-dir "${seen_dir}" \
      --num-sequences 25 \
      --ep-len 160 \
      --eval-scan-sequences 5000 \
      --eval-task-filter open_drawer move_slider_left move_slider_right \
      --action-execution-horizon 1 \
      --query-g-control normal \
      --device "${DEVICE:-cuda:0}" \
        >> "${out_dir}/eval_seen_tasks.log" 2>&1
  fi
done
