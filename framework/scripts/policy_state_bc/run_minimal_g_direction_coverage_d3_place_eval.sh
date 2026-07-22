#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
PYTHON="${PYTHON:-${ROOT}/phase1_calvin/envs/calvin_h200/bin/python}"
CALVIN_CODE="${CALVIN_CODE:-${ROOT}/phase1_calvin/calvin_code}"
DATA_ROOT="${DATA_ROOT:-${ROOT}/phase1_calvin/data/task_D_D}"
read -r -a SEED_LIST <<< "${SEEDS:-0 1 2 3 4}"
read -r -a WAIT_PIDS <<< "${WAIT_FOR_PIDS:-}"
MAX_RETRIES="${MAX_RETRIES:-50}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MODE="chunk_bc_minimal_g_film"
STEM="2026-07-15_D3_signed_direction_coverage_film_h1_seed"

for wait_pid in "${WAIT_PIDS[@]}"; do
  [[ -z "${wait_pid}" ]] && continue
  while kill -0 "${wait_pid}" 2>/dev/null; do
    sleep "${POLL_SECONDS}"
  done
done

cd "${ROOT}"
for seed in "${SEED_LIST[@]}"; do
  out_dir="${ROOT}/experiments/phase1_calvin/${STEM}${seed}"
  checkpoint="${out_dir}/${MODE}/policy.pt"
  run_meta="${out_dir}/run_meta.json"
  eval_dir="${out_dir}/${MODE}/eval_seen_place_in_slider_n25"
  summary="${eval_dir}/state_bc_summary.json"
  [[ -f "${summary}" ]] && continue
  attempt=1
  while (( attempt <= MAX_RETRIES )); do
    echo "[D3-place-eval] seed=${seed} attempt=${attempt}"
    if "${PYTHON}" framework/scripts/policy_state_bc/eval_state_chunk_policy_ckpt.py \
      --calvin-code "${CALVIN_CODE}" \
      --data-root "${DATA_ROOT}" \
      --checkpoint "${checkpoint}" \
      --run-meta "${run_meta}" \
      --eval-dir "${eval_dir}" \
      --num-sequences 25 \
      --ep-len 160 \
      --eval-scan-sequences 50000 \
      --eval-task-filter place_in_slider \
      --action-execution-horizon 1 \
      --query-g-control normal \
      --language-control normal \
      --device "${DEVICE:-cuda:0}"; then
      break
    fi
    [[ -f "${summary}" ]] && break
    attempt=$((attempt + 1))
    sleep 10
  done
  [[ -f "${summary}" ]] || {
    echo "D3 place_in_slider evaluation failed after ${MAX_RETRIES} attempts" >&2
    exit 4
  }
done

echo "[complete] D3 place_in_slider seen-task evaluation finished"
