#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE] [EVAL_TAG]}"
GPU="${2:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE] [EVAL_TAG]}"
STATE_SCOPE="${3:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE] [EVAL_TAG]}"
VARIANT="${4:?usage: $0 SEED GPU STATE_SCOPE VARIANT [LANGUAGE_MODE] [EVAL_TAG]}"
LANGUAGE_MODE="${5:-onehot}"
EVAL_TAG="${6:-}"

ROOT="/mnt/nvme33n1/wuk_code/AutoResearch"
LANGUAGE_TAG=""
if [[ "${LANGUAGE_MODE}" != "onehot" ]]; then
  LANGUAGE_TAG="_${LANGUAGE_MODE}"
fi
RUN_ROOT="${ROOT}/experiments/phase1_calvin/2026-07-19_M1_abc_to_d_oracle_g${LANGUAGE_TAG}_${STATE_SCOPE}_seed${SEED}/${VARIANT}"
CHECKPOINT="${RUN_ROOT}/chunk_bc_minimal_g_tokens/policy.pt"
LOG_ROOT="${ROOT}/logs/M1_abc_d_scene_gate${LANGUAGE_TAG}"
mkdir -p "${LOG_ROOT}"

until [[ -f "${CHECKPOINT}" ]]; do
  sleep 30
done

if [[ "${VARIANT}" == "with_g" ]]; then
  CONTROLS=(normal zero sign_flip)
elif [[ "${VARIANT}" == "language_only" ]]; then
  CONTROLS=(zero)
else
  echo "VARIANT must be with_g or language_only" >&2
  exit 2
fi

for control in "${CONTROLS[@]}"; do
  echo "[eval] seed=${SEED} scope=${STATE_SCOPE} variant=${VARIANT} control=${control}"
  LOG_SUFFIX=""
  if [[ -n "${EVAL_TAG}" ]]; then
    LOG_SUFFIX="_${EVAL_TAG}"
  fi
  bash "${ROOT}/framework/scripts/run_stage1_abc_d_oracle_g_scene_gate_eval.sh" \
    "${SEED}" "${GPU}" "${STATE_SCOPE}" "${VARIANT}" "${control}" normal "${LANGUAGE_MODE}" "${EVAL_TAG}" \
    > "${LOG_ROOT}/eval_seed${SEED}_${STATE_SCOPE}_${VARIANT}_${control}${LOG_SUFFIX}.log" 2>&1
done

echo "[done] seed=${SEED} scope=${STATE_SCOPE} variant=${VARIANT}"
