#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
QUEUE_KIND="${QUEUE_KIND:?Set QUEUE_KIND to lang or projected}"

case "${QUEUE_KIND}" in
  lang)
    VARIANT="lang"
    STEM="2026-07-13_lang_firstw16_e50_n100_h160_seed"
    ;;
  projected)
    VARIANT="projected_nolang"
    STEM="2026-07-13_minimal_g_projected_firstw16_e50_n100_h160_seed"
    ;;
  *)
    echo "Unknown QUEUE_KIND=${QUEUE_KIND}; expected lang or projected" >&2
    exit 2
    ;;
esac

cd "${ROOT}"
for seed in 0 1 2 3 4; do
  out_dir="${ROOT}/experiments/phase1_calvin/${STEM}${seed}"
  echo "[queue:${QUEUE_KIND}] evaluating seed ${seed}" | tee -a "${out_dir}/run.log"
  VARIANT="${VARIANT}" \
  SEED="${seed}" \
  OUT_DIR="${out_dir}" \
  DEVICE="${DEVICE:-cuda:0}" \
  FIRST_ACTION_LOSS_WEIGHT=16 \
    bash framework/scripts/policy_state_bc/run_minimal_g_concat_structured_seed0.sh \
      >> "${out_dir}/run.log" 2>&1
done
