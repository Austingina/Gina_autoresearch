#!/usr/bin/env bash
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/../../simulators/resolve_paths.sh"
ROOT="${AUTORESEARCH_ROOT}"
CALVIN_CODE="${CALVIN_CODE}"
CALVIN_MODELS="$CALVIN_CODE/calvin_models"
CALVIN_ENV="${CALVIN_ENV}"
CALVIN_DATASET="${CALVIN_DATASET}"

printf '[check] root: %s\n' "$ROOT"
printf '[check] calvin_code: %s\n' "$CALVIN_CODE"
printf '[check] calvin_env: %s\n' "$CALVIN_ENV"
printf '[check] dataset: %s\n' "$CALVIN_DATASET"

test -d "$ROOT"
test -d "$CALVIN_CODE"
test -d "$CALVIN_MODELS"
test -d "$CALVIN_ENV"
test -f "$CALVIN_DATASET/training/.hydra/merged_config.yaml"
test -f "$CALVIN_DATASET/validation/.hydra/merged_config.yaml"

conda run --no-capture-output -p "$CALVIN_ENV" python - <<'PY'
import torch
print('[torch]', torch.__version__, 'cuda', torch.version.cuda)
print('[cuda_available]', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('CUDA is not available')
print('[device_count]', torch.cuda.device_count())
print('[device0]', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
print('[arch]', torch.cuda.get_arch_list())
if 'sm_90' not in torch.cuda.get_arch_list():
    raise SystemExit('Torch build does not include sm_90 for H200')
PY

if [ -x "$CALVIN_CODE/calvin_env/egl_check/EGL_options.o" ]; then
  "$CALVIN_CODE/calvin_env/egl_check/EGL_options.o" >/tmp/autoresearch_egl_stdout.txt 2>/tmp/autoresearch_egl_stderr.txt || true
  sed -n '1,20p' /tmp/autoresearch_egl_stderr.txt
else
  echo '[warn] EGL_options.o is not executable yet; CALVIN can build it during rollout.'
fi

echo '[check] CALVIN ready.'
