#!/usr/bin/env bash
# Resolve AutoResearch + simulator roots for launch scripts.
# Usage: source simulators/resolve_paths.sh

# Caller may already have set ROOT / AUTORESEARCH_ROOT.
if [[ -z "${AUTORESEARCH_ROOT:-}" ]]; then
  if [[ -n "${ROOT:-}" ]]; then
    AUTORESEARCH_ROOT="${ROOT}"
  else
    _here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "${_here}/../pyproject.toml" ]]; then
      AUTORESEARCH_ROOT="$(cd "${_here}/.." && pwd)"
    elif [[ -f "${_here}/../../pyproject.toml" ]]; then
      AUTORESEARCH_ROOT="$(cd "${_here}/../.." && pwd)"
    else
      AUTORESEARCH_ROOT="$(cd "${_here}/.." && pwd)"
    fi
  fi
fi
export AUTORESEARCH_ROOT
export ROOT="${ROOT:-$AUTORESEARCH_ROOT}"

# --- Isaac Lab (Phase 0) ---
_bundled_isaac="${AUTORESEARCH_ROOT}/simulators/isaaclab"
_legacy_a="${HOME}/code/IsaacLab"
_legacy_b="/home/l/code/IsaacLab"
_legacy_c="/home/wukui/code/IsaacLab"

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
  if [[ -x "${_bundled_isaac}/isaaclab.sh" ]]; then
    ISAACLAB_ROOT="${_bundled_isaac}"
  elif [[ -x "${_legacy_a}/isaaclab.sh" ]]; then
    ISAACLAB_ROOT="${_legacy_a}"
  elif [[ -x "${_legacy_b}/isaaclab.sh" ]]; then
    ISAACLAB_ROOT="${_legacy_b}"
  elif [[ -x "${_legacy_c}/isaaclab.sh" ]]; then
    ISAACLAB_ROOT="${_legacy_c}"
  else
    ISAACLAB_ROOT="${_bundled_isaac}"
  fi
fi
export ISAACLAB_ROOT

export MANISKILL_ROOT="${MANISKILL_ROOT:-${AUTORESEARCH_ROOT}/simulators/maniskill}"
export ARENA_ROOT="${ARENA_ROOT:-${AUTORESEARCH_ROOT}/simulators/arena}"
export CONDA_ENV="${CONDA_ENV:-env_isaaclab}"

# --- CALVIN (Phase 1) ---
# Prefer explicit CALVIN_G_ROOT; else common H200 path; else phase1_calvin links.
if [[ -z "${CALVIN_G_ROOT:-}" ]]; then
  if [[ -d /mnt/nvme33n1/wuk_code/calvin_g ]]; then
    CALVIN_G_ROOT="/mnt/nvme33n1/wuk_code/calvin_g"
  elif [[ -d "${AUTORESEARCH_ROOT}/phase1_calvin/calvin_g" ]]; then
    CALVIN_G_ROOT="${AUTORESEARCH_ROOT}/phase1_calvin/calvin_g"
  fi
fi
export CALVIN_G_ROOT="${CALVIN_G_ROOT:-}"

_calvin_link_code="${AUTORESEARCH_ROOT}/phase1_calvin/calvin_code"
_calvin_link_env="${AUTORESEARCH_ROOT}/phase1_calvin/envs/calvin_h200"
_calvin_link_data="${AUTORESEARCH_ROOT}/phase1_calvin/data"

if [[ -z "${CALVIN_CODE:-}" ]]; then
  if [[ -d "${_calvin_link_code}" ]]; then
    CALVIN_CODE="${_calvin_link_code}"
  elif [[ -n "${CALVIN_G_ROOT}" && -d "${CALVIN_G_ROOT}/code/calvin" ]]; then
    CALVIN_CODE="${CALVIN_G_ROOT}/code/calvin"
  else
    CALVIN_CODE="${_calvin_link_code}"
  fi
fi
export CALVIN_CODE

if [[ -z "${CALVIN_ENV:-}" ]]; then
  if [[ -d "${_calvin_link_env}" ]]; then
    CALVIN_ENV="${_calvin_link_env}"
  elif [[ -n "${CALVIN_G_ROOT}" && -d "${CALVIN_G_ROOT}/envs/calvin_h200" ]]; then
    CALVIN_ENV="${CALVIN_G_ROOT}/envs/calvin_h200"
  else
    CALVIN_ENV="${_calvin_link_env}"
  fi
fi
export CALVIN_ENV

if [[ -z "${CALVIN_DATASET:-}" ]]; then
  if [[ -d "${_calvin_link_data}/calvin_debug_dataset" ]]; then
    CALVIN_DATASET="${_calvin_link_data}/calvin_debug_dataset"
  elif [[ -n "${CALVIN_G_ROOT}" && -d "${CALVIN_G_ROOT}/data/calvin_debug_dataset" ]]; then
    CALVIN_DATASET="${CALVIN_G_ROOT}/data/calvin_debug_dataset"
  else
    CALVIN_DATASET="${_calvin_link_data}/calvin_debug_dataset"
  fi
fi
export CALVIN_DATASET
