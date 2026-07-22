#!/usr/bin/env bash
# Resolve AutoResearch + simulator roots for launch scripts.
# Usage: source "$AUTORESEARCH_ROOT/simulators/resolve_paths.sh"
#    or: source "$(cd "$(dirname "$0")/.." && pwd)/simulators/resolve_paths.sh"

# Caller may already have set ROOT / AUTORESEARCH_ROOT.
if [[ -z "${AUTORESEARCH_ROOT:-}" ]]; then
  if [[ -n "${ROOT:-}" ]]; then
    AUTORESEARCH_ROOT="${ROOT}"
  else
    # When sourced from simulators/, parent is repo root; from scripts/, go up one.
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

# Prefer bundled IsaacLab; fall back to legacy sibling /home/.../IsaacLab or $HOME/code/IsaacLab.
_bundled_isaac="${AUTORESEARCH_ROOT}/simulators/isaaclab"
_legacy_a="${HOME}/code/IsaacLab"
_legacy_b="/home/l/code/IsaacLab"

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
  if [[ -x "${_bundled_isaac}/isaaclab.sh" ]]; then
    ISAACLAB_ROOT="${_bundled_isaac}"
  elif [[ -x "${_legacy_a}/isaaclab.sh" ]]; then
    ISAACLAB_ROOT="${_legacy_a}"
  elif [[ -x "${_legacy_b}/isaaclab.sh" ]]; then
    ISAACLAB_ROOT="${_legacy_b}"
  else
    ISAACLAB_ROOT="${_bundled_isaac}"
  fi
fi
export ISAACLAB_ROOT

# Optional secondary platforms (only set defaults if unset).
export MANISKILL_ROOT="${MANISKILL_ROOT:-${AUTORESEARCH_ROOT}/simulators/maniskill}"
export ARENA_ROOT="${ARENA_ROOT:-${AUTORESEARCH_ROOT}/simulators/arena}"
export CONDA_ENV="${CONDA_ENV:-env_isaaclab}"
