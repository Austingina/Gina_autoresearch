#!/usr/bin/env bash
# Initialize a new autoresearch run branch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
  TAG="$(date +%b%d | tr '[:upper:]' '[:lower:]')"
fi

BRANCH="autoresearch/${TAG}"

if git show-ref --verify --quiet "refs/heads/${BRANCH}"; then
  echo "Branch ${BRANCH} already exists. Choose a different tag."
  exit 1
fi

git checkout -b "${BRANCH}"
echo "Created branch ${BRANCH}"
echo "Next: run baseline experiment and start agent loop per program.md"
