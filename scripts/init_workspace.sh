#!/usr/bin/env bash
# Create AutoResearch-owned data/checkpoint/run directories.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "${ROOT}/simulators/resolve_paths.sh"
cd "$ROOT"

STAGES=(s0_fixed s1_random_cube s2_random_goal s3_ood_object)

for stage in "${STAGES[@]}"; do
  mkdir -p "data/lift/${stage}/demos"
done
mkdir -p data/manifests checkpoints/lift runs

echo "=== AutoResearch workspace ==="
echo "  data/         project-owned demos (gitignored .hdf5)"
echo "  data/manifests/  import/collect provenance (track in git)"
echo "  checkpoints/  field + policy weights"
echo "  runs/         eval logs"
echo ""
echo "IsaacLab datasets/ is read-only — import or collect into data/ first."
echo ""
echo "Quick start:"
echo "  python scripts/import_isaac_demos.py --source \$ISAACLAB_ROOT/datasets/lift_cube_lenient_teacher_maskdepth_smoke.hdf5 --stage s0_fixed"
echo "  bash scripts/collect_lift_demos.sh   # new demos -> data/lift/s0_fixed/demos/"
