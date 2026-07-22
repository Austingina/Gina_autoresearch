#!/usr/bin/env bash
set -euo pipefail

ROOT="${AUTORESEARCH_ROOT:-/mnt/nvme33n1/wuk_code/AutoResearch}"
CALVIN_ROOT="${CALVIN_ROOT:-/mnt/nvme33n1/wuk_code/calvin_g}"
DATA_DIR="${CALVIN_DATA_DIR:-$CALVIN_ROOT/data}"
DATASET_DIR="$DATA_DIR/task_D_D"
ZIP_PATH="$DATA_DIR/task_D_D.zip"
DATA_URL="http://calvin.cs.uni-freiburg.de/dataset/task_D_D.zip"
FIX_URL="http://calvin.cs.uni-freiburg.de/scene_info_fix/task_D_D_scene_info.zip"
RUN_DIR="${RUN_DIR:-$ROOT/experiments/phase1_calvin/$(date +%F_%H%M%S)_download_task_D_D_reusable_g_v1}"
PY="${PY:-$CALVIN_ROOT/envs/calvin_h200/bin/python}"
PIPELINE="$ROOT/framework/scripts/pipelines/run_reusable_g_validation_v1.sh"

extract_zip() {
  local zip_path="$1"
  local out_dir="$2"
  local mode="${3:-preserve}"
  if command -v unzip >/dev/null 2>&1; then
    if [ "$mode" = "overwrite" ]; then
      unzip -q -o "$zip_path" -d "$out_dir"
    else
      unzip -q -n "$zip_path" -d "$out_dir"
    fi
  else
    echo "[zip] unzip not found; extracting with Python zipfile"
    "$PY" - "$zip_path" "$out_dir" "$mode" <<'PY'
import sys
from pathlib import Path
from zipfile import ZipFile

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
mode = sys.argv[3]
out_dir.mkdir(parents=True, exist_ok=True)
with ZipFile(zip_path) as zf:
    members = zf.infolist()
    total = len(members)
    for i, info in enumerate(members, 1):
        target = out_dir / info.filename
        if mode != "overwrite" and target.exists():
            continue
        zf.extract(info, out_dir)
        if i % 5000 == 0 or i == total:
            print(f"[zip] extracted {i}/{total}", flush=True)
PY
  fi
}

mkdir -p "$RUN_DIR/logs" "$RUN_DIR/artifacts"
exec > >(tee -a "$RUN_DIR/logs/stdout.log") 2>&1

echo "[start] $(date -Is)"
echo "[run_dir] $RUN_DIR"
echo "[root] $ROOT"
echo "[calvin_root] $CALVIN_ROOT"
echo "[data_dir] $DATA_DIR"
echo "[dataset_dir] $DATASET_DIR"
echo "[python] $PY"

df -h /mnt/nvme33n1 /mnt/raid / || true

test -x "$PY"
test -x "$PIPELINE"
mkdir -p "$DATA_DIR"

cd "$DATA_DIR"

if [ -d "$DATASET_DIR/training" ] && [ -d "$DATASET_DIR/validation" ]; then
  echo "[download] dataset already present: $DATASET_DIR"
else
  echo "[download] fetching task_D_D.zip with resume support"
  wget -c "$DATA_URL" -O "$ZIP_PATH"
  echo "[download] unzip task_D_D.zip"
  extract_zip "$ZIP_PATH" "$DATA_DIR" preserve
fi

test -d "$DATASET_DIR/training"
test -d "$DATASET_DIR/validation"

echo "[fix] applying task_D_D scene_info fix if needed"
cd "$DATASET_DIR"
if [ ! -f ".autoresearch_scene_info_fix_applied" ]; then
  wget -c "$FIX_URL" -O task_D_D_scene_info.zip
  extract_zip task_D_D_scene_info.zip "$DATASET_DIR" overwrite
  touch ".autoresearch_scene_info_fix_applied"
else
  echo "[fix] already applied"
fi

cd "$ROOT"

echo "[verify] dataset tree"
du -sh "$DATASET_DIR" "$ZIP_PATH" 2>/dev/null || true
find "$DATASET_DIR" -maxdepth 2 -type d | sort | sed -n '1,80p'

RECIPES=(
  "object_transfer_red_blue_to_pink"
  "primitive_transfer_push_lift_rotate_to_place_stack"
  "articulation_transfer_slider_to_drawer"
  "articulation_transfer_drawer_to_slider"
)

for recipe in "${RECIPES[@]}"; do
  echo "[pipeline] recipe=$recipe"
  DATA_ROOT="$DATASET_DIR" \
  TAG="task_D_D_reusable_g_v1" \
  RECIPE="$recipe" \
  RAW_NUM_SEQUENCES="${RAW_NUM_SEQUENCES:-20}" \
  RAW_MAX_FRAMES="${RAW_MAX_FRAMES:-6}" \
  G_MAX_SEQUENCES="${G_MAX_SEQUENCES:-0}" \
  G_MAX_FRAMES_PER_SEQUENCE="${G_MAX_FRAMES_PER_SEQUENCE:-16}" \
  G_VIS_NUM_SEQUENCES="${G_VIS_NUM_SEQUENCES:-20}" \
  G_VIS_MAX_FRAMES="${G_VIS_MAX_FRAMES:-8}" \
  PREDICTOR_EPOCHS="${PREDICTOR_EPOCHS:-150}" \
  PREDICTOR_HORIZON="${PREDICTOR_HORIZON:-8}" \
  "$PIPELINE"
done

echo "[done] $(date -Is)"
