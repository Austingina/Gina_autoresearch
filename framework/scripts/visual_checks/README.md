# Visual Checks

First-pass scripts for CALVIN reusable interaction geometry experiments.

Run with the H200 CALVIN environment:

```bash
/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python framework/scripts/visual_checks/visualize_calvin_episode.py
```

Suggested first smoke:

```bash
DATA=/mnt/nvme33n1/wuk_code/calvin_g/data/calvin_debug_dataset
PY=/mnt/nvme33n1/wuk_code/calvin_g/envs/calvin_h200/bin/python

$PY framework/scripts/visual_checks/visualize_calvin_episode.py --dataset-root "$DATA" --num-sequences 2 --max-frames 4
$PY framework/scripts/visual_checks/summarize_calvin_dataset.py --dataset-root "$DATA"
$PY framework/scripts/visual_checks/make_calvin_task_split_manifest.py --dataset-root "$DATA" --recipe object_transfer_red_blue_to_pink
$PY framework/scripts/visual_checks/extract_candidate_g_labels.py --dataset-root "$DATA" --split-manifest artifacts/task_splits/object_transfer_red_blue_to_pink.json --partition source --max-sequences 4 --max-frames-per-sequence 8
```
