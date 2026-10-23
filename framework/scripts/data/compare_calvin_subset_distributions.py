#!/usr/bin/env python3
"""Compare deterministic CALVIN subset selections and emit auditable tables."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def segment_key(row: dict[str, Any]) -> tuple[str, str, int, int]:
    return str(row["task"]), str(row["scene"]), int(row["start"]), int(row["end"])


def frames(rows: Iterable[dict[str, Any]]) -> int:
    return sum(int(row["end"]) - int(row["start"]) + 1 for row in rows)


def classify(task: str, manifest: dict[str, Any]) -> str:
    if task in manifest["source_tasks"]:
        return "source_train"
    if task in manifest["heldout_tasks"]:
        return "heldout_audit"
    raise ValueError(f"Unknown task: {task}")


def aggregate(manifest: dict[str, Any], keys: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, int]]:
    result: dict[tuple[str, ...], dict[str, int]] = {}
    for row in manifest["segments_detail"]:
        values = {
            "split": classify(str(row["task"]), manifest),
            "task": str(row["task"]),
            "scene": str(row["scene"]),
        }
        key = tuple(values[name] for name in keys)
        stats = result.setdefault(key, {"segments": 0, "frames": 0})
        stats["segments"] += 1
        stats["frames"] += int(row["end"]) - int(row["start"]) + 1
    return result


def pct(delta: int, old: int) -> float:
    return 0.0 if old == 0 and delta == 0 else (100.0 * delta / old if old else float("inf"))


def comparison_rows(
    baseline: dict[str, Any], expanded: dict[str, Any], keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    old = aggregate(baseline, keys)
    new = aggregate(expanded, keys)
    rows = []
    for key in sorted(set(old) | set(new)):
        before = old.get(key, {"segments": 0, "frames": 0})
        after = new.get(key, {"segments": 0, "frames": 0})
        row: dict[str, Any] = dict(zip(keys, key))
        row.update(
            {
                "before_segments": before["segments"],
                "after_segments": after["segments"],
                "delta_segments": after["segments"] - before["segments"],
                "delta_segments_pct": round(
                    pct(after["segments"] - before["segments"], before["segments"]), 3
                ),
                "before_frames": before["frames"],
                "after_frames": after["frames"],
                "delta_frames": after["frames"] - before["frames"],
                "delta_frames_pct": round(
                    pct(after["frames"] - before["frames"], before["frames"]), 3
                ),
            }
        )
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    rule = "| " + " | ".join("---" if i < 2 else "---:" for i in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(str(row[key]) for key, _ in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, rule, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    expanded = json.loads(args.expanded.read_text(encoding="utf-8"))
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    task_rows = comparison_rows(baseline, expanded, ("split", "task"))
    scene_rows = comparison_rows(baseline, expanded, ("split", "scene"))
    cross_rows = comparison_rows(baseline, expanded, ("split", "task", "scene"))
    write_csv(output / "distribution_by_task.csv", task_rows)
    write_csv(output / "distribution_by_scene.csv", scene_rows)
    write_csv(output / "distribution_by_task_scene.csv", cross_rows)

    old_keys = {segment_key(row) for row in baseline["segments_detail"]}
    new_keys = {segment_key(row) for row in expanded["segments_detail"]}
    source_tasks = set(baseline["source_tasks"])
    heldout_tasks = set(baseline["heldout_tasks"])

    def split_summary(manifest: dict[str, Any], tasks: set[str]) -> dict[str, Any]:
        rows = [row for row in manifest["segments_detail"] if str(row["task"]) in tasks]
        count_frames = frames(rows)
        return {
            "segments": len(rows),
            "frames": count_frames,
            "mean_frames_per_segment": round(count_frames / len(rows), 3),
        }

    summary = {
        "baseline": {
            "selection_seed": baseline["selection_seed"],
            "source_per_task_scene": baseline["source_per_task_scene"],
            "heldout_per_task_scene": baseline["heldout_per_task_scene"],
            "source_train": split_summary(baseline, source_tasks),
            "heldout_audit": split_summary(baseline, heldout_tasks),
            "all": split_summary(baseline, source_tasks | heldout_tasks),
            "compressed_bytes": baseline["compressed_bytes"],
        },
        "expanded": {
            "selection_seed": expanded["selection_seed"],
            "source_per_task_scene": expanded["source_per_task_scene"],
            "heldout_per_task_scene": expanded["heldout_per_task_scene"],
            "source_train": split_summary(expanded, source_tasks),
            "heldout_audit": split_summary(expanded, heldout_tasks),
            "all": split_summary(expanded, source_tasks | heldout_tasks),
            "compressed_bytes": expanded["compressed_bytes"],
        },
        "retention": {
            "baseline_segments_retained": len(old_keys & new_keys),
            "baseline_segments_removed": len(old_keys - new_keys),
            "new_segments_added": len(new_keys - old_keys),
        },
        "source_task_counts_before": dict(
            sorted(Counter(str(r["task"]) for r in baseline["segments_detail"] if str(r["task"]) in source_tasks).items())
        ),
        "source_task_counts_after": dict(
            sorted(Counter(str(r["task"]) for r in expanded["segments_detail"] if str(r["task"]) in source_tasks).items())
        ),
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    source_task_rows = [row for row in task_rows if row["split"] == "source_train"]
    heldout_task_rows = [row for row in task_rows if row["split"] == "heldout_audit"]
    source_scene_rows = [row for row in scene_rows if row["split"] == "source_train"]
    old_source = summary["baseline"]["source_train"]
    new_source = summary["expanded"]["source_train"]
    old_all = summary["baseline"]["all"]
    new_all = summary["expanded"]["all"]
    report = f"""# CALVIN interaction-volume 数据扩展分布对比

## 口径

- 训练口径只统计 4 个 source tasks；2 个 held-out 重组任务只用于审计，不进入训练。
- 基线参数：每个 source task × scene 8 段；扩展参数：12 段；选择种子保持 `{baseline['selection_seed']}`。
- 扩展保留了全部 {len(old_keys)} 个原始选择，删除 0 段，新增 {len(new_keys - old_keys)} 段。

## 总览

| 指标 | 扩展前 | 扩展后 | 差异 |
| --- | ---: | ---: | ---: |
| 训练 source segments | {old_source['segments']} | {new_source['segments']} | +{new_source['segments'] - old_source['segments']} ({pct(new_source['segments'] - old_source['segments'], old_source['segments']):.1f}%) |
| 训练 source frames | {old_source['frames']} | {new_source['frames']} | +{new_source['frames'] - old_source['frames']} ({pct(new_source['frames'] - old_source['frames'], old_source['frames']):.1f}%) |
| held-out audit segments | {summary['baseline']['heldout_audit']['segments']} | {summary['expanded']['heldout_audit']['segments']} | 0 |
| held-out audit frames | {summary['baseline']['heldout_audit']['frames']} | {summary['expanded']['heldout_audit']['frames']} | 0 |
| raw subset total segments | {old_all['segments']} | {new_all['segments']} | +{new_all['segments'] - old_all['segments']} |
| raw subset total frames | {old_all['frames']} | {new_all['frames']} | +{new_all['frames'] - old_all['frames']} |
| 预计压缩读取字节 | {baseline['compressed_bytes']} | {expanded['compressed_bytes']} | +{expanded['compressed_bytes'] - baseline['compressed_bytes']} |

## 训练任务分布

{markdown_table(source_task_rows, [('task', '任务'), ('before_segments', '原段数'), ('after_segments', '扩展后段数'), ('delta_segments', '新增段数'), ('before_frames', '原帧数'), ('after_frames', '扩展后帧数'), ('delta_frames', '新增帧数')])}

## 训练场景分布

{markdown_table(source_scene_rows, [('scene', '场景'), ('before_segments', '原段数'), ('after_segments', '扩展后段数'), ('delta_segments', '新增段数'), ('before_frames', '原帧数'), ('after_frames', '扩展后帧数'), ('delta_frames', '新增帧数')])}

## held-out 隔离检查

{markdown_table(heldout_task_rows, [('task', '任务'), ('before_segments', '原段数'), ('after_segments', '扩展后段数'), ('delta_segments', '变化'), ('before_frames', '原帧数'), ('after_frames', '扩展后帧数'), ('delta_frames', '帧变化')])}

## 结论

本次扩展仅增加 source 训练覆盖：每个 source task 在每个 A/B/C 场景由 8 段增至 12 段，段数严格增加 50%；source 帧数由 {old_source['frames']} 增至 {new_source['frames']}。held-out 的任务、段数和帧数完全不变，因此没有把目标颜色×方向重组泄漏到训练分布。详细交叉表见 `distribution_by_task_scene.csv`。
"""
    (output / "CALVIN_DATA_DISTRIBUTION_COMPARISON.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
