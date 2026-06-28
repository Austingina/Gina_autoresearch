#!/usr/bin/env python3
"""Summarize eval_diagnostics.json into human-readable failure analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODE_LABELS = {
    "success": "成功到达 goal",
    "never_reached_cube": "EE 从未靠近 cube（approach 失败）",
    "never_grasped": "靠近 cube 但未抓起来（cube 未离桌）",
    "grasp_then_dropped": "抓起来后又掉回桌面",
    "lifted_not_at_goal": "保持抬起但未到达 goal（post-grasp wander）",
    "timeout": "超时结束",
    "terminated": "异常终止（如 cube 掉落触发 termination）",
    "other": "其他",
}


def summarize_report(data: dict) -> str:
    agg = data.get("aggregate", {})
    episodes = data.get("episodes", [])
    lines = [
        "# Eval Diagnostic Analysis",
        "",
        f"- **checkpoint:** `{data.get('checkpoint', '')}`",
        f"- **guidance_mode:** {data.get('guidance_mode', '?')}",
        f"- **episodes:** {agg.get('num_episodes', len(episodes))}",
        f"- **success_rate:** {agg.get('success_rate', 0):.1%}",
        f"- **success_rate@5cm (aux):** {agg.get('success_rate_5cm', 0):.1%}",
        f"- **grasp_rate:** {agg.get('grasp_rate', 0):.1%}",
        f"- **mean post_grasp_wander_m:** {agg.get('mean_post_grasp_wander_m', 0):.4f}",
        f"- **mean min EE-cube dist:** {agg.get('mean_min_ee_cube_dist', 0):.4f} m",
        "",
        "## Failure mode breakdown",
        "",
        "| Mode | Count | % | 含义 |",
        "|------|-------|---|------|",
    ]
    n = max(agg.get("num_episodes", 1), 1)
    for mode, count in sorted(agg.get("failure_mode_counts", {}).items(), key=lambda x: -x[1]):
        pct = 100 * count / n
        label = MODE_LABELS.get(mode, mode)
        lines.append(f"| `{mode}` | {count} | {pct:.0f}% | {label} |")

    lines.extend(["", "## Per-episode detail", ""])
    for ep in episodes:
        lines.append(
            f"- **ep {ep['episode_id']}** `{ep['failure_mode']}` "
            f"steps={ep['steps']} max_z={ep['max_cube_z']:.3f} "
            f"final_dist_goal={ep['final_dist_to_goal']:.3f} "
            f"min_ee_cube={ep['min_ee_cube_dist']:.3f} "
            f"wander={ep['post_grasp_wander_m']:.3f}"
        )

    lines.extend(["", "## Interpretation hints", ""])
    counts = agg.get("failure_mode_counts", {})
    if counts.get("never_reached_cube", 0) > n * 0.3:
        lines.append("- 主要失败在 **approach**：policy 没有把 EE 移到 cube 附近。")
    if counts.get("never_grasped", 0) > n * 0.3:
        lines.append("- 主要失败在 **grasp**：能接近但未闭合 gripper / 未抬起 cube。")
    if counts.get("grasp_then_dropped", 0) > 0:
        lines.append("- 存在 **抓起后掉落**：grasp 不稳定或 lift 阶段松爪。")
    if counts.get("lifted_not_at_goal", 0) > 0:
        lines.append("- 存在 **抬起但未 relocation**：正是 G 场要解决的 post-grasp wander。")
    if agg.get("success_rate", 0) == 0 and counts.get("never_grasped", 0) == n:
        lines.append("- **全部未 grasp**：B1 vision student 尚未学到有效抓取；与 DAgger buffer 全 teacher 轨迹一致。")

    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("json_path", type=Path)
    p.add_argument("--markdown", type=Path, default=None)
    args = p.parse_args()
    if not args.json_path.is_file():
        print(f"Missing {args.json_path}", file=sys.stderr)
        return 1
    data = json.loads(args.json_path.read_text())
    text = summarize_report(data)
    print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
