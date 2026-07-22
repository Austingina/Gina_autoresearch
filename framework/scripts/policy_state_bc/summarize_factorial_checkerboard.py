#!/usr/bin/env python3
"""Summarize E3 checkerboard held-out and source-task rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HELD_OUT_TASKS = [
    "close_drawer",
    "move_slider_left",
    "push_red_block_right",
    "push_blue_block_left",
]

CONDITIONS = ["with_g_normal", "with_g_sign_flip", "matched_no_g"]


def load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def task_result(summary: dict[str, Any] | None, task: str) -> tuple[int | None, int | None]:
    if summary is None:
        return None, None
    info = summary.get("task_info", {}).get(task)
    if info is None:
        return None, None
    return int(info["success"]), int(info["total"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    rows: list[dict[str, Any]] = []
    for task in HELD_OUT_TASKS:
        for condition in CONDITIONS:
            summary = load_summary(
                run_root / "eval" / f"{condition}_{task}" / "state_bc_summary.json"
            )
            success, total = task_result(summary, task)
            rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "success": success,
                    "total": total,
                    "sr": None if success is None or not total else success / total,
                }
            )

    source_rows: list[dict[str, Any]] = []
    for condition in ("with_g", "matched_no_g"):
        summary = load_summary(
            run_root / "eval" / f"{condition}_source12" / "state_bc_summary.json"
        )
        task_info = {} if summary is None else summary.get("task_info", {})
        success = sum(int(item["success"]) for item in task_info.values())
        total = sum(int(item["total"]) for item in task_info.values())
        source_rows.append(
            {
                "condition": condition,
                "success": success if task_info else None,
                "total": total if task_info else None,
                "sr": success / total if total else None,
                "tasks": task_info,
            }
        )

    report = {
        "run_root": run_root.as_posix(),
        "held_out_tasks": HELD_OUT_TASKS,
        "rows": rows,
        "source_health": source_rows,
    }
    with (run_root / "factorial_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    lines = [
        "# E3 Factorial Checkerboard",
        "",
        "| Held-out task | G normal | F sign-flip | Matched no-G |",
        "| --- | ---: | ---: | ---: |",
    ]
    indexed = {(row["task"], row["condition"]): row for row in rows}

    def cell(task: str, condition: str) -> str:
        row = indexed[(task, condition)]
        if row["success"] is None:
            return "pending"
        return f"{row['success']}/{row['total']}"

    for task in HELD_OUT_TASKS:
        lines.append(
            f"| {task} | {cell(task, 'with_g_normal')} | "
            f"{cell(task, 'with_g_sign_flip')} | {cell(task, 'matched_no_g')} |"
        )
    lines.extend(["", "## Source-task health", ""])
    for row in source_rows:
        if row["success"] is None:
            lines.append(f"- {row['condition']}: pending")
        else:
            lines.append(f"- {row['condition']}: {row['success']}/{row['total']}")
    (run_root / "factorial_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
