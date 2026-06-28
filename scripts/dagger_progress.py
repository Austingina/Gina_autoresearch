#!/usr/bin/env python3
"""Poll TensorBoard events and write human-readable progress (works while DAgger runs)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def read_progress(log_dir: Path) -> dict:
    tb = log_dir / "tb"
    files = sorted(tb.glob("events.out.tfevents.*"))
    if not files:
        return {"status": "no_tb_events", "log_dir": str(log_dir)}

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(files[-1]), size_guidance={"scalars": 0})
    ea.Reload()

    def last(tag: str):
        if tag not in ea.Tags().get("scalars", []):
            return None
        ev = ea.Scalars(tag)
        return ev[-1].value if ev else None

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "log_dir": str(log_dir.resolve()),
        "iteration": last("train/iteration"),
        "buffer_episodes": last("train/num_episodes"),
        "buffer_steps": last("train/buffer_steps"),
        "loss": last("train/total_loss"),
        "lift_success_rate": last("sampler/lift_success_rate"),
        "eval_success_rate": last("eval/success_rate"),
    }


def format_progress(data: dict) -> str:
    if data.get("status") == "no_tb_events":
        return f"[{data.get('updated_at', '')}] waiting for TensorBoard events in {data['log_dir']}"
    it = data.get("iteration")
    buf = data.get("buffer_episodes")
    steps = data.get("buffer_steps")
    loss = data.get("loss")
    lift = data.get("lift_success_rate")
    ev = data.get("eval_success_rate")
    lines = [
        f"updated: {data.get('updated_at', '')}",
        f"iteration: {it}",
        f"buffer: {buf} episodes, {steps} steps",
        f"loss: {loss}",
        f"teacher lift_success: {lift}",
    ]
    if ev is not None:
        lines.append(f"eval success: {ev}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("log_dir", type=Path, nargs="?", default=Path("runs/s0_B1_restart"))
    p.add_argument("--watch", type=float, default=0, help="Poll every N seconds (0 = once)")
    args = p.parse_args()

    while True:
        try:
            data = read_progress(args.log_dir)
        except Exception as e:
            data = {"status": "error", "error": str(e), "log_dir": str(args.log_dir)}

        text = format_progress(data)
        out_txt = args.log_dir / "progress.txt"
        out_json = args.log_dir / "progress_from_tb.json"
        out_txt.write_text(text + "\n")
        out_json.write_text(json.dumps(data, indent=2) + "\n")
        print(text)

        if args.watch <= 0:
            break
        time.sleep(args.watch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
