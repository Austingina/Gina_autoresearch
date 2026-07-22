#!/usr/bin/env python3
"""Export mask+depth preview videos from teacher HDF5 demos or DAgger episode .pt files.

Layout matches live collection preview (lenient_teacher_datacollection_maskdepth.py):

    ┌──────────┬──────────┐
    │ ee mask  │ ee depth │
    ├──────────┼──────────┤
    │ top mask │ top depth│
    └──────────┴──────────┘
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _float01_to_bgr(arr: np.ndarray, scale: int = 1) -> np.ndarray:
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    rgb = (np.clip(arr.astype(np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    if rgb.shape[-1] == 1:
        rgb = np.repeat(rgb, 3, axis=-1)
    elif rgb.shape[-1] >= 3:
        rgb = rgb[..., :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if scale > 1:
        bgr = cv2.resize(
            bgr,
            (bgr.shape[1] * scale, bgr.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return bgr


def _depth_to_bgr(arr: np.ndarray, scale: int = 1) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] == 1 else arr.mean(axis=-1)
    depth = np.clip(arr.astype(np.float32), 0.0, None)
    depth_max = float(depth.max())
    if depth_max < 1.0e-6:
        depth_max = 1.0
    gray = (np.clip(depth / depth_max, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    if scale > 1:
        bgr = cv2.resize(
            bgr,
            (bgr.shape[1] * scale, bgr.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    return bgr


def build_maskdepth_panel(
    ee_mask: np.ndarray,
    ee_depth: np.ndarray,
    tp_mask: np.ndarray,
    tp_depth: np.ndarray,
    *,
    scale: int,
    title: str,
    step: int,
    total: int,
    depth_colormap: bool,
) -> np.ndarray:
    depth_fn = _depth_to_bgr if depth_colormap else _float01_to_bgr
    row_ee = np.concatenate([_float01_to_bgr(ee_mask, scale), depth_fn(ee_depth, scale)], axis=1)
    row_tp = np.concatenate([_float01_to_bgr(tp_mask, scale), depth_fn(tp_depth, scale)], axis=1)
    panel = np.concatenate([row_ee, row_tp], axis=0)
    stats = (
        f"{title}  step={step}/{total}  "
        f"ee_mask mean={float(ee_mask.mean()):.3f}  top_mask mean={float(tp_mask.mean()):.3f}"
    )
    cv2.putText(panel, stats, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def _write_video(frames: list[np.ndarray], out_path: Path, fps: float) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")
    for frame in frames:
        if frame.shape[:2] != (h, w):
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_NEAREST)
        writer.write(frame)
    writer.release()


def _list_hdf5_demos(h5: h5py.File) -> list[str]:
    return sorted(k for k in h5["data"].keys() if k.startswith("demo_"))


def export_hdf5_demo(
    hdf5_path: Path,
    *,
    demo_name: str,
    output_dir: Path,
    max_steps: int,
    stride: int,
    fps: float,
    scale: int,
    depth_colormap: bool,
    show: bool,
) -> Path:
    with h5py.File(hdf5_path, "r") as f:
        grp = f["data"][demo_name]["obs"]
        ee_mask_ds = grp["ee_mask"]
        ee_depth_ds = grp["ee_depth"]
        tp_mask_ds = grp["third_person_mask"]
        tp_depth_ds = grp["third_person_depth"]
        total = int(ee_mask_ds.shape[0])
        end = total if max_steps <= 0 else min(total, max_steps)

        frames: list[np.ndarray] = []
        for step in range(0, end, stride):
            panel = build_maskdepth_panel(
                np.asarray(ee_mask_ds[step]),
                np.asarray(ee_depth_ds[step]),
                np.asarray(tp_mask_ds[step]),
                np.asarray(tp_depth_ds[step]),
                scale=scale,
                title=f"{hdf5_path.stem}/{demo_name}",
                step=step,
                total=total,
                depth_colormap=depth_colormap,
            )
            frames.append(panel)
            if show:
                cv2.imshow("train_maskdepth_preview", panel)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break

    out_path = output_dir / f"{hdf5_path.stem}_{demo_name}_maskdepth.mp4"
    _write_video(frames, out_path, fps)
    if show:
        cv2.destroyAllWindows()
    return out_path


def _decode_mosaic_latest_column(
    visual_t: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode latest mosaic column (4, 2H, W) into ee/top mask RGB + depth.

    Mosaic layout (per column):
      rows 0:H   = ee  mask RGB (ch0-2) + ee depth (ch3)
      rows H:2H  = top mask RGB (ch0-2) + top depth (ch3)
    """
    h = int(image_size)
    col = visual_t[:, :, -h:]
    ee = col[:, :h, :]
    top = col[:, h : 2 * h, :]
    ee_mask = np.transpose(ee[:3], (1, 2, 0))
    ee_depth = np.transpose(ee[3:4], (1, 2, 0))
    top_mask = np.transpose(top[:3], (1, 2, 0))
    top_depth = np.transpose(top[3:4], (1, 2, 0))
    mosaic_raw = col.transpose(1, 2, 0)
    return ee_mask, ee_depth, top_mask, top_depth, mosaic_raw


def build_mosaic_buffer_panel(
    visual_t: np.ndarray,
    *,
    image_size: int,
    scale: int,
    title: str,
    step: int,
    total: int,
    depth_colormap: bool,
) -> np.ndarray:
    ee_mask, ee_depth, top_mask, top_depth, _ = _decode_mosaic_latest_column(visual_t, image_size)
    depth_fn = _depth_to_bgr if depth_colormap else _float01_to_bgr

    row_top = np.concatenate([_float01_to_bgr(ee_mask, scale), depth_fn(ee_depth, scale)], axis=1)
    row_bot = np.concatenate([_float01_to_bgr(top_mask, scale), depth_fn(top_depth, scale)], axis=1)
    panel = np.concatenate([row_top, row_bot], axis=0)

    cv2.putText(panel, f"{title} step={step}/{total}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        f"ee_mask RGB mean={float(ee_mask.mean()):.3f} unique={len(np.unique(ee_mask.reshape(-1, 3), axis=0))}",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (200, 255, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(panel, "top: ee mask | ee depth", (8, row_top.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(
        panel,
        "bot: top mask | top depth (mosaic 4ch layout)",
        (8, panel.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    return panel


def _mosaic_latest_to_obs(visual_t: np.ndarray, image_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ee_mask, ee_depth, top_mask, top_depth, _ = _decode_mosaic_latest_column(visual_t, image_size)
    return ee_mask, ee_depth, top_mask, top_depth


def _detect_visual_layout(visual: np.ndarray, image_size: int) -> str:
    if visual.ndim != 4:
        raise ValueError(f"Expected visual (T,C,H,W), got {visual.shape}")
    _, channels, height, width = visual.shape
    h = int(image_size)
    if channels == 4 and height == 2 * h and width in (h, 3 * h):
        return "mosaic"
    if channels == 2 and height == 2 * h and width in (h, 3 * h):
        raise ValueError(
            f"Legacy 2-channel mosaic shape {visual.shape} is no longer supported; re-record buffer with 4ch mosaic."
        )
    if channels >= 8:
        return "channels8"
    raise ValueError(
        f"Unsupported visual shape {visual.shape}; expected mosaic (T,4,2H,W) or 8ch (T,8,H,W)"
    )


def _panel_from_visual_step(
    visual_t: np.ndarray,
    *,
    layout: str,
    image_size: int,
    scale: int,
    title: str,
    step: int,
    total: int,
    depth_colormap: bool,
) -> np.ndarray:
    if layout == "mosaic":
        return build_mosaic_buffer_panel(
            visual_t,
            image_size=image_size,
            scale=scale,
            title=title,
            step=step,
            total=total,
            depth_colormap=depth_colormap,
        )
    ee_mask = visual_t[0:3].transpose(1, 2, 0)
    ee_depth = visual_t[3:4].transpose(1, 2, 0)
    tp_mask = visual_t[4:7].transpose(1, 2, 0)
    tp_depth = visual_t[7:8].transpose(1, 2, 0)
    return build_maskdepth_panel(
        ee_mask,
        ee_depth,
        tp_mask,
        tp_depth,
        scale=scale,
        title=title,
        step=step,
        total=total,
        depth_colormap=depth_colormap,
    )


def _resolve_buffer_dir(source: Path) -> Path:
    if source.is_file():
        return source.parent
    if (source / "episodes").is_dir():
        return source / "episodes"
    if any(source.glob("ep_*.pt")):
        return source
    raise FileNotFoundError(f"No DAgger episodes under {source}")


def play_dagger_buffer_opencv(
    buffer_dir: Path,
    *,
    image_size: int = 240,
    fps: float = 20.0,
    stride: int = 1,
    scale: int = 1,
    depth_colormap: bool = True,
    start_episode: int = 0,
    max_episodes: int = 0,
    window_name: str = "dagger_buffer_preview",
) -> int:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch required for DAgger buffer playback") from exc

    files = sorted(buffer_dir.glob("ep_*.pt"))
    if not files:
        raise FileNotFoundError(f"No ep_*.pt files in {buffer_dir}")

    start_episode = max(0, min(start_episode, len(files) - 1))
    end_episode = len(files) if max_episodes <= 0 else min(len(files), start_episode + max_episodes)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    delay_ms = max(1, int(round(1000.0 / fps)))
    ep_idx = start_episode
    paused = False

    print(f"[INFO] Buffer: {buffer_dir}")
    print(f"[INFO] Episodes: {len(files)} | playing [{start_episode}, {end_episode})")
    print("[INFO] Keys: q=quit | n/→=next ep | p/←=prev ep | space=pause | +/-=speed | g=jump")

    while 0 <= ep_idx < end_episode:
        ep_path = files[ep_idx]
        ep = torch.load(ep_path, map_location="cpu", weights_only=False)
        visual = ep["visual"].numpy()
        layout = _detect_visual_layout(visual, image_size)
        total = int(visual.shape[0])
        title = f"{ep_path.name} ({layout}) [{ep_idx + 1}/{len(files)}]"

        step = 0
        while step < total:
            panel = _panel_from_visual_step(
                visual[step],
                layout=layout,
                image_size=image_size,
                scale=scale,
                title=title,
                step=step,
                total=total,
                depth_colormap=depth_colormap,
            )
            help_y = panel.shape[0] - 28
            cv2.putText(
                panel,
                "q quit | n next ep | p prev | space pause | +/- speed",
                (8, help_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, panel)

            wait = 0 if paused else delay_ms
            key = cv2.waitKey(wait) & 0xFF
            if key == ord("q"):
                cv2.destroyAllWindows()
                return 0
            if key in (ord("n"), 83):  # n or right arrow
                ep_idx += 1
                break
            if key in (ord("p"), 81):  # p or left arrow
                ep_idx = max(0, ep_idx - 1)
                break
            if key == ord(" "):
                paused = not paused
                continue
            if key in (ord("+"), ord("=")):
                delay_ms = max(1, delay_ms - 5)
                continue
            if key in (ord("-"), ord("_")):
                delay_ms = min(500, delay_ms + 5)
                continue
            if key == ord("g"):
                cv2.destroyWindow(window_name)
                try:
                    text = input(f"Jump to episode index [0,{len(files) - 1}]: ").strip()
                    ep_idx = int(text)
                except (ValueError, EOFError):
                    ep_idx = ep_idx
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                ep_idx = max(0, min(ep_idx, len(files) - 1))
                break

            if not paused:
                step += stride
        else:
            ep_idx += 1

    cv2.destroyAllWindows()
    print("[INFO] Finished all episodes in range.")
    return 0


def export_dagger_episode(
    episode_path: Path,
    *,
    output_dir: Path,
    image_size: int,
    max_steps: int,
    stride: int,
    fps: float,
    scale: int,
    depth_colormap: bool,
    show: bool,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch required for DAgger episode export") from exc

    ep = torch.load(episode_path, map_location="cpu", weights_only=False)
    visual = ep["visual"].numpy()
    layout = _detect_visual_layout(visual, image_size)
    total = int(visual.shape[0])
    end = total if max_steps <= 0 else min(total, max_steps)
    frames: list[np.ndarray] = []
    for step in range(0, end, stride):
        panel = _panel_from_visual_step(
            visual[step],
            layout=layout,
            image_size=image_size,
            scale=scale,
            title=f"{episode_path.stem} {layout}",
            step=step,
            total=total,
            depth_colormap=depth_colormap,
        )
        frames.append(panel)
        if show:
            cv2.imshow("train_maskdepth_preview", panel)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    out_path = output_dir / f"{episode_path.stem}_maskdepth.mp4"
    _write_video(frames, out_path, fps)
    if show:
        cv2.destroyAllWindows()
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        type=Path,
        help="Teacher .hdf5, DAgger episode .pt, or buffer dir (dagger_episodes/ or episodes/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "train_maskdepth_video",
    )
    parser.add_argument("--demo", default="demo_0", help="HDF5 demo group name or index (e.g. 0 -> demo_0)")
    parser.add_argument("--image-size", type=int, default=240, help="Spatial size for mosaic .pt decode")
    parser.add_argument("--max-steps", type=int, default=0, help="Cap frames per export (0 = all)")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--scale", type=int, default=1, help="Upscale panels for easier viewing")
    parser.add_argument("--depth-colormap", action="store_true", help="Use TURBO colormap for depth tiles")
    parser.add_argument("--show", action="store_true", help="Live OpenCV preview while exporting")
    parser.add_argument(
        "--play-buffer",
        action="store_true",
        help="OpenCV playback of all ep_*.pt in a DAgger buffer directory (no mp4 export)",
    )
    parser.add_argument("--start-episode", type=int, default=0, help="First episode index for --play-buffer")
    parser.add_argument("--max-episodes", type=int, default=0, help="Episode cap for --play-buffer (0=all)")
    parser.add_argument("--list-demos", action="store_true", help="List demos in HDF5 and exit")
    parser.add_argument("--list-episodes", action="store_true", help="List ep_*.pt in buffer dir and exit")
    args = parser.parse_args()

    source = args.source if args.source.is_absolute() else ROOT / args.source
    if not source.exists():
        print(f"[ERROR] not found: {source}", file=sys.stderr)
        return 1

    if source.is_dir() or args.play_buffer:
        buffer_dir = _resolve_buffer_dir(source)
        files = sorted(buffer_dir.glob("ep_*.pt"))
        if args.list_episodes:
            for i, fp in enumerate(files):
                print(f"{i:4d}  {fp.name}")
            print(f"[INFO] total={len(files)} dir={buffer_dir}")
            return 0
        if not files:
            print(f"[ERROR] no ep_*.pt in {buffer_dir}", file=sys.stderr)
            return 1
        return play_dagger_buffer_opencv(
            buffer_dir,
            image_size=args.image_size,
            fps=args.fps,
            stride=args.stride,
            scale=args.scale,
            depth_colormap=args.depth_colormap or True,
            start_episode=args.start_episode,
            max_episodes=args.max_episodes,
        )

    if not source.is_file():
        print(f"[ERROR] not a file: {source}", file=sys.stderr)
        return 1

    if source.suffix == ".hdf5":
        with h5py.File(source, "r") as f:
            demos = _list_hdf5_demos(f)
            if args.list_demos:
                for name in demos:
                    n = int(f["data"][name]["obs"]["ee_mask"].shape[0])
                    print(f"{name}: {n} steps")
                return 0
        demo_name = args.demo
        if demo_name.isdigit():
            demo_name = f"demo_{demo_name}"
        if demo_name not in demos:
            print(f"[ERROR] {demo_name} not in {source}; available: {demos[:5]}...", file=sys.stderr)
            return 1
        out = export_hdf5_demo(
            source,
            demo_name=demo_name,
            output_dir=args.output_dir,
            max_steps=args.max_steps,
            stride=args.stride,
            fps=args.fps,
            scale=args.scale,
            depth_colormap=args.depth_colormap,
            show=args.show,
        )
    elif source.suffix == ".pt":
        out = export_dagger_episode(
            source,
            output_dir=args.output_dir,
            image_size=args.image_size,
            max_steps=args.max_steps,
            stride=args.stride,
            fps=args.fps,
            scale=args.scale,
            depth_colormap=args.depth_colormap,
            show=args.show,
        )
    else:
        print(f"[ERROR] unsupported source type: {source.suffix}", file=sys.stderr)
        return 1

    print(f"[OK] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
