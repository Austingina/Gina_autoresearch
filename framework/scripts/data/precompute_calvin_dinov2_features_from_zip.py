#!/usr/bin/env python3
"""Cache frozen DINOv2 features for a compact CALVIN episode subset."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F


IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def preprocess(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    array = np.stack(images)
    tensor = torch.from_numpy(array).to(device=device, dtype=torch.float32)
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGE_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGE_STD, device=device).view(1, 3, 1, 1)
    return (tensor - mean) / std


def encode_batch(
    model: torch.nn.Module,
    static_images: list[np.ndarray],
    gripper_images: list[np.ndarray],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    static = preprocess(static_images, device)
    gripper = preprocess(gripper_images, device)
    pixels = torch.cat([static, gripper], dim=0)
    amp_enabled = device.type == "cuda"
    amp_dtype = torch.float32
    if amp_enabled:
        major, _ = torch.cuda.get_device_capability(device)
        amp_dtype = torch.bfloat16 if major >= 8 else torch.float16
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        features = model(pixel_values=pixels).last_hidden_state[:, 0]
    features = features.float().cpu().numpy().astype(np.float16)
    split = len(static_images)
    return features[:split], features[split:]


def shard_matches(path: Path, expected_indices: np.ndarray) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            actual = np.asarray(data["frame_index"], dtype=np.int64)
            return np.array_equal(actual, expected_indices)
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--source-zip", type=Path)
    source_group.add_argument(
        "--source-dir",
        type=Path,
        help="Extracted CALVIN dataset root containing training/episode_*.npz.",
    )
    parser.add_argument("--compact-data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="facebook/dinov2-small")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Limit frames for a smoke run; 0 processes the full compact subset.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.batch_size <= 0 or args.shard_size <= 0:
        parser.error("--batch-size and --shard-size must be positive")
    episode_files = sorted((args.compact_data_root / "training").glob("episode_*.npz"))
    if not episode_files:
        raise FileNotFoundError(
            f"No compact CALVIN episodes under {args.compact_data_root / 'training'}"
        )
    if args.max_frames > 0:
        episode_files = episode_files[: args.max_frames]
    frame_indices = np.asarray(
        [int(path.stem.rsplit("_", 1)[1]) for path in episode_files], dtype=np.int64
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    from transformers import Dinov2Model

    model = Dinov2Model.from_pretrained(args.model_name).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    started = time.time()
    completed = 0
    archive = zipfile.ZipFile(args.source_zip) if args.source_zip is not None else None
    try:
        first_name = episode_files[0].name
        if archive is not None:
            suffix = f"/training/{first_name}"
            matches = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one archive member ending in {suffix!r}, found {len(matches)}"
                )
            training_prefix = matches[0][: -len(first_name)]
        else:
            training_dir = args.source_dir / "training"
            if not (training_dir / first_name).is_file():
                raise FileNotFoundError(
                    f"Source episode not found: {training_dir / first_name}"
                )

        for shard_start in range(0, len(episode_files), args.shard_size):
            shard_stop = min(len(episode_files), shard_start + args.shard_size)
            shard_id = shard_start // args.shard_size
            shard_path = args.output_dir / f"shard_{shard_id:05d}.npz"
            expected_indices = frame_indices[shard_start:shard_stop]
            if shard_matches(shard_path, expected_indices):
                completed += len(expected_indices)
                print(
                    f"[skip] shard={shard_id} frames={completed}/{len(episode_files)}",
                    flush=True,
                )
                continue

            shard_static: list[np.ndarray] = []
            shard_gripper: list[np.ndarray] = []
            for batch_start in range(shard_start, shard_stop, args.batch_size):
                batch_stop = min(shard_stop, batch_start + args.batch_size)
                static_images: list[np.ndarray] = []
                gripper_images: list[np.ndarray] = []
                for episode_file in episode_files[batch_start:batch_stop]:
                    if archive is not None:
                        payload = io.BytesIO(archive.read(training_prefix + episode_file.name))
                    else:
                        payload = args.source_dir / "training" / episode_file.name
                    with np.load(payload, allow_pickle=False) as data:
                        static_images.append(np.asarray(data["rgb_static"], dtype=np.uint8))
                        gripper_images.append(np.asarray(data["rgb_gripper"], dtype=np.uint8))
                static_features, gripper_features = encode_batch(
                    model, static_images, gripper_images, device
                )
                shard_static.append(static_features)
                shard_gripper.append(gripper_features)

            temporary_path = shard_path.with_suffix(".tmp.npz")
            np.savez(
                temporary_path,
                frame_index=expected_indices,
                static=np.concatenate(shard_static, axis=0),
                gripper=np.concatenate(shard_gripper, axis=0),
            )
            os.replace(temporary_path, shard_path)
            completed += len(expected_indices)
            elapsed = max(time.time() - started, 1e-6)
            print(
                f"[write] shard={shard_id} frames={completed}/{len(episode_files)} "
                f"rate={completed / elapsed:.1f} frame/s",
                flush=True,
            )
    finally:
        if archive is not None:
            archive.close()

    manifest = {
        "source_zip": str(args.source_zip.resolve()) if args.source_zip is not None else None,
        "source_dir": str(args.source_dir.resolve()) if args.source_dir is not None else None,
        "compact_data_root": str(args.compact_data_root.resolve()),
        "model_name": args.model_name,
        "feature_dim": int(model.config.hidden_size),
        "cameras": ["rgb_static", "rgb_gripper"],
        "num_frames": int(len(episode_files)),
        "frame_index_min": int(frame_indices.min()),
        "frame_index_max": int(frame_indices.max()),
        "dtype": "float16",
        "image_size": 224,
        "normalization_mean": IMAGE_MEAN,
        "normalization_std": IMAGE_STD,
        "shard_size": int(args.shard_size),
        "max_frames": int(args.max_frames),
    }
    manifest_path = args.output_dir / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    with open(temporary_manifest, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    os.replace(temporary_manifest, manifest_path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
