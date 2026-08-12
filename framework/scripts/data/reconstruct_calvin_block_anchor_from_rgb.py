#!/usr/bin/env python3
"""Diagnose RGB reconstruction of CALVIN block anchors for oracle-G recovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


BLOCK_SLICES = {
    "red": slice(6, 9),
    "blue": slice(12, 15),
    "pink": slice(18, 21),
}
TASK_COLORS = {
    f"push_{color}_block_{direction}": color
    for color in BLOCK_SLICES
    for direction in ("left", "right")
}


def color_mask(image: np.ndarray, color: str) -> np.ndarray:
    rgb = image.astype(np.int16)
    red, green, blue = (rgb[..., index] for index in range(3))
    if color == "red":
        mask = (red > 140) & (green < 130) & (blue < 130) & (red - green > 60) & (red - blue > 60)
    elif color == "blue":
        mask = (blue > 120) & (red < 150) & (green < 170) & (blue - red > 45) & (blue - green > 20)
    elif color == "pink":
        mask = (red > 130) & (blue > 130) & (green < 150) & (red - green > 50) & (blue - green > 50)
    else:
        raise ValueError(color)
    return mask.astype(np.uint8)


def component_features(image: np.ndarray, color: str) -> list[dict[str, object]]:
    mask = color_mask(image, color)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = mask.shape
    result: list[dict[str, object]] = []
    for label in range(1, count):
        x, y, box_width, box_height, area = stats[label]
        if area < 8:
            continue
        yy, xx = np.nonzero(labels == label)
        u = float(np.median(xx))
        v = float(np.median(yy))
        centered = np.stack([xx - xx.mean(), yy - yy.mean()], axis=1)
        covariance = centered.T @ centered / max(len(centered), 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        pixels = image[yy, xx].astype(np.float32) / 255.0
        feature = np.asarray(
            [
                u / width,
                v / height,
                float(centroids[label, 0]) / width,
                float(centroids[label, 1]) / height,
                np.log1p(area),
                box_width / width,
                box_height / height,
                area / max(box_width * box_height, 1),
                np.sqrt(max(eigenvalues[0], 0.0)) / width,
                np.sqrt(max(eigenvalues[1], 0.0)) / height,
                *pixels.mean(axis=0).tolist(),
                float(color == "red"),
                float(color == "blue"),
                float(color == "pink"),
            ],
            dtype=np.float32,
        )
        result.append(
            {
                "feature": feature,
                "center": np.asarray([u, v], dtype=np.float32),
                "area": int(area),
            }
        )
    return result


def camera_matrices(config: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    eye = np.asarray(config["look_from"], dtype=np.float64)
    target = np.asarray(config["look_at"], dtype=np.float64)
    up = np.asarray(config["up_vector"], dtype=np.float64)
    forward = (target - eye) / np.linalg.norm(target - eye)
    side = np.cross(forward, up)
    side /= np.linalg.norm(side)
    camera_up = np.cross(side, forward)
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = np.stack([side, camera_up, -forward], axis=0)
    view[:3, 3] = -view[:3, :3] @ eye

    fov = np.deg2rad(float(config["fov"]))
    aspect = float(config["aspect"])
    near = float(config["nearval"])
    far = float(config["farval"])
    focal = 1.0 / np.tan(fov / 2.0)
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = focal / aspect
    projection[1, 1] = focal
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = 2.0 * far * near / (near - far)
    projection[3, 2] = -1.0
    return view, projection @ view


def project(point: np.ndarray, matrix: np.ndarray, width: int, height: int) -> np.ndarray:
    homogeneous = matrix @ np.concatenate([point.astype(np.float64), [1.0]])
    normalized = homogeneous[:3] / homogeneous[3]
    return np.asarray(
        [(normalized[0] + 1.0) * width / 2.0, (1.0 - normalized[1]) * height / 2.0],
        dtype=np.float32,
    )


def deproject_with_camera_depth(
    pixel: np.ndarray,
    camera_depth: float,
    view: np.ndarray,
    config: dict[str, object],
) -> np.ndarray:
    width = int(config["width"])
    height = int(config["height"])
    focal_y = 1.0 / np.tan(np.deg2rad(float(config["fov"])) / 2.0)
    focal_x = focal_y / float(config["aspect"])
    x_ndc = 2.0 * (float(pixel[0]) + 0.5) / width - 1.0
    y_ndc = 1.0 - 2.0 * (float(pixel[1]) + 0.5) / height
    camera_point = np.asarray(
        [x_ndc * camera_depth / focal_x, y_ndc * camera_depth / focal_y, -camera_depth, 1.0],
        dtype=np.float64,
    )
    world = np.linalg.inv(view) @ camera_point
    return (world[:3] / world[3]).astype(np.float32)


def frame_tasks(split_dir: Path) -> dict[int, str]:
    annotation = np.load(
        split_dir / "lang_annotations" / "auto_lang_ann.npy", allow_pickle=True
    ).item()
    result: dict[int, str] = {}
    for task, interval in zip(annotation["language"]["task"], annotation["info"]["indx"]):
        start, end = map(int, interval)
        for frame_index in range(start, end + 1):
            result[frame_index] = str(task)
    return result


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--held-out-task", default="push_pink_block_right", choices=sorted(TASK_COLORS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees", type=int, default=256)
    parser.add_argument(
        "--regressor",
        choices=("projective_ridge", "ridge", "extra_trees"),
        default="projective_ridge",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    split_dir = args.data_root / "training"
    task_by_frame = frame_tasks(split_dir)
    camera_config = yaml.safe_load((split_dir / ".hydra" / "merged_config.yaml").read_text())["cameras"]["static"]
    view, matrix = camera_matrices(camera_config)
    width = int(camera_config["width"])
    height = int(camera_config["height"])

    train_features: list[np.ndarray] = []
    train_targets: list[np.ndarray] = []
    test_rows: list[dict[str, object]] = []
    matched_component_distances: list[float] = []
    for episode_path in sorted(split_dir.glob("episode_*.npz")):
        frame_index = int(episode_path.stem.rsplit("_", 1)[1])
        task = task_by_frame.get(frame_index, "")
        with np.load(episode_path, allow_pickle=False) as data:
            image = np.asarray(data["rgb_static"], dtype=np.uint8)
            scene = np.asarray(data["scene_obs"], dtype=np.float32)
            robot = np.asarray(data["robot_obs"], dtype=np.float32)
        for color, block_slice in BLOCK_SLICES.items():
            truth = scene[block_slice]
            components = component_features(image, color)
            if not components:
                continue
            truth_pixel = project(truth, matrix, width, height)
            distances = [float(np.linalg.norm(component["center"] - truth_pixel)) for component in components]
            matched_index = int(np.argmin(distances))
            if distances[matched_index] > 25.0:
                continue
            matched_component_distances.append(distances[matched_index])
            if task != args.held_out_task:
                train_features.append(components[matched_index]["feature"])
                if args.regressor == "projective_ridge":
                    camera_point = view @ np.concatenate([truth.astype(np.float64), [1.0]])
                    train_targets.append(np.asarray([-camera_point[2]], dtype=np.float32))
                else:
                    train_targets.append(truth)
            elif color == TASK_COLORS[args.held_out_task]:
                test_rows.append(
                    {
                        "frame_index": frame_index,
                        "truth": truth,
                        "eef": robot[:3],
                        "components": components,
                        "matched_index": matched_index,
                    }
                )

    if not train_features or not test_rows:
        raise RuntimeError("Insufficient calibration or held-out data")
    if args.regressor in {"projective_ridge", "ridge"}:
        model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            Ridge(alpha=1.0),
        )
    else:
        model = ExtraTreesRegressor(
            n_estimators=args.trees,
            min_samples_leaf=2,
            max_features=1.0,
            random_state=args.seed,
            n_jobs=-1,
        )
    model.fit(np.stack(train_features), np.stack(train_targets))

    automatic_errors: list[float] = []
    oracle_component_errors: list[float] = []
    attraction_cosines: list[float] = []
    component_correct: list[bool] = []
    rows_json: list[dict[str, object]] = []
    for row in test_rows:
        components = row["components"]
        raw_predictions = model.predict(np.stack([component["feature"] for component in components]))
        if args.regressor == "projective_ridge":
            predictions = np.stack(
                [
                    deproject_with_camera_depth(
                        component["center"], float(depth[0]), view, camera_config
                    )
                    for component, depth in zip(components, raw_predictions)
                ]
            )
        else:
            predictions = raw_predictions
        eef = row["eef"]
        truth = row["truth"]
        selected = int(np.argmin(np.linalg.norm(predictions - eef[None, :], axis=1)))
        prediction = predictions[selected]
        oracle_prediction = predictions[int(row["matched_index"])]
        automatic_errors.append(float(np.linalg.norm(prediction - truth)))
        oracle_component_errors.append(float(np.linalg.norm(oracle_prediction - truth)))
        component_correct.append(selected == int(row["matched_index"]))

        truth_attraction = truth - eef
        predicted_attraction = prediction - eef
        cosine = float(
            np.dot(truth_attraction, predicted_attraction)
            / max(np.linalg.norm(truth_attraction) * np.linalg.norm(predicted_attraction), 1e-8)
        )
        attraction_cosines.append(cosine)
        rows_json.append(
            {
                "frame_index": int(row["frame_index"]),
                "num_components": len(components),
                "selected_component": selected,
                "matched_component": int(row["matched_index"]),
                "truth": truth.tolist(),
                "prediction": prediction.tolist(),
                "error_m": automatic_errors[-1],
                "attraction_cosine": cosine,
            }
        )

    summary = {
        "data_root": str(args.data_root.resolve()),
        "held_out_task": args.held_out_task,
        "regressor": args.regressor,
        "train_examples": len(train_features),
        "test_frames": len(test_rows),
        "component_match_pixel_distance": summarize(matched_component_distances),
        "automatic_component_accuracy": float(np.mean(component_correct)),
        "automatic_anchor_error_m": summarize(automatic_errors),
        "oracle_component_anchor_error_m": summarize(oracle_component_errors),
        "attraction_cosine": summarize(attraction_cosines),
        "rows": rows_json,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
