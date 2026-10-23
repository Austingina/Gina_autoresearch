"""Quota runner and readback validator for native simulator artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .contracts import EpisodeRequest, NativeCollectorAdapter, NativeEpisodeResult, assert_no_candidate_g


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=_json_default)
        stream.write("\n")
    os.replace(temporary, path)


class NativeCollectionRunner:
    def __init__(self, output_root: Path, *, adapter: NativeCollectorAdapter):
        self.output_root = Path(output_root).resolve()
        self.adapter = adapter

    def _artifact_record(self, path: Path) -> dict[str, Any]:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(self.output_root)
        except ValueError as exc:
            raise ValueError(f"Native artifact must be under collection root: {resolved}") from exc
        if not relative.parts or relative.parts[0] != "accepted":
            raise ValueError(f"Accepted native artifact must be under accepted/: {relative}")
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return {"path": relative.as_posix(), "bytes": resolved.stat().st_size, "sha256": _sha256(resolved)}

    def _base_row(self, result: NativeEpisodeResult) -> dict[str, Any]:
        request = result.request
        row = {
            "episode_id": request.episode_id,
            "task": request.task,
            "seed": request.seed,
            "reset_id": request.reset_id,
            "task_metadata": dict(request.task_metadata),
            "simulator": result.oracle.simulator,
            "oracle_checker": result.oracle.checker,
            "oracle_success": result.oracle.success,
            "oracle_details": dict(result.oracle.details),
            "native_action_schema": dict(result.native_action_schema),
            "native_observation_schema": dict(result.native_observation_schema),
            "canonical_task": dict(result.canonical_task),
            "provenance": dict(result.provenance),
        }
        assert_no_candidate_g(row, path="manifest.result")
        return row

    def collect(self, requests: Iterable[EpisodeRequest], *, accepted_quota: int) -> dict[str, Any]:
        if accepted_quota < 1:
            raise ValueError("accepted_quota must be positive")
        manifest_path = self.output_root / "collection_manifest.json"
        if manifest_path.exists():
            raise FileExistsError(f"Refusing to overwrite collection: {manifest_path}")
        results: list[dict[str, Any]] = []
        accepted = 0
        failed = 0
        for request in requests:
            if accepted >= accepted_quota:
                break
            result = self.adapter.run_episode(request)
            if result.request != request:
                raise ValueError("Adapter returned evidence for a different episode request")
            if result.oracle.simulator != self.adapter.simulator_id:
                raise ValueError("Adapter simulator_id and Oracle evidence disagree")
            row = self._base_row(result)
            if result.oracle.success:
                row["failure_storage_mode"] = None
                row["artifacts"] = [self._artifact_record(Path(path)) for path in result.artifact_paths]
                accepted += 1
            else:
                row["failure_storage_mode"] = "metadata_only"
                row["artifacts"] = []
                receipt_path = self.output_root / "failures" / f"{request.episode_id}.receipt.json"
                if receipt_path.exists():
                    raise FileExistsError(receipt_path)
                receipt_payload = dict(row)
                receipt_payload["receipt_format"] = "native_failure_receipt_v1"
                _write_json_atomic(receipt_path, receipt_payload)
                row["receipt_path"] = receipt_path.relative_to(self.output_root).as_posix()
                failed += 1
            results.append(row)
        manifest = {
            "format": "cross_sim_native_collection_v1",
            "simulator": self.adapter.simulator_id,
            "accepted_quota": accepted_quota,
            "accepted": accepted,
            "failed": failed,
            "complete": accepted >= accepted_quota,
            "candidate_guidance": "not_part_of_collection_contract",
            "results": results,
        }
        assert_no_candidate_g(manifest, path="manifest")
        _write_json_atomic(manifest_path, manifest)
        return manifest

    def validate_manifest(self) -> dict[str, Any]:
        manifest_path = self.output_root / "collection_manifest.json"
        with open(manifest_path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("format") != "cross_sim_native_collection_v1":
            raise ValueError(f"Unexpected manifest format: {manifest.get('format')!r}")
        assert_no_candidate_g(manifest, path="manifest")
        accepted_verified = 0
        failed_verified = 0
        for row in manifest.get("results", []):
            if bool(row.get("oracle_success")):
                artifacts = list(row.get("artifacts", []))
                if not artifacts:
                    raise ValueError("Accepted row has no native artifacts")
                for artifact in artifacts:
                    path = self.output_root / artifact["path"]
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    if path.stat().st_size != int(artifact["bytes"]):
                        raise ValueError(f"Artifact size mismatch: {path}")
                    if _sha256(path) != artifact["sha256"]:
                        raise ValueError(f"Artifact SHA-256 mismatch: {path}")
                accepted_verified += 1
            else:
                if row.get("failure_storage_mode") != "metadata_only" or row.get("artifacts"):
                    raise ValueError("Failed row is not metadata-only")
                receipt = self.output_root / row["receipt_path"]
                if not receipt.is_file():
                    raise FileNotFoundError(receipt)
                failed_verified += 1
        if accepted_verified != int(manifest.get("accepted", -1)):
            raise ValueError("Accepted count mismatch")
        if failed_verified != int(manifest.get("failed", -1)):
            raise ValueError("Failed count mismatch")
        return {
            "manifest": manifest_path,
            "accepted_verified": accepted_verified,
            "failed_verified": failed_verified,
            "complete": bool(manifest.get("complete", False)),
        }
