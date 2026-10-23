"""Simulator-neutral native data collection contracts.

The common schema intentionally contains no candidate guidance representation.
Each simulator retains its native action/observation schema and native success
checker; only provenance, task semantics, Oracle evidence, and artifact hashes
are standardized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


FORBIDDEN_DERIVED_KEYS = {"g", "minimal_g", "candidate_g", "provisional_g"}


def assert_no_candidate_g(value: Any, *, path: str = "root") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in FORBIDDEN_DERIVED_KEYS:
                raise ValueError(f"Candidate G field is forbidden in native collection contract: {path}.{key}")
            assert_no_candidate_g(child, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_no_candidate_g(child, path=f"{path}[{index}]")


@dataclass(frozen=True)
class EpisodeRequest:
    episode_id: str
    task: str
    seed: int
    reset_id: str
    task_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.episode_id or not self.task or not self.reset_id:
            raise ValueError("episode_id, task, and reset_id are required")
        assert_no_candidate_g(self.task_metadata, path="task_metadata")


@dataclass(frozen=True)
class OracleEvidence:
    simulator: str
    task: str
    checker: str
    success: bool
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.simulator or not self.task or not self.checker:
            raise ValueError("simulator, task, and checker are required in Oracle evidence")
        if not isinstance(self.success, bool):
            raise TypeError("Oracle success must be an explicit bool")
        assert_no_candidate_g(self.details, path="oracle.details")


@dataclass(frozen=True)
class NativeEpisodeResult:
    request: EpisodeRequest
    oracle: OracleEvidence
    artifact_paths: Sequence[Path]
    native_action_schema: Mapping[str, Any]
    native_observation_schema: Mapping[str, Any]
    canonical_task: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.oracle.task != self.request.task:
            raise ValueError("Oracle task must match requested task")
        for name, payload in (
            ("native_action_schema", self.native_action_schema),
            ("native_observation_schema", self.native_observation_schema),
            ("canonical_task", self.canonical_task),
            ("provenance", self.provenance),
        ):
            assert_no_candidate_g(payload, path=name)
        if self.oracle.success and not self.artifact_paths:
            raise ValueError("Successful episode must provide at least one native artifact")
        if not self.oracle.success and self.artifact_paths:
            raise ValueError("Failed episode must be metadata-only in the common contract")


class NativeCollectorAdapter(Protocol):
    simulator_id: str

    def run_episode(self, request: EpisodeRequest) -> NativeEpisodeResult:
        """Run the native expert and return native Oracle evidence/artifacts."""
        ...
