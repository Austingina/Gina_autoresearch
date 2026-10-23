#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, REPO.as_posix())

from framework.sim_data.contracts import EpisodeRequest, NativeEpisodeResult, OracleEvidence
from framework.sim_data.gates import calvin_oracle_gate, libero_oracle_gate, robotwin_oracle_gate
from framework.sim_data.runner import NativeCollectionRunner


class FakeCalvinOracle:
    def get_task_info_for_set(self, initial, final, tasks):
        return {next(iter(tasks))} if final.get("done") else set()


class FakeLiberoEnv:
    def __init__(self, success=True): self.success = success
    def check_success(self): return self.success


class FakeRoboTwinTask:
    plan_success = True
    def check_success(self): return True
    def planned_joints_legal(self): return True, 0.0


class FakeAdapter:
    simulator_id = "calvin"
    def __init__(self, root: Path):
        self.root = root
        self.calls = 0

    def run_episode(self, request: EpisodeRequest) -> NativeEpisodeResult:
        self.calls += 1
        success = self.calls > 1
        evidence = OracleEvidence(
            simulator="calvin",
            task=request.task,
            checker="fake_native_oracle",
            success=success,
            details={"same_rollout": True},
        )
        paths = ()
        if success:
            path = self.root / "accepted" / f"{request.episode_id}.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"native-trajectory")
            paths = (path,)
        return NativeEpisodeResult(
            request=request,
            oracle=evidence,
            artifact_paths=paths,
            native_action_schema={"kind": "calvin_rel7"},
            native_observation_schema={"kind": "calvin_dual_rgb_state"},
            canonical_task={"primitive": "push", "object": "blue_block", "direction": "left"},
            provenance={"collector": "fake"},
        )


def test_native_gates_fail_closed_and_use_native_checkers():
    assert calvin_oracle_gate(FakeCalvinOracle(), {"done": False}, {"done": True}, "push").success
    assert libero_oracle_gate(FakeLiberoEnv(True), "open_drawer").success
    assert robotwin_oracle_gate(FakeRoboTwinTask(), "handover_block").success
    for bad in (object(),):
        for gate, args in (
            (libero_oracle_gate, (bad, "task")),
            (robotwin_oracle_gate, (bad, "task")),
        ):
            try:
                gate(*args)
            except (AttributeError, RuntimeError, TypeError, ValueError):
                pass
            else:
                raise AssertionError("missing native checker must fail closed")


def test_runner_keeps_failures_metadata_only_and_hashes_accepts():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runner = NativeCollectionRunner(root, adapter=FakeAdapter(root))
        requests = [
            EpisodeRequest("ep_fail", "push_blue_block_left", seed=1, reset_id="r1"),
            EpisodeRequest("ep_ok", "push_blue_block_left", seed=2, reset_id="r2"),
        ]
        manifest = runner.collect(requests, accepted_quota=1)
        assert manifest["accepted"] == 1 and manifest["failed"] == 1
        assert not list((root / "failures").glob("*.npz"))
        assert len(list((root / "failures").glob("*.receipt.json"))) == 1
        accepted = next(row for row in manifest["results"] if row["oracle_success"])
        assert accepted["artifacts"][0]["sha256"]
        serialized = json.dumps(manifest).lower()
        assert '"g"' not in serialized and "minimal_g" not in serialized
        verified = runner.validate_manifest()
        assert verified["accepted_verified"] == 1


def test_missing_accepted_artifact_fails_closed():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        runner = NativeCollectionRunner(root, adapter=FakeAdapter(root))
        manifest = runner.collect(
            [EpisodeRequest("ep_fail", "push", 1, "r1"), EpisodeRequest("ep_ok", "push", 2, "r2")],
            accepted_quota=1,
        )
        artifact = Path(manifest["results"][1]["artifacts"][0]["path"])
        (root / artifact).unlink()
        try:
            runner.validate_manifest()
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("missing accepted artifact must fail closed")


def main():
    test_native_gates_fail_closed_and_use_native_checkers()
    test_runner_keeps_failures_metadata_only_and_hashes_accepts()
    test_missing_accepted_artifact_fails_closed()
    print("cross-simulator native collection contract: OK")


if __name__ == "__main__":
    main()
