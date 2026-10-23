#!/usr/bin/env python3
"""Focused contracts for the bounded RoboTwin native pilot adapter."""
from __future__ import annotations

import unittest

from framework.scripts.data.collect_robotwin_bounded_pilot import (
    PilotQuotaError,
    ReplayOracleError,
    replay_accepted_plan,
    search_accepted_plan,
)


class FakeTask:
    def __init__(self, outcome, events):
        self.outcome = outcome
        self.events = events
        self.plan_success = False

    def setup_demo(self, **kwargs):
        self.seed = kwargs["seed"]
        self.events.append(("setup", self.seed))

    def play_once(self):
        self.plan_success = self.outcome["plan"]
        self.events.append(("play", self.seed))
        return {"seed": self.seed}

    def check_success(self):
        self.events.append(("oracle", self.seed))
        return self.outcome["oracle"]

    def planned_joints_legal(self):
        self.events.append(("joints", self.seed))
        return self.outcome["joints"], self.outcome.get("excess", 0.0)

    def save_traj_data(self, idx):
        self.events.append(("save", self.seed, idx))

    def close_env(self):
        self.events.append(("close", self.seed))


class BoundedSearchTests(unittest.TestCase):
    def test_accepts_only_combined_native_gate_and_closes_every_attempt(self):
        outcomes = {
            0: {"plan": False, "oracle": True, "joints": True},
            1: {"plan": True, "oracle": False, "joints": True},
            2: {"plan": True, "oracle": True, "joints": False, "excess": 0.2},
            3: {"plan": True, "oracle": True, "joints": True},
        }
        events = []

        def factory(seed):
            return FakeTask(outcomes[seed], events)

        result = search_accepted_plan(
            factory,
            base_args={"task_name": "place_empty_cup", "save_path": "/tmp/pilot"},
            start_seed=0,
            max_attempts=4,
            trajectory_index=0,
        )

        self.assertEqual(result.accepted_seed, 3)
        self.assertEqual(len(result.attempts), 4)
        self.assertEqual([a.accepted for a in result.attempts], [False, False, False, True])
        self.assertEqual([event for event in events if event[0] == "save"], [("save", 3, 0)])
        self.assertEqual([event for event in events if event[0] == "close"], [
            ("close", 0), ("close", 1), ("close", 2), ("close", 3)
        ])

    def test_exhaustion_is_bounded_and_preserves_receipts(self):
        events = []

        def factory(seed):
            return FakeTask({"plan": True, "oracle": False, "joints": True}, events)

        with self.assertRaises(PilotQuotaError) as raised:
            search_accepted_plan(
                factory,
                base_args={"task_name": "place_empty_cup", "save_path": "/tmp/pilot"},
                start_seed=10,
                max_attempts=3,
                trajectory_index=0,
            )

        self.assertEqual([a.seed for a in raised.exception.attempts], [10, 11, 12])
        self.assertEqual(len([e for e in events if e[0] == "close"]), 3)
        self.assertFalse(any(e[0] == "save" for e in events))

    def test_nonpositive_attempt_bound_is_rejected_without_creating_env(self):
        calls = []
        with self.assertRaises(ValueError):
            search_accepted_plan(
                lambda seed: calls.append(seed),
                base_args={},
                start_seed=0,
                max_attempts=0,
                trajectory_index=0,
            )
        self.assertEqual(calls, [])


class FakeReplayTask:
    def __init__(self, oracle_success, events):
        self.oracle_success = oracle_success
        self.events = events

    def setup_demo(self, **kwargs):
        self.seed = kwargs["seed"]
        self.events.append("setup")

    def load_tran_data(self, idx):
        self.events.append("load")
        return {"left_joint_path": [1], "right_joint_path": [2]}

    def set_path_lst(self, args):
        self.events.append("set_paths")
        self.asserted_args = args

    def play_once(self):
        self.events.append("play")
        return {"info": {"object": "cup", "target": "coaster"}}

    def check_success(self):
        self.events.append("oracle")
        return self.oracle_success

    def close_env(self):
        self.events.append("close")

    def merge_pkl_to_hdf5_video(self, instructions=None):
        self.events.append(("merge", tuple(instructions or [])))

    def remove_data_cache(self):
        self.events.append("remove_cache")


class ReplayTests(unittest.TestCase):
    def test_oracle_is_checked_before_hdf5_write(self):
        events = []
        task = FakeReplayTask(True, events)
        result = replay_accepted_plan(
            lambda seed: task,
            base_args={"task_name": "place_empty_cup", "save_path": "/tmp/pilot"},
            accepted_seed=7,
            trajectory_index=0,
            instructions=["place the empty cup on the coaster"],
        )
        self.assertTrue(result.oracle_success)
        self.assertLess(events.index("oracle"), events.index(("merge", ("place the empty cup on the coaster",))))
        self.assertEqual(events.count("close"), 1)
        self.assertTrue(task.asserted_args["save_data"])
        self.assertFalse(task.asserted_args["need_plan"])
        self.assertEqual(task.asserted_args["render_freq"], 0)

    def test_pre_replay_probe_captures_reference_after_setup_before_motion(self):
        events = []
        task = FakeReplayTask(True, events)

        def capture_reference(replay_task):
            events.append("reference")
            replay_task.upright_reference = "captured"

        def quality_probe(replay_task):
            events.append("quality")
            return {
                "passed": replay_task.upright_reference == "captured",
                "reference": replay_task.upright_reference,
            }

        result = replay_accepted_plan(
            lambda seed: task,
            base_args={"task_name": "place_empty_cup", "save_path": "/tmp/pilot"},
            accepted_seed=7,
            trajectory_index=0,
            instructions=["place cup"],
            pre_replay_probe=capture_reference,
            quality_probe=quality_probe,
        )

        self.assertLess(events.index("setup"), events.index("reference"))
        self.assertLess(events.index("reference"), events.index("play"))
        self.assertLess(events.index("oracle"), events.index("quality"))
        self.assertTrue(result.quality["passed"])

    def test_failed_replay_never_writes_hdf5(self):
        events = []
        task = FakeReplayTask(False, events)
        with self.assertRaises(ReplayOracleError):
            replay_accepted_plan(
                lambda seed: task,
                base_args={"task_name": "place_empty_cup", "save_path": "/tmp/pilot"},
                accepted_seed=9,
                trajectory_index=0,
                instructions=["place cup"],
            )
        self.assertNotIn("remove_cache", events)
        self.assertFalse(any(isinstance(event, tuple) and event[0] == "merge" for event in events))
        self.assertEqual(events.count("close"), 1)


if __name__ == "__main__":
    unittest.main()
