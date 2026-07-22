# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Offline CQL-SAC training on lenient-teacher mask+depth HDF5 datasets.

Reference implementation: https://github.com/wukui-muc/Offline_RL_Active_Tracking

Uses LSTM sequences (default length 20) and performs ``updates_per_epoch`` gradient
steps per training epoch (default 20), matching the reference repo's batched update style.

Transition tuple from dataset:
    (s_t, env_action_t, reward_t, s_{t+1})

Training reward (recomputed at load from ee_pose + cube position):
    reward = reach_reward + near_reward + success_bonus
    (progress_delta in HDF5 is diagnostic only; not used for CQL)

Example:

.. code-block:: bash

    python scripts/reinforcement_learning/train_lift_cube_offline_cql_lstm.py \\
        --dataset_file ./datasets/lift_cube_lenient_teacher_maskdepth \\
        --log_dir ./runs/lift_cube_offline_cql_lstm \\
        --obs_mode vision \\
        --lstm_seq_len 20 \\
        --updates_per_epoch 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

# Import offline_cql without loading isaaclab_tasks (which requires Isaac Sim / pxr).
_OFFLINE_CQL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/mdp/teacher_student"
)
if str(_OFFLINE_CQL_ROOT) not in sys.path:
    sys.path.insert(0, str(_OFFLINE_CQL_ROOT))

from offline_cql import (  # noqa: E402
    CQLSACAgentLSTM,
    EpisodeSequenceBuffer,
    load_maskdepth_hdf5_episodes,
    resolve_maskdepth_dataset_paths,
)
from offline_cql.rewards import format_reward_summary, summarize_dataset_rewards  # noqa: E402
from offline_cql.checkpoint import pack_checkpoint  # noqa: E402
from offline_cql.sim_eval_runner import run_isaac_sim_eval  # noqa: E402

_ISAACLAB_DIR = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline CQL-SAC (LSTM) for lift-cube mask+depth demos.")
    parser.add_argument(
        "--dataset_file",
        type=str,
        default="./datasets/lift_cube_lenient_teacher_maskdepth",
        help="HDF5 file or directory of *.hdf5 files from lenient_teacher_datacollection_maskdepth.py",
    )
    parser.add_argument("--log_dir", type=str, default="./runs/lift_cube_offline_cql_lstm")
    parser.add_argument("--run_name", type=str, default="cql_sac_lstm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1000, help="Number of outer training epochs.")
    parser.add_argument(
        "--updates_per_epoch",
        type=int,
        default=20,
        help="Number of CQL gradient updates per epoch (reference repo uses batched inner updates).",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lstm_seq_len", type=int, default=20, help="LSTM sequence length.")
    parser.add_argument("--lstm_hidden_dim", type=int, default=128)
    parser.add_argument("--lstm_layers", type=int, default=1)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--cnn_feat_dim", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=5e-3)
    parser.add_argument("--cql_weight", type=float, default=1.0)
    parser.add_argument("--cql_temp", type=float, default=1.0)
    parser.add_argument("--target_action_gap", type=float, default=10.0)
    parser.add_argument("--with_lagrange", action="store_true", default=False)
    parser.add_argument("--num_random_actions", type=int, default=10)
    parser.add_argument(
        "--obs_mode",
        type=str,
        default="vision",
        choices=("vision", "proprio", "vision_proprio"),
        help="Observation format loaded from HDF5.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Resize visual observations to this square size (reference repo uses 64x64).",
    )
    parser.add_argument("--max_demos", type=int, default=0, help="Limit number of demos loaded (0 = all).")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument(
        "--eval_every",
        type=int,
        default=50,
        help="Run Isaac Sim rollout evaluation every N epochs (0 = disabled).",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=20,
        help="Number of simulation episodes per eval.",
    )
    parser.add_argument(
        "--eval_task",
        type=str,
        default="Isaac-Lift-Cube-Franka-Vision-MaskDepth-Play-v0",
    )
    parser.add_argument(
        "--eval_num_envs",
        type=int,
        default=1,
        help="Parallel envs during Isaac Sim eval.",
    )
    parser.add_argument(
        "--eval_episode_length_s",
        type=float,
        default=30.0,
        help="Episode timeout for Isaac Sim eval.",
    )
    parser.add_argument(
        "--eval_position_threshold",
        type=float,
        default=0.02,
        help="Lift success distance threshold for Isaac Sim eval.",
    )
    parser.add_argument(
        "--isaaclab_dir",
        type=str,
        default=str(_ISAACLAB_DIR),
        help="Isaac Lab root containing isaaclab.sh (for subprocess sim eval).",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        default=False,
        help="Disable periodic Isaac Sim evaluation.",
    )
    parser.add_argument(
        "--eval_headless",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Isaac Sim eval headless (recommended during training).",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run_dir = os.path.join(args.log_dir, args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    max_demos = None if args.max_demos <= 0 else args.max_demos
    image_size = None if args.image_size <= 0 else args.image_size
    dataset_files = resolve_maskdepth_dataset_paths(args.dataset_file)
    episodes, action_stats = load_maskdepth_hdf5_episodes(
        args.dataset_file,
        obs_mode=args.obs_mode,
        image_size=image_size,
        max_demos=max_demos,
    )
    reward_summary = summarize_dataset_rewards(episodes)
    action_dim = int(episodes[0]["actions"].shape[-1])

    buffer = EpisodeSequenceBuffer(
        episodes,
        action_stats=action_stats,
        seq_len=args.lstm_seq_len,
        obs_mode=args.obs_mode,
    )

    proprio_dim = 0
    if args.obs_mode in ("proprio", "vision_proprio"):
        proprio_dim = int(episodes[0]["proprio"].shape[-1])

    agent = CQLSACAgentLSTM(
        action_dim=action_dim,
        device=device,
        obs_mode=args.obs_mode,
        visual_channels=8,
        proprio_dim=proprio_dim,
        hidden_dim=args.hidden_dim,
        cnn_feat_dim=args.cnn_feat_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        lstm_layers=args.lstm_layers,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        cql_weight=args.cql_weight,
        cql_temp=args.cql_temp,
        target_action_gap=args.target_action_gap,
        with_lagrange=args.with_lagrange,
        num_random_actions=args.num_random_actions,
    )

    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    with open(os.path.join(run_dir, "reward_stats.json"), "w", encoding="utf-8") as f:
        json.dump(reward_summary, f, indent=2)
    with open(os.path.join(run_dir, "action_stats.json"), "w", encoding="utf-8") as f:
        json.dump({"low": action_stats.low.tolist(), "high": action_stats.high.tolist()}, f, indent=2)

    print(f"[INFO] Dataset: {args.dataset_file}")
    print(f"[INFO] Loaded {len(dataset_files)} HDF5 file(s): {', '.join(Path(p).name for p in dataset_files)}")
    print(f"[INFO] Episodes: {len(episodes)} | sequence starts: {len(buffer)} | action_dim={action_dim}")
    print(format_reward_summary(reward_summary))
    print(f"[INFO] obs_mode={args.obs_mode} | lstm_seq_len={args.lstm_seq_len} | updates_per_epoch={args.updates_per_epoch}")
    if not args.no_eval and args.eval_every > 0:
        print(
            f"[INFO] Isaac Sim eval every {args.eval_every} epochs "
            f"({args.eval_episodes} episodes, task={args.eval_task})"
        )
    print(f"[INFO] Logging to: {run_dir}")

    eval_metrics_dir = os.path.join(run_dir, "sim_eval")
    os.makedirs(eval_metrics_dir, exist_ok=True)
    best_success_rate = -1.0
    best_ckpt_path = os.path.join(run_dir, "checkpoint_best.pt")

    avg_actor = deque(maxlen=20)
    avg_critic = deque(maxlen=20)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        epoch_actor = 0.0
        epoch_critic = 0.0
        t0 = time.time()

        for _ in range(args.updates_per_epoch):
            batch = buffer.sample(args.batch_size, device)
            stats = agent.learn(batch)
            global_step += 1
            epoch_actor += stats.actor_loss
            epoch_critic += 0.5 * (stats.critic1_loss + stats.critic2_loss)
            avg_actor.append(stats.actor_loss)
            avg_critic.append(0.5 * (stats.critic1_loss + stats.critic2_loss))

        epoch_actor /= args.updates_per_epoch
        epoch_critic /= args.updates_per_epoch

        if epoch % args.log_every == 0:
            os.makedirs(tb_dir, exist_ok=True)
            writer.add_scalar("loss/actor", epoch_actor, epoch)
            writer.add_scalar("loss/critic", epoch_critic, epoch)
            writer.add_scalar("loss/cql1", stats.cql1_loss, epoch)
            writer.add_scalar("loss/cql2", stats.cql2_loss, epoch)
            writer.add_scalar("q/mean_q1", stats.q1_mean, epoch)
            writer.add_scalar("q/mean_q2", stats.q2_mean, epoch)
            writer.add_scalar("alpha", stats.alpha, epoch)
            writer.add_scalar("time/epoch_sec", time.time() - t0, epoch)
            print(
                f"[Epoch {epoch:04d}] actor={epoch_actor:.4f} critic={epoch_critic:.4f} "
                f"q1={stats.q1_mean:.3f} alpha={stats.alpha:.4f} "
                f"avg20_actor={np.mean(avg_actor):.4f} avg20_critic={np.mean(avg_critic):.4f}"
            )

        if args.eval_every > 0 and not args.no_eval and epoch % args.eval_every == 0:
            eval_ckpt_path = os.path.join(run_dir, "checkpoint_eval_latest.pt")
            torch.save(
                pack_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    agent=agent,
                    action_stats=action_stats,
                    args=vars(args),
                ),
                eval_ckpt_path,
            )
            eval_json_path = os.path.join(eval_metrics_dir, f"metrics_epoch_{epoch:04d}.json")
            eval_device = str(device)
            if device.type == "cuda":
                agent.to("cpu")
                torch.cuda.empty_cache()
            try:
                sim_metrics = run_isaac_sim_eval(
                    checkpoint_path=eval_ckpt_path,
                    output_json=eval_json_path,
                    isaaclab_dir=args.isaaclab_dir,
                    eval_episodes=args.eval_episodes,
                    eval_task=args.eval_task,
                    device=eval_device,
                    episode_length_s=args.eval_episode_length_s,
                    position_threshold=args.eval_position_threshold,
                    num_envs=args.eval_num_envs,
                    headless=args.eval_headless,
                )
            except RuntimeError as exc:
                print(f"[WARN] Isaac Sim eval failed at epoch {epoch}: {exc}")
                sim_metrics = None
            finally:
                if device.type == "cuda":
                    agent.to(device)
            if sim_metrics is not None:
                writer.add_scalar("eval/success_rate", sim_metrics.success_rate, epoch)
                writer.add_scalar("eval/grasp_rate", sim_metrics.grasp_rate, epoch)
                writer.add_scalar("eval/mean_episode_length", sim_metrics.mean_episode_length, epoch)
                writer.add_scalar("eval/mean_final_cube_to_goal", sim_metrics.mean_final_cube_to_goal, epoch)
                writer.add_scalar("eval/num_episodes", float(sim_metrics.num_episodes), epoch)
                print(
                    f"[SimEval {epoch:04d}] lift_success={sim_metrics.success_rate:.3f} "
                    f"grasp_success={sim_metrics.grasp_rate:.3f} "
                    f"len={sim_metrics.mean_episode_length:.1f} "
                    f"cube_to_goal={sim_metrics.mean_final_cube_to_goal:.4f} "
                    f"({sim_metrics.num_episodes} eps)"
                )
                if sim_metrics.success_rate >= best_success_rate:
                    best_success_rate = sim_metrics.success_rate
                    shutil.copy2(eval_ckpt_path, best_ckpt_path)
                    print(f"[INFO] New best checkpoint: {best_ckpt_path} (success={best_success_rate:.3f})")

        if epoch % args.save_every == 0:
            ckpt_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch:04d}.pt")
            torch.save(
                pack_checkpoint(
                    epoch=epoch,
                    global_step=global_step,
                    agent=agent,
                    action_stats=action_stats,
                    args=vars(args),
                ),
                ckpt_path,
            )
            print(f"[INFO] Saved checkpoint: {ckpt_path}")

    final_path = os.path.join(run_dir, "checkpoint_final.pt")
    torch.save(
        pack_checkpoint(
            epoch=args.epochs,
            global_step=global_step,
            agent=agent,
            action_stats=action_stats,
            args=vars(args),
        ),
        final_path,
    )
    writer.close()
    print(f"[INFO] Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
