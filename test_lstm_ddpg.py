#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_lstm_ddpg.py

测试 LSTM-DDPG（窗口化序列输入）。

与 test.py 的差异：Actor 输入为 (H, state_dim) 的状态序列窗口；测试时不加探索噪声。
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch

from config import DDPGConfig, EnvConfig, MODEL_DIR
from environment import NavigationEnv
from lstm_ddpg import LSTMDdpgAgent
from visualizer import Visualizer


def _infer_state_dim_from_checkpoint(ckpt: dict) -> int | None:
    actor_sd = ckpt.get("actor_local", {})
    w = actor_sd.get("fc_in.weight", None)
    if w is not None:
        return int(w.shape[1])
    net_cfg = ckpt.get("net_cfg", {})
    if isinstance(net_cfg, dict) and "state_dim" in net_cfg:
        return int(net_cfg["state_dim"])
    return None


def _candidates_from_state_dim(state_dim: int) -> list[dict]:
    """Enumerate plausible enhanced-state configs that match a given state_dim.

    Our enhanced-state dimension is:
      n + (n if use_lidar_diff else 0) + 3 + 2 + 2 + 1 + (1 if use_delta_yaw else 0) + 2
    """
    cands: list[dict] = []
    for n in (8, 16):
        for use_diff in (False, True):
            for use_dyaw in (False, True):
                dim = n + (n if use_diff else 0) + 3 + 2 + 2 + 1 + (1 if use_dyaw else 0) + 2
                if dim == state_dim:
                    cands.append({
                        "n_sectors": n,
                        "disable_lidar_diff": (not use_diff),
                        "disable_delta_yaw": (not use_dyaw),
                    })
    return cands


def _apply_state_meta_to_args(args: argparse.Namespace, meta: dict) -> None:
    """Override args using checkpoint meta (if present)."""
    if not isinstance(meta, dict):
        return
    # state flags
    if "legacy_state" in meta:
        args.legacy_state = bool(meta["legacy_state"])
    if "n_sectors" in meta:
        args.n_sectors = int(meta["n_sectors"])
    if "sector_method" in meta:
        args.sector_method = str(meta["sector_method"])
    if "disable_lidar_diff" in meta:
        args.disable_lidar_diff = bool(meta["disable_lidar_diff"])
    if "disable_delta_yaw" in meta:
        args.disable_delta_yaw = bool(meta["disable_delta_yaw"])
    if "history_len" in meta:
        args.history_len = int(meta["history_len"])
    # 网络参数
    if "embed_dim" in meta:
        args.embed_dim = int(meta["embed_dim"])
    if "lstm_hidden_dim" in meta:
        args.lstm_hidden_dim = int(meta["lstm_hidden_dim"])
    # dynamics
    for k in ("dynamic_speed_min", "dynamic_speed_max", "dynamic_patterns", "dynamic_stop_prob"):
        if k in meta:
            setattr(args, k, meta[k])

def test(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("LSTM-DDPG Navigation Testing")
    print("=" * 60)

    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return

    # --- Load checkpoint FIRST to ensure state config matches ---
    ckpt = torch.load(args.model, map_location="cpu")
    state_meta = ckpt.get("state_meta", {})
    if state_meta:
        _apply_state_meta_to_args(args, state_meta)

    expected_state_dim = _infer_state_dim_from_checkpoint(ckpt)

    # Build env with (possibly overridden) args
    dyn_patterns = tuple([s.strip() for s in str(args.dynamic_patterns).split(",") if s.strip()])
    enhanced_cfg = {
        "n_sectors": args.n_sectors,
        "sector_method": args.sector_method,
        "use_lidar_diff": (not args.disable_lidar_diff),
        "use_delta_yaw": (not args.disable_delta_yaw),
    }
    env = NavigationEnv(
        use_enhanced_state=(not args.legacy_state),
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=float(args.dynamic_speed_min),
        dynamic_speed_max=float(args.dynamic_speed_max),
        dynamic_patterns=dyn_patterns,
        dynamic_stop_prob=float(args.dynamic_stop_prob),
    )

    if expected_state_dim is not None and int(env.state_dim) != int(expected_state_dim):
        cands = _candidates_from_state_dim(int(expected_state_dim))
        print("[StateDimMismatch]")
        print(f"  checkpoint expects state_dim={expected_state_dim}")
        print(f"  current args/env gives state_dim={env.state_dim}")
        if cands:
            print("  plausible enhanced-state flags that match checkpoint:")
            for c in cands:
                print("   - --n_sectors {n_sectors} {ld} {dy}".format(
                    n_sectors=c['n_sectors'],
                    ld='--disable_lidar_diff' if c['disable_lidar_diff'] else '',
                    dy='--disable_delta_yaw' if c['disable_delta_yaw'] else '',
                ))
        print("  Please run test with the SAME state flags as training.")
        env.close()
        return

    # 从 checkpoint 获取网络参数
    net_cfg = ckpt.get("net_cfg", {})
    embed_dim = getattr(args, 'embed_dim', net_cfg.get('embed_dim', DDPGConfig.LSTM_EMBED_DIM))
    lstm_hidden_dim = getattr(args, 'lstm_hidden_dim', net_cfg.get('lstm_hidden_dim', DDPGConfig.LSTM_HIDDEN_DIM))
    
    agent = LSTMDdpgAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        history_len=args.history_len,
        embed_dim=embed_dim,
        lstm_hidden_dim=lstm_hidden_dim,
    )
    agent.load(args.model)
    print(f"Model loaded from {args.model}")

    visualizer = Visualizer(env) if args.render else None

    total_rewards: list[float] = []
    total_steps: list[int] = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    print(f"\nRunning {args.num_episodes} test episodes...")
    print("-" * 60)

    for ep in range(args.num_episodes):
        state = env.reset()
        state_queue = deque([state.copy() for _ in range(args.history_len)], maxlen=args.history_len)

        ep_reward = 0.0
        trajectory = [(env.robot.x, env.robot.y)]

        for step in range(EnvConfig.MAX_STEPS):
            state_seq = np.asarray(state_queue, dtype=np.float32)
            action = agent.act(state_seq, add_noise=False)
            action_flat = action.reshape(-1)

            next_state, reward, done, info = env.step(action_flat)
            ep_reward += float(reward)

            trajectory.append((env.robot.x, env.robot.y))
            if visualizer:
                visualizer.render(trajectory)
                time.sleep(0.02)

            state_queue.append(next_state.copy())
            state = next_state

            if done:
                break

        total_rewards.append(ep_reward)
        total_steps.append(step + 1)

        success, _ = env.get_episode_status()
        if success:
            success_count += 1
            result = "SUCCESS"
        elif info.get("reason") in ("collision_obstacle", "collision_wall"):
            collision_count += 1
            result = "COLLISION"
        else:
            timeout_count += 1
            result = "TIMEOUT"

        print(
            f"Episode {ep + 1:3d} | Reward: {ep_reward:8.2f} | Steps: {step + 1:4d} | Result: {result}"
        )

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.2f}")
    print(f"Success rate: {success_count / args.num_episodes * 100:.1f}%")
    print(f"Collision rate: {collision_count / args.num_episodes * 100:.1f}%")
    print(f"Timeout rate: {timeout_count / args.num_episodes * 100:.1f}%")

    if visualizer:
        visualizer.close()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM-DDPG Navigation Testing")
    p.add_argument(
        "--model",
        type=str,
        default=os.path.join(MODEL_DIR, "lstm_ddpg_best.pth"),
        help="Path to model checkpoint",
    )
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--render", action="store_true")
    p.add_argument("--history_len", type=int, default=DDPGConfig.HISTORY_LEN)
    p.add_argument("--embed_dim", type=int, default=DDPGConfig.LSTM_EMBED_DIM)
    p.add_argument("--lstm_hidden_dim", type=int, default=DDPGConfig.LSTM_HIDDEN_DIM)

    # 与 train_lstm_ddpg.py 保持一致，确保 state_dim 与环境动态配置一致
    p.add_argument("--legacy_state", action="store_true")
    p.add_argument("--n_sectors", type=int, default=16, choices=[8, 16])
    p.add_argument("--sector_method", type=str, default="min", choices=["min", "mean"])
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")
    p.add_argument("--dynamic_speed_min", type=float, default=0.3)
    p.add_argument("--dynamic_speed_max", type=float, default=0.7)
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(args)
