#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG 训练脚本（对照版）
- 与 LSTM-DDPG(-Att) 保持“同环境配置”可控
- 训练日志写入：SR(100) + C/T(100) + 元信息（seed/learn_start/batch_size 等）
说明：
- 不改环境、不改 agent，实现 success/collision/timeout 仅依赖 env.step() 的 info["reason"] 推断
"""

from __future__ import annotations

import os
import time
import argparse
import random
from collections import deque
from datetime import datetime

import numpy as np
import torch

from config import DDPGConfig, EnvConfig, LOG_DIR, MODEL_DIR, RESULT_DIR
from environment import NavigationEnv
from ddpg import DDPGAgent
from utils import Logger, plot_training_curves


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 尽量可复现（会略影响速度）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infer_termination(info: dict, steps: int) -> tuple[bool, bool, bool, str]:
    """
    统一终止条件推断（不改环境）
    Returns: (success, collision, timeout, reason)
    """
    reason = (info or {}).get("reason", "") or ""
    success = (reason == "goal_reached")
    collision = (reason in ("collision_obstacle", "collision_wall"))
    timeout = (reason == "max_steps")

    # 兜底：如果 reason 为空但回合已结束（理论上不会发生）
    if (not success) and (not collision) and (not timeout):
        if steps >= EnvConfig.MAX_STEPS:
            timeout = True
            reason = "max_steps"
    return success, collision, timeout, reason


def train(args: argparse.Namespace) -> None:
    set_global_seed(args.seed)

    print("=" * 60)
    print("DDPG Navigation Training (Compare)")
    print("=" * 60)

    dyn_patterns = tuple([s.strip() for s in args.dynamic_patterns.split(",") if s.strip()])
    enhanced_cfg = {
        "n_sectors": args.n_sectors,
        "sector_method": args.sector_method,
        "use_lidar_diff": (not args.disable_lidar_diff),
        "use_delta_yaw": (not args.disable_delta_yaw),
    }

    env = NavigationEnv(
        use_enhanced_state=(not args.legacy_state),
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=args.dynamic_speed_min,
        dynamic_speed_max=args.dynamic_speed_max,
        dynamic_patterns=dyn_patterns,
        dynamic_stop_prob=args.dynamic_stop_prob,
    )

    print("Environment created")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Map size: {EnvConfig.MAP_WIDTH} x {EnvConfig.MAP_HEIGHT}")
    if not args.legacy_state:
        print("\nEnhanced state:")
        print(f"  Sectors: {args.n_sectors} ({args.sector_method})")
        print(f"  use_lidar_diff: {not args.disable_lidar_diff}")
        print(f"  use_delta_yaw: {not args.disable_delta_yaw}")
    print(f"\nDynamic obstacles: {args.dynamic_speed_min:.2f}~{args.dynamic_speed_max:.2f} m/s")
    print(f"  Patterns: {dyn_patterns}")
    print(f"\nSeed: {args.seed}")

    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
    )

    # Resume
    start_episode = 0
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        try:
            base = os.path.basename(args.resume)
            if "_ep" in base:
                start_episode = int(base.split("_ep")[-1].split(".")[0])
        except Exception:
            start_episode = 0
        print(f"Resumed from {args.resume}, starting at episode {start_episode}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(os.path.join(LOG_DIR, f"ddpg_training_{timestamp}.csv"))

    total_steps = 0
    learn_count = 0
    best_avg_reward = -float("inf")

    episode_rewards: list[float] = []
    episode_steps: list[int] = []

    # rolling window (100)
    win_s = deque(maxlen=100)
    win_c = deque(maxlen=100)
    win_t = deque(maxlen=100)

    succ_all = 0
    coll_all = 0
    tout_all = 0

    train_t0 = time.time()
    ep_times: list[float] = []

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    for episode in range(start_episode, args.num_episodes):
        ep_t0 = time.time()

        state = env.reset()
        agent.reset_noise()

        ep_reward = 0.0
        info = {}

        for step in range(EnvConfig.MAX_STEPS):
            total_steps += 1

            action = agent.act(state, step=total_steps, add_noise=True)
            action_flat = action.reshape(-1)

            next_state, reward, done, info = env.step(action_flat)

            agent.step(state, action_flat, reward, next_state, done)

            if total_steps > args.learn_start:
                agent.learn()
                learn_count += 1

            ep_reward += float(reward)
            state = next_state

            if done:
                break

        steps = step + 1
        success, collision, timeout, reason = infer_termination(info, steps)

        # stats
        ep_time = time.time() - ep_t0
        ep_times.append(ep_time)
        episode_rewards.append(ep_reward)
        episode_steps.append(steps)

        succ_all += int(success)
        coll_all += int(collision)
        tout_all += int(timeout)

        win_s.append(int(success))
        win_c.append(int(collision))
        win_t.append(int(timeout))

        avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
        sr100 = float(np.mean(win_s) * 100.0) if len(win_s) > 0 else 0.0
        c100 = float(np.mean(win_c) * 100.0) if len(win_c) > 0 else 0.0
        t100 = float(np.mean(win_t) * 100.0) if len(win_t) > 0 else 0.0
        sr_all = succ_all / (episode + 1) * 100.0

        logger.log({
            "episode": episode + 1,
            "reward": ep_reward,
            "steps": steps,
            "avg_reward": avg_reward,
            "success": int(success),
            "collision": int(collision),
            "timeout": int(timeout),
            "success_rate": sr_all,         # 兼容字段：累计成功率
            "sr100": sr100,
            "coll100": c100,
            "to100": t100,
            "total_steps": total_steps,
            "learn_count": learn_count,
            "time": ep_time,
            # --- meta (用于ZIP自动对齐) ---
            "seed": int(args.seed),
            "learn_start": int(args.learn_start),
            "batch_size": int(args.batch_size),
            "history_len": 1,
            # --- 统一字段（便于与Att脚本对齐）---
            "att_drop_cur": float("nan"),
            "att_temp_cur": float("nan"),
            "att_tau_cur": float("nan"),
            "use_spatial_att": 0,
            "use_temporal_att": 0,
        })

        if (episode + 1) % args.print_every == 0:
            avg_ep_time = float(np.mean(ep_times[-100:])) if ep_times else ep_time
            remaining = args.num_episodes - episode - 1
            eta_h = (remaining * avg_ep_time) / 3600.0

            print(
                f"Ep {episode + 1:5d} | "
                f"R: {ep_reward:7.1f} | "
                f"Avg: {avg_reward:7.1f} | "
                f"Steps: {steps:3d} | "
                f"SR(100): {sr100:5.1f}% | "
                f"C/T(100): {c100:4.1f}/{t100:4.1f}% | "
                f"SR(all): {sr_all:5.1f}% | "
                f"T: {ep_time:.2f}s | "
                f"ETA: {eta_h:.1f}h | "
                f"{reason}"
            )

        # save best by avg_reward (after warmup)
        if (episode + 1) > max(100, start_episode + 1) and avg_reward > best_avg_reward:
            best_avg_reward = avg_reward
            agent.save(os.path.join(MODEL_DIR, "ddpg_best.pth"))
            print(f"  -> New best! Avg reward: {best_avg_reward:.2f}")

        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(MODEL_DIR, f"ddpg_ep{episode + 1}.pth"))

    total_time = time.time() - train_t0

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Total learn updates: {learn_count}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Avg time/episode: {total_time/args.num_episodes:.2f}s")
    print(f"Success rate: {succ_all / args.num_episodes * 100:.1f}%")
    print(f"Collision rate: {coll_all / args.num_episodes * 100:.1f}%")
    print(f"Timeout rate: {tout_all / args.num_episodes * 100:.1f}%")
    print(f"Best avg reward: {best_avg_reward:.2f}")

    agent.save(os.path.join(MODEL_DIR, "ddpg_final.pth"))

    # 仍调用原 plot（对照即可；精确SR请以CSV的sr100/coll100/to100为准）
    plot_training_curves(
        episode_rewards,
        episode_steps,
        save_path=os.path.join(RESULT_DIR, f"ddpg_training_curves_{timestamp}.png"),
    )

    logger.close()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DDPG Navigation Training (Compare)")

    # 基础训练参数
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_episodes", type=int, default=DDPGConfig.NUM_EPISODES)
    p.add_argument("--save_interval", type=int, default=DDPGConfig.SAVE_INTERVAL)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--learn_start", type=int, default=DDPGConfig.LEARN_START)
    p.add_argument("--batch_size", type=int, default=DDPGConfig.BATCH_SIZE)
    p.add_argument("--buffer_size", type=int, default=DDPGConfig.BUFFER_SIZE)
    p.add_argument("--hidden_dim", type=int, default=DDPGConfig.HIDDEN_DIM)
    p.add_argument("--print_every", type=int, default=10)

    # 环境/状态配置（默认与Att脚本一致）
    p.add_argument("--legacy_state", action="store_true", help="Use original 39D state.")
    p.add_argument("--n_sectors", type=int, default=16, choices=[8, 16])
    p.add_argument("--sector_method", type=str, default="min", choices=["min", "mean"])
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    # 动态障碍物（默认与Att脚本一致：0.30~0.70 + bounce/random_walk）
    p.add_argument("--dynamic_speed_min", type=float, default=0.30)
    p.add_argument("--dynamic_speed_max", type=float, default=0.70)
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
