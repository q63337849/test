#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_lstm_ddpg.py - 优化版

加速优化：
1. UPDATE_EVERY=4: 每4步更新一次（最重要！）
2. HISTORY_LEN=5: 减少序列长度
3. 更小的网络: embed_dim=64, lstm_hidden_dim=64

用法：
  python train_lstm_ddpg.py --num_episodes 5000
  python train_lstm_ddpg.py --history_len 5 --update_every 4
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from datetime import datetime

import numpy as np

from config import DDPGConfig, EnvConfig, LOG_DIR, MODEL_DIR, RESULT_DIR
from environment import NavigationEnv
from lstm_ddpg import LSTMDdpgAgent
from utils import Logger, plot_training_curves


def train(args: argparse.Namespace) -> None:
    print("=" * 60)
    print("LSTM-DDPG Navigation Training (Optimized)")
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
    
    print("\n⚡ Speed Optimizations:")
    print(f"  History len: {args.history_len} (原10)")
    print(f"  Update every: {args.update_every} steps (原1)")
    print(f"  Embed dim: {args.embed_dim} (原128)")
    print(f"  LSTM hidden: {args.lstm_hidden_dim} (原128)")
    
    if not args.legacy_state:
        print(f"\nEnhanced state:")
        print(f"  Sectors: {args.n_sectors} ({args.sector_method})")
        print(f"  use_lidar_diff: {not args.disable_lidar_diff}")
        print(f"  use_delta_yaw: {not args.disable_delta_yaw}")
    
    print(f"\nDynamic obstacles: {args.dynamic_speed_min:.2f}~{args.dynamic_speed_max:.2f} m/s")
    print(f"  Patterns: {dyn_patterns}")

    agent = LSTMDdpgAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        history_len=args.history_len,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        embed_dim=args.embed_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        update_every=args.update_every,
        update_times=args.update_times,
    )
    
    agent.state_meta = {
        'legacy_state': bool(args.legacy_state),
        'n_sectors': int(args.n_sectors),
        'sector_method': str(args.sector_method),
        'disable_lidar_diff': bool(args.disable_lidar_diff),
        'disable_delta_yaw': bool(args.disable_delta_yaw),
        'dynamic_speed_min': float(args.dynamic_speed_min),
        'dynamic_speed_max': float(args.dynamic_speed_max),
        'dynamic_patterns': str(args.dynamic_patterns),
        'dynamic_stop_prob': float(args.dynamic_stop_prob),
        'history_len': int(args.history_len),
        'embed_dim': int(args.embed_dim),
        'lstm_hidden_dim': int(args.lstm_hidden_dim),
        'update_every': int(args.update_every),
    }

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
    logger = Logger(os.path.join(LOG_DIR, f"lstm_ddpg_training_{timestamp}.csv"))

    total_steps = 0
    learn_count = 0
    best_reward = -float("inf")

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    # 时间统计
    train_start_time = time.time()
    episode_times: list[float] = []

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    for episode in range(start_episode, args.num_episodes):
        ep_t0 = time.time()

        state = env.reset()
        agent.reset_noise()

        state_queue = deque([state.copy() for _ in range(args.history_len)], maxlen=args.history_len)

        episode_reward = 0.0

        for step in range(EnvConfig.MAX_STEPS):
            total_steps += 1

            state_seq = np.asarray(state_queue, dtype=np.float32)
            action = agent.act(state_seq, step=total_steps, add_noise=True)
            action_flat = action.reshape(-1)

            next_state, reward, done, info = env.step(action_flat)

            agent.step(state, action_flat, reward, next_state, done)

            state_queue.append(next_state.copy())

            # ⚡ 关键优化：使用 should_learn() 代替每步更新
            if total_steps > args.learn_start and agent.should_learn():
                agent.learn()
                learn_count += 1

            episode_reward += float(reward)
            state = next_state

            if done:
                break

        episode_time = time.time() - ep_t0
        episode_times.append(episode_time)
        episode_rewards.append(episode_reward)
        episode_steps.append(step + 1)

        success, _ = env.get_episode_status()
        if success:
            success_count += 1
        elif info.get("reason") in ("collision_obstacle", "collision_wall"):
            collision_count += 1
        else:
            timeout_count += 1

        avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
        success_rate = success_count / (episode + 1) * 100.0

        logger.log({
            "episode": episode + 1,
            "reward": episode_reward,
            "steps": step + 1,
            "avg_reward": avg_reward,
            "success": int(success),
            "collision": int(info.get("reason") in ("collision_obstacle", "collision_wall")),
            "timeout": int(info.get("reason") == "max_steps"),
            "success_rate": success_rate,
            "total_steps": total_steps,
            "learn_count": learn_count,
            "time": episode_time,
            "history_len": args.history_len,
        })

        if (episode + 1) % 10 == 0:
            avg_ep_time = np.mean(episode_times[-100:])
            remaining_eps = args.num_episodes - episode - 1
            eta_seconds = remaining_eps * avg_ep_time
            eta_hours = eta_seconds / 3600
            
            print(
                f"Ep {episode + 1:5d} | "
                f"R: {episode_reward:7.1f} | "
                f"Avg: {avg_reward:7.1f} | "
                f"Steps: {step + 1:3d} | "
                f"SR: {success_rate:5.1f}% | "
                f"T: {episode_time:.2f}s | "
                f"ETA: {eta_hours:.1f}h | "
                f"{info.get('reason', '')[:8]}"
            )

        if avg_reward > best_reward and (episode + 1) > 100:
            best_reward = avg_reward
            agent.save(os.path.join(MODEL_DIR, "lstm_ddpg_best.pth"))
            print(f"  -> New best! Avg reward: {best_reward:.2f}")

        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(MODEL_DIR, f"lstm_ddpg_ep{episode + 1}.pth"))

    total_time = time.time() - train_start_time
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Total learn updates: {learn_count}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Avg time/episode: {total_time/args.num_episodes:.2f}s")
    print(f"Success rate: {success_count / args.num_episodes * 100:.1f}%")
    print(f"Collision rate: {collision_count / args.num_episodes * 100:.1f}%")
    print(f"Timeout rate: {timeout_count / args.num_episodes * 100:.1f}%")
    print(f"Best avg reward: {best_reward:.2f}")

    agent.save(os.path.join(MODEL_DIR, "lstm_ddpg_final.pth"))

    plot_training_curves(
        episode_rewards,
        episode_steps,
        save_path=os.path.join(RESULT_DIR, f"lstm_ddpg_training_curves_{timestamp}.png"),
    )

    logger.close()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM-DDPG Navigation Training (Optimized)")
    p.add_argument("--num_episodes", type=int, default=DDPGConfig.NUM_EPISODES)
    p.add_argument("--save_interval", type=int, default=DDPGConfig.SAVE_INTERVAL)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--learn_start", type=int, default=DDPGConfig.LEARN_START)
    p.add_argument("--batch_size", type=int, default=DDPGConfig.BATCH_SIZE)
    p.add_argument("--buffer_size", type=int, default=DDPGConfig.BUFFER_SIZE)

    # ⚡ 加速参数
    p.add_argument("--history_len", type=int, default=DDPGConfig.HISTORY_LEN,
                   help="History length for LSTM (default: 5, was 10)")
    p.add_argument("--embed_dim", type=int, default=DDPGConfig.LSTM_EMBED_DIM,
                   help="LSTM embedding dim (default: 64, was 128)")
    p.add_argument("--lstm_hidden_dim", type=int, default=DDPGConfig.LSTM_HIDDEN_DIM,
                   help="LSTM hidden dim (default: 64, was 128)")
    p.add_argument("--update_every", type=int, default=DDPGConfig.UPDATE_EVERY,
                   help="Learn every N steps (default: 4, was 1)")
    p.add_argument("--update_times", type=int, default=DDPGConfig.UPDATE_TIMES,
                   help="Number of updates per learn call (default: 1)")

    # 环境/状态配置
    p.add_argument("--legacy_state", action="store_true", help="Use original 39D state.")
    p.add_argument("--n_sectors", type=int, default=16, choices=[8, 16])
    p.add_argument("--sector_method", type=str, default="min", choices=["min", "mean"])
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    # 动态障碍物
    p.add_argument("--dynamic_speed_min", type=float, default=0.10)
    p.add_argument("--dynamic_speed_max", type=float, default=0.35)
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)
    
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
