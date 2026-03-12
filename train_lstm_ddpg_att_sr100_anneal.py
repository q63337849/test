#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""train_lstm_ddpg_att.py - LSTM-DDPG + Attention（可运行版）

在 train_lstm_ddpg.py 的基础上：
- 默认使用 lstm_ddpg_att.py 的 Agent（空间注意力 + 时间注意力）
- 增加注意力开关与超参：
    --use_spatial_att / --no-use_spatial_att
    --use_temporal_att / --no-use_temporal_att
    --sector_model_dim, --spatial_att_heads
    --temporal_att_dim, --temporal_att_heads
    --att_dropout

示例：
  # 两种注意力都开（默认就是开）
  python train_lstm_ddpg_att.py --num_episodes 5000 --history_len 10 --n_sectors 16

  # 只开时间注意力（关空间注意力）
  python train_lstm_ddpg_att.py --no-use_spatial_att --use_temporal_att

  # 只开空间注意力（关时间注意力）
  python train_lstm_ddpg_att.py --use_spatial_att --no-use_temporal_att

  # 调整 heads / 维度
  python train_lstm_ddpg_att.py --sector_model_dim 64 --spatial_att_heads 8 --temporal_att_dim 128 --temporal_att_heads 8
"""

from __future__ import annotations

import argparse
import os
import time
import math
from collections import deque
from datetime import datetime

import numpy as np

from config import DDPGConfig, EnvConfig, LOG_DIR, MODEL_DIR, RESULT_DIR
from environment import NavigationEnv
from lstm_ddpg_att import LSTMDdpgAgent
from utils import Logger, plot_training_curves


# =========================
# Attention annealing utils
# =========================

def _lin_anneal(progress: float, start: float, end: float, p0: float, p1: float) -> float:
    # Linear anneal from start->end over progress in [p0, p1].
    if p1 <= p0:
        return end
    if progress <= p0:
        return start
    if progress >= p1:
        return end
    u = (progress - p0) / (p1 - p0)
    return start + u * (end - start)


def _apply_att_anneal(agent, att_drop: float, att_temp: float, att_tau: float) -> None:
    # Apply annealed hyperparams to attention modules (actor/critic + target nets if present).
    nets = []
    for name in ("actor_local", "critic_local", "actor_target", "critic_target"):
        net = getattr(agent, name, None)
        if net is not None:
            nets.append(net)

    # clamp safe ranges
    att_drop = float(max(0.0, min(0.5, att_drop)))
    att_temp = float(max(0.3, min(2.0, att_temp)))
    att_tau = float(max(1e-3, min(0.2, att_tau)))

    for net in nets:
        for m in net.modules():
            # (1) dropout on attention weights / FFN dropout (if present)
            for dn in ("attn_drop", "ffn_drop", "dropout"):
                d = getattr(m, dn, None)
                if d is not None and hasattr(d, "p"):
                    d.p = att_drop

            # (2) temperature: modules that expose log_temp (Parameter or tensor)
            if hasattr(m, "log_temp"):
                lt = getattr(m, "log_temp")
                # freeze once to avoid fighting with optimizer
                if hasattr(lt, "requires_grad") and getattr(lt, "requires_grad", False):
                    try:
                        lt.requires_grad_(False)
                    except Exception:
                        pass
                try:
                    lt.data.fill_(math.log(att_temp))
                except Exception:
                    pass

            # (3) smooth-valid tau: modules that expose tau
            if hasattr(m, "tau"):
                try:
                    m.tau = float(att_tau)
                except Exception:
                    pass



def _validate_att_args(args: argparse.Namespace) -> None:
    if args.use_spatial_att:
        if args.sector_model_dim % args.spatial_att_heads != 0:
            raise ValueError(
                f"sector_model_dim({args.sector_model_dim}) 必须能被 spatial_att_heads({args.spatial_att_heads}) 整除"
            )
    if args.use_temporal_att:
        if args.temporal_att_dim % args.temporal_att_heads != 0:
            raise ValueError(
                f"temporal_att_dim({args.temporal_att_dim}) 必须能被 temporal_att_heads({args.temporal_att_heads}) 整除"
            )


def train(args: argparse.Namespace) -> None:
    _validate_att_args(args)

    print("=" * 60)
    print("LSTM-DDPG + Attention Navigation Training")
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

    print("\n⚡ Core training params:")
    print(f"  Episodes: {args.num_episodes}")
    print(f"  History len: {args.history_len}")
    print(f"  Update every: {args.update_every} steps")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  LSTM hidden: {args.lstm_hidden_dim}")

    if not args.legacy_state:
        print("\nEnhanced state:")
        print(f"  Sectors: {args.n_sectors} ({args.sector_method})")
        print(f"  use_lidar_diff: {not args.disable_lidar_diff}")
        print(f"  use_delta_yaw: {not args.disable_delta_yaw}")

    print(f"\nDynamic obstacles: {args.dynamic_speed_min:.2f}~{args.dynamic_speed_max:.2f} m/s")
    print(f"  Patterns: {dyn_patterns}")
    if "stop_and_go" in dyn_patterns:
        print(f"  stop_and_go prob: {args.dynamic_stop_prob:.3f}")

    print("\nAttention config:")
    print(f"  use_spatial_att: {args.use_spatial_att}")
    print(f"  use_temporal_att: {args.use_temporal_att}")
    if args.use_spatial_att:
        print(f"  sector_model_dim: {args.sector_model_dim}")
        print(f"  spatial_att_heads: {args.spatial_att_heads}")
    if args.use_temporal_att:
        print(f"  temporal_att_dim: {args.temporal_att_dim}")
        print(f"  temporal_att_heads: {args.temporal_att_heads}")
    print(f"  att_dropout: {args.att_dropout}")



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
        use_spatial_att=args.use_spatial_att,
        use_temporal_att=args.use_temporal_att,
        sector_model_dim=args.sector_model_dim,
        spatial_att_heads=args.spatial_att_heads,
        temporal_att_dim=args.temporal_att_dim,
        temporal_att_heads=args.temporal_att_heads,
        att_dropout=args.att_dropout,
    )



    # 保存训练/状态元信息（用于 load 时核对）
    agent.state_meta = {
        "legacy_state": bool(args.legacy_state),
        "n_sectors": int(args.n_sectors),
        "sector_method": str(args.sector_method),
        "disable_lidar_diff": bool(args.disable_lidar_diff),
        "disable_delta_yaw": bool(args.disable_delta_yaw),
        "dynamic_speed_min": float(args.dynamic_speed_min),
        "dynamic_speed_max": float(args.dynamic_speed_max),
        "dynamic_patterns": str(args.dynamic_patterns),
        "dynamic_stop_prob": float(args.dynamic_stop_prob),
        "history_len": int(args.history_len),
        "embed_dim": int(args.embed_dim),
        "lstm_hidden_dim": int(args.lstm_hidden_dim),
        "update_every": int(args.update_every),
        "update_times": int(args.update_times),
        "use_spatial_att": bool(args.use_spatial_att),
        "use_temporal_att": bool(args.use_temporal_att),
        "sector_model_dim": int(args.sector_model_dim),
        "spatial_att_heads": int(args.spatial_att_heads),
        "temporal_att_dim": int(args.temporal_att_dim),
        "temporal_att_heads": int(args.temporal_att_heads),
        "att_dropout": float(args.att_dropout),
    }

    # Resume
    start_episode = 0
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        try:
            base = os.path.basename(args.resume)
            if "_ep" in base:
                start_episode = 0
        except Exception:
            start_episode = 0
        print(f"Resumed from {args.resume}, starting at episode {start_episode}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(os.path.join(LOG_DIR, f"lstm_ddpg_att_training_{timestamp}.csv"))

    total_steps = 0
    learn_count = 0
    best_reward = -float("inf")

    episode_rewards: list[float] = []
    episode_steps: list[int] = []
    success_count = 0
    collision_count = 0
    timeout_count = 0



    # Rolling window metrics
    recent_window = 100
    episode_buffer = []  # each: {'success':0/1, 'collision':0/1, 'timeout':0/1}
    train_start_time = time.time()
    episode_times: list[float] = []

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    for episode in range(start_episode, args.num_episodes):
        ep_t0 = time.time()

        # ---- Anneal attention dropout / temperature / tau (minimal, framework-preserving) ----
        progress = (episode / max(1, (args.num_episodes - 1)))  # in [0,1]

        # Dropout anneal: keep initial value early, then decay to 0 near the end
        # 建议：dropout 不要退火到 0，temp 不要退火到 0.8（太硬）
        att_drop_floor = 0.015  # 经验上 0.01~0.02 都可
        att_temp_floor = 0.88  # 经验上 0.85~0.90 都可

        att_drop_cur = _lin_anneal(progress, args.att_dropout, att_drop_floor, 0.40, 1.00)
        att_temp_cur = _lin_anneal(progress, 1.15, att_temp_floor, 0.20, 1.00)

        att_tau_cur = _lin_anneal(progress, 0.05, 0.03, 0.00, 0.80)  # tau 维持不变即可

        _apply_att_anneal(agent, att_drop_cur, att_temp_cur, att_tau_cur)

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

        is_collision = (not success) and (info.get("reason") in ("collision_obstacle", "collision_wall"))

        is_timeout = (not success) and (not is_collision)

        if success:
            success_count += 1
        elif is_collision:
            collision_count += 1
        else:
            timeout_count += 1

        episode_buffer.append({
            "success": int(success),
            "collision": int(is_collision),
            "timeout": int(is_timeout),
        })

        win = min(recent_window, len(episode_buffer))
        recent_eps = episode_buffer[-win:]
        sr100 = 100.0 * sum(e["success"] for e in recent_eps) / win
        coll100 = 100.0 * sum(e["collision"] for e in recent_eps) / win
        to100 = 100.0 * sum(e["timeout"] for e in recent_eps) / win

        avg_reward = float(np.mean(episode_rewards[-100:])) if episode_rewards else 0.0
        success_rate = success_count / (episode + 1) * 100.0

        logger.log(
            {
                "episode": episode + 1,
                "reward": episode_reward,
                "steps": step + 1,
                "avg_reward": avg_reward,
                "success": int(success),
                "collision": int(is_collision),
                "timeout": int(is_timeout),
                "success_rate": success_rate,
                "sr100": sr100,
                "coll100": coll100,
                "to100": to100,
                "total_steps": total_steps,
                "learn_count": learn_count,
                "time": episode_time,
                "history_len": args.history_len,
                "batch_size": args.batch_size,
                "learn_start": args.learn_start,
                "att_drop_cur": float(att_drop_cur),
                "att_temp_cur": float(att_temp_cur),
                "att_tau_cur": float(att_tau_cur),
                "use_spatial_att": int(args.use_spatial_att),
                "use_temporal_att": int(args.use_temporal_att),
            }
        )

        if (episode + 1) % 10 == 0:
            avg_ep_time = float(np.mean(episode_times[-100:]))
            remaining_eps = args.num_episodes - episode - 1
            eta_seconds = remaining_eps * avg_ep_time
            eta_hours = eta_seconds / 3600

            print(
                f"Ep {episode + 1:5d} | "
                f"R: {episode_reward:7.1f} | "
                f"Avg: {avg_reward:7.1f} | "
                f"Steps: {step + 1:3d} | "
                f"SR(100): {sr100:5.1f}% | "
                f"C/T(100): {coll100:4.1f}/{to100:4.1f}% | "
                f"SR(all): {success_rate:5.1f}% | "
                f"T: {episode_time:.2f}s | "
                f"ETA: {eta_hours:.1f}h | "
                f"{info.get('reason', '')[:12]}"
            )

        if avg_reward > best_reward and (episode + 1) > 100:
            best_reward = avg_reward
            agent.save(os.path.join(MODEL_DIR, "lstm_ddpg_att_best.pth"))
            print(f"  -> New best! Avg reward: {best_reward:.2f}")

        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(MODEL_DIR, f"lstm_ddpg_att_ep{episode + 1}.pth"))

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

    agent.save(os.path.join(MODEL_DIR, "lstm_ddpg_att_final.pth"))

    plot_training_curves(
        episode_rewards,
        episode_steps,
        save_path=os.path.join(RESULT_DIR, f"lstm_ddpg_att_training_curves_{timestamp}.png"),
    )

    logger.close()
    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM-DDPG + Attention Navigation Training")

    p.add_argument("--num_episodes", type=int, default=DDPGConfig.NUM_EPISODES)
    p.add_argument("--save_interval", type=int, default=DDPGConfig.SAVE_INTERVAL)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--learn_start", type=int, default=DDPGConfig.LEARN_START)
    p.add_argument("--batch_size", type=int, default=DDPGConfig.BATCH_SIZE)
    p.add_argument("--buffer_size", type=int, default=DDPGConfig.BUFFER_SIZE)

    # 训练速度相关
    p.add_argument("--history_len", type=int, default=DDPGConfig.HISTORY_LEN)
    p.add_argument("--embed_dim", type=int, default=DDPGConfig.LSTM_EMBED_DIM)
    p.add_argument("--lstm_hidden_dim", type=int, default=DDPGConfig.LSTM_HIDDEN_DIM)
    p.add_argument("--update_every", type=int, default=DDPGConfig.UPDATE_EVERY)
    p.add_argument("--update_times", type=int, default=DDPGConfig.UPDATE_TIMES)

    # 环境/状态配置
    p.add_argument("--legacy_state", action="store_true", help="Use original 39D state.")
    p.add_argument("--n_sectors", type=int, default=16, choices=[8, 16])
    p.add_argument("--sector_method", type=str, default="min", choices=["min", "mean"])
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    # 动态障碍物
    p.add_argument("--dynamic_speed_min", type=float, default=0.30)
    p.add_argument("--dynamic_speed_max", type=float, default=0.70)
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)

    # Attention 开关（支持 --use_xxx / --no-use_xxx）
    p.add_argument(
        "--use_spatial_att",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable beam/sector-level spatial attention.",
    )
    p.add_argument(
        "--use_temporal_att",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable/disable temporal attention pooling over LSTM outputs.",
    )

    # Attention 超参
    p.add_argument("--sector_model_dim", type=int, default=32, help="Spatial attention model dim (Dm)")
    p.add_argument("--spatial_att_heads", type=int, default=4, help="Spatial attention heads (Dm must be divisible)")
    p.add_argument("--temporal_att_dim", type=int, default=64, help="Temporal attention dim (Da)")
    p.add_argument("--temporal_att_heads", type=int, default=4, help="Temporal attention heads (Da must be divisible)")
    p.add_argument("--att_dropout", type=float, default=0.0, help="Dropout applied to attention weights")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
