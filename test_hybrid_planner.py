#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""混合航迹规划测试脚本。

用途：
1) 不加载模型时：测试 MIDBO + 规则局部避障兜底控制器。
2) 加载 attention 模型时：测试 MIDBO + LSTM-DDPG-Attention 真正混合系统。

示例：
- 仅做快速冒烟
  python test_hybrid_planner.py --episodes 3 --max_steps 200

- 使用训练好的 LSTM-DDPG-Attention 模型
  python test_hybrid_planner.py --episodes 20 --att_model models/lstm_ddpg_att_best.pth
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from environment import NavigationEnv
from hybrid_path_planner import HybridConfig, HybridTrajectoryPlanner
from config import EnvConfig


def _build_local_policy(att_model: str, state_dim: int, action_dim: int):
    """按需加载 LSTM-DDPG-Attention 策略。"""
    if not att_model:
        return None

    from lstm_ddpg_att import LSTMDdpgAgent as LSTMDdpgAttAgent

    agent = LSTMDdpgAttAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        history_len=5,
        embed_dim=64,
        lstm_hidden_dim=64,
        use_spatial_att=True,
        use_temporal_att=True,
        sector_model_dim=32,
        spatial_att_heads=4,
        temporal_att_dim=64,
        temporal_att_heads=4,
    )
    agent.load(att_model, strict=False, load_optimizers=False)
    return agent


def _apply_env_overrides(args: argparse.Namespace) -> None:
    """将测试参数写入 EnvConfig，便于构造 50m×50m 等大规模测试环境。"""
    EnvConfig.MAP_WIDTH = float(args.map_width)
    EnvConfig.MAP_HEIGHT = float(args.map_height)
    EnvConfig.NUM_STATIC_OBSTACLES = int(args.num_static_obstacles)
    EnvConfig.NUM_DYNAMIC_OBSTACLES = int(args.num_dynamic_obstacles)
    EnvConfig.MAX_STEPS = int(args.max_steps)
    EnvConfig.LIDAR_MAX_RANGE = float(args.lidar_max_range)
    EnvConfig.GOAL_RADIUS = float(args.goal_radius)


def run(args: argparse.Namespace) -> None:
    np.random.seed(args.seed)
    _apply_env_overrides(args)

    env = NavigationEnv(
        use_enhanced_state=True,
        enhanced_state_config={
            "n_sectors": 16,
            "sector_method": "min",
            "use_lidar_diff": True,
            "use_delta_yaw": True,
        },
        dynamic_speed_min=args.dynamic_speed_min,
        dynamic_speed_max=args.dynamic_speed_max,
        dynamic_patterns=("bounce", "random_walk"),
        dynamic_stop_prob=0.05,
    )

    local_policy = _build_local_policy(args.att_model, env.state_dim, env.action_dim)
    cfg = HybridConfig(
        midbo_population=args.midbo_population,
        midbo_iterations=args.midbo_iterations,
        waypoint_count=args.waypoint_count,
        max_episode_steps=args.max_steps,
    )

    planner = HybridTrajectoryPlanner(
        env=env,
        local_policy=local_policy,
        config=cfg,
        random_state=args.seed,
    )

    results = []
    reason_counter = Counter()

    print("=" * 72)
    print("Test Environment Design")
    print("=" * 72)
    print(f"map: {EnvConfig.MAP_WIDTH:.1f}m x {EnvConfig.MAP_HEIGHT:.1f}m")
    print(f"static_obstacles: {EnvConfig.NUM_STATIC_OBSTACLES}")
    print(f"dynamic_obstacles: {EnvConfig.NUM_DYNAMIC_OBSTACLES}")
    print(f"dynamic_speed: [{args.dynamic_speed_min:.2f}, {args.dynamic_speed_max:.2f}] m/s")
    print(f"lidar: rays={EnvConfig.LIDAR_RAYS}, range={EnvConfig.LIDAR_MAX_RANGE:.1f}m, fov={EnvConfig.LIDAR_FOV}°")
    print(f"goal_radius: {EnvConfig.GOAL_RADIUS:.2f}m")
    print(f"control_dt: {EnvConfig.DT:.2f}s, max_steps={EnvConfig.MAX_STEPS}")
    print(f"planner: MIDBO(pop={args.midbo_population}, iter={args.midbo_iterations}, wp={args.waypoint_count})")
    print("=" * 72)

    for ep in range(args.episodes):
        planner.global_path = np.empty((0, 2), dtype=np.float32)
        out = planner.run_episode(reset=True)
        results.append(out)
        reason_counter[out["reason"]] += 1

        print(
            f"[EP {ep + 1:03d}] success={out['success']} steps={out['steps']} "
            f"reason={out['reason']} mode_end={out['mode']} "
            f"replan={out['stats'].replan_count} trigger={out['stats'].trigger_count}"
        )

    success_rate = 100.0 * np.mean([r["success"] for r in results])
    avg_steps = float(np.mean([r["steps"] for r in results]))
    avg_reward = float(np.mean([r["reward"] for r in results]))
    avg_min_front = float(np.mean([r["trace"]["min_lidar_front"] for r in results]))
    avg_max_dev = float(np.mean([r["trace"]["max_deviation"] for r in results]))

    print("\n" + "=" * 72)
    print("Hybrid Planner Evaluation Summary")
    print("=" * 72)
    print(f"episodes: {args.episodes}")
    print(f"local_policy: {'LSTM-DDPG-Attention' if local_policy is not None else 'Rule-based fallback'}")
    print(f"success_rate: {success_rate:.1f}%")
    print(f"avg_steps: {avg_steps:.1f}")
    print(f"avg_reward: {avg_reward:.2f}")
    print(f"avg_min_lidar_front: {avg_min_front:.3f} m")
    print(f"avg_max_deviation: {avg_max_dev:.3f} m")
    print(f"terminal_reasons: {dict(reason_counter)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    # 测试环境设计参数（默认按文档的大规模设置）
    parser.add_argument("--map_width", type=float, default=50.0)
    parser.add_argument("--map_height", type=float, default=50.0)
    parser.add_argument("--num_static_obstacles", type=int, default=40)
    parser.add_argument("--num_dynamic_obstacles", type=int, default=20)
    parser.add_argument("--lidar_max_range", type=float, default=5.0)
    parser.add_argument("--goal_radius", type=float, default=0.30)

    parser.add_argument("--att_model", type=str, default="")

    parser.add_argument("--dynamic_speed_min", type=float, default=0.30)
    parser.add_argument("--dynamic_speed_max", type=float, default=0.50)

    parser.add_argument("--midbo_population", type=int, default=20)
    parser.add_argument("--midbo_iterations", type=int, default=80)
    parser.add_argument("--waypoint_count", type=int, default=8)

    run(parser.parse_args())
