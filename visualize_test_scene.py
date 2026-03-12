#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试场景可视化（独立脚本）

图例约定：
- 蓝色圆点：起点（机器人初始位置）
- 黑色空心圆：终点（目标点）
- 灰色圆点：静态障碍物（按真实半径）
- 红色圆点：动态障碍物（按真实半径）
- 红色线+箭头：动态障碍物轨迹与运动方向

支持按参数覆盖 config.py 中的障碍物数量、大小和速度。
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from config import EnvConfig
from environment import NavigationEnv


@dataclass
class ObstacleCfgBackup:
    n_static: int
    n_dynamic: int
    rmin: float
    rmax: float


class TempEnvObstacleCfg:
    def __init__(self, n_static: int, n_dynamic: int, rmin: float, rmax: float):
        self.new = ObstacleCfgBackup(int(n_static), int(n_dynamic), float(rmin), float(rmax))
        self.old = ObstacleCfgBackup(
            int(EnvConfig.NUM_STATIC_OBSTACLES),
            int(EnvConfig.NUM_DYNAMIC_OBSTACLES),
            float(EnvConfig.OBSTACLE_RADIUS_MIN),
            float(EnvConfig.OBSTACLE_RADIUS_MAX),
        )

    def __enter__(self):
        EnvConfig.NUM_STATIC_OBSTACLES = self.new.n_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = self.new.n_dynamic
        EnvConfig.OBSTACLE_RADIUS_MIN = self.new.rmin
        EnvConfig.OBSTACLE_RADIUS_MAX = self.new.rmax

    def __exit__(self, exc_type, exc, tb):
        EnvConfig.NUM_STATIC_OBSTACLES = self.old.n_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = self.old.n_dynamic
        EnvConfig.OBSTACLE_RADIUS_MIN = self.old.rmin
        EnvConfig.OBSTACLE_RADIUS_MAX = self.old.rmax


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def create_env(args: argparse.Namespace) -> NavigationEnv:
    return NavigationEnv(
        use_enhanced_state=True,
        enhanced_state_config={
            "n_sectors": 16,
            "sector_method": "min",
            "use_lidar_diff": True,
            "use_delta_yaw": True,
        },
        dynamic_speed_min=float(args.dynamic_speed_min),
        dynamic_speed_max=float(args.dynamic_speed_max),
        dynamic_patterns=tuple([x.strip() for x in str(args.dynamic_patterns).split(",") if x.strip()]),
        dynamic_stop_prob=float(args.dynamic_stop_prob),
    )


def advance_dyn_only(env: NavigationEnv, dyn_trajs: List[List[Tuple[float, float]]], dyn_indices: List[int], keep_steps: int) -> None:
    dt = float(EnvConfig.DT)
    w = float(EnvConfig.MAP_WIDTH)
    h = float(EnvConfig.MAP_HEIGHT)
    for k, idx in enumerate(dyn_indices):
        obs = env.obstacles[idx]
        obs.step(dt, w, h)
        dyn_trajs[k].append((float(obs.x), float(obs.y)))
        if len(dyn_trajs[k]) > keep_steps:
            dyn_trajs[k] = dyn_trajs[k][-keep_steps:]


def draw_scene(
    env: NavigationEnv,
    start_xy: Tuple[float, float],
    dyn_indices: List[int],
    dyn_trajs: List[List[Tuple[float, float]]],
    out_path: str,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.2), dpi=130)
    w, h = float(EnvConfig.MAP_WIDTH), float(EnvConfig.MAP_HEIGHT)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Test Scene Visualization (seed={seed})")

    # 地图边界
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color="black", linewidth=1.0)

    # 终点：黑色空心圆（目标点）
    ax.scatter([float(env.goal_x)], [float(env.goal_y)], s=170, facecolors='none', edgecolors='black', linewidths=1.8, zorder=8)

    # 起点：蓝色圆点（机器人初始位置）
    ax.scatter([start_xy[0]], [start_xy[1]], s=30, c="#1f77b4", marker='o', zorder=8)

    dyn_set = set(dyn_indices)

    # 障碍物：静态灰、动态红（按真实半径）
    for i, obs in enumerate(env.obstacles):
        x, y, r = float(obs.x), float(obs.y), float(obs.radius)
        if i in dyn_set:
            c, ec, a = "#d62728", "#d62728", 0.35
        else:
            c, ec, a = "0.7", "0.55", 0.75
        ax.add_patch(plt.Circle((x, y), radius=r, facecolor=c, edgecolor=ec, alpha=a, linewidth=1.0, zorder=4))

    # 动态障碍物：轨迹+方向
    for k, idx in enumerate(dyn_indices):
        traj = dyn_trajs[k]
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color="#ff6b6b", linewidth=1.2, alpha=0.9, zorder=5)

            dx = xs[-1] - xs[-2]
            dy = ys[-1] - ys[-2]
            norm = float(np.hypot(dx, dy))
            if norm > 1e-8:
                ux, uy = dx / norm, dy / norm
                ax.arrow(
                    xs[-1], ys[-1],
                    ux * 0.25, uy * 0.25,
                    head_width=0.06, head_length=0.08,
                    fc="#d62728", ec="#d62728",
                    linewidth=1.0, alpha=0.95,
                    length_includes_head=True, zorder=6,
                )

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="test_scene_visual.png")
    p.add_argument("--trail_steps", type=int, default=120, help="推进动态障碍物的步数")
    p.add_argument("--trail_keep_steps", type=int, default=30, help="轨迹保留最近N步")

    # 可配置：障碍物数量/大小/速度（默认参考 config.py）
    p.add_argument("--num_static", type=int, default=int(EnvConfig.NUM_STATIC_OBSTACLES))
    p.add_argument("--num_dynamic", type=int, default=int(EnvConfig.NUM_DYNAMIC_OBSTACLES))
    p.add_argument("--radius_min", type=float, default=float(EnvConfig.OBSTACLE_RADIUS_MIN))
    p.add_argument("--radius_max", type=float, default=float(EnvConfig.OBSTACLE_RADIUS_MAX))
    p.add_argument("--dynamic_speed_min", type=float, default=float(EnvConfig.DYNAMIC_OBS_VEL_MIN))
    p.add_argument("--dynamic_speed_max", type=float, default=float(EnvConfig.DYNAMIC_OBS_VEL_MAX))
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    with TempEnvObstacleCfg(args.num_static, args.num_dynamic, args.radius_min, args.radius_max):
        env = create_env(args)
        env.reset()

        start_xy = (float(env.robot.x), float(env.robot.y))

        dyn_indices = [i for i, o in enumerate(env.obstacles) if bool(getattr(o, "is_dynamic", False))]
        dyn_trajs: List[List[Tuple[float, float]]] = []
        for idx in dyn_indices:
            o = env.obstacles[idx]
            dyn_trajs.append([(float(o.x), float(o.y))])

        for _ in range(max(0, int(args.trail_steps) - 1)):
            advance_dyn_only(env, dyn_trajs, dyn_indices, keep_steps=int(args.trail_keep_steps))

        draw_scene(env, start_xy, dyn_indices, dyn_trajs, args.out, seed=args.seed)
        env.close()

    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
