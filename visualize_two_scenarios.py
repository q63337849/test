#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""visualize_two_scenarios.py

仅用于“场景展示”（不跑策略，不做航迹规划）。

- 场景1：训练场景（EnvConfig 默认障碍物数量）
- 场景2：在场景1基础上 +2 静态 +2 动态障碍物

输出方式：
- 默认弹窗显示
- --save_png 保存对比图
- --save_gif 保存 GIF（默认关闭）

GIF 说明：
- 通过 env.step(action=[0,0]) 驱动环境更新，使动态障碍物运动（机器人尽量保持静止）。
- 若某个场景提前 done（碰撞/终止），后续帧将保持最后状态，不再 step，避免 reset 造成场景变化。

兼容：
- NavigationEnv 目标字段为 goal_x/goal_y（本项目环境如此）；若存在 env.goal 也可兼容。
- Matplotlib 不同版本的 canvas 抽帧接口差异（buffer_rgba/print_to_buffer/tostring_*）。
"""

from __future__ import annotations

import argparse
import random
from typing import List, Tuple

import numpy as np

from config import EnvConfig
from environment import NavigationEnv


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class TempObstacleCounts:
    def __init__(self, n_static: int, n_dynamic: int):
        self.new_static = int(n_static)
        self.new_dynamic = int(n_dynamic)
        self.old_static = int(EnvConfig.NUM_STATIC_OBSTACLES)
        self.old_dynamic = int(EnvConfig.NUM_DYNAMIC_OBSTACLES)

    def __enter__(self):
        EnvConfig.NUM_STATIC_OBSTACLES = self.new_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = self.new_dynamic

    def __exit__(self, exc_type, exc, tb):
        EnvConfig.NUM_STATIC_OBSTACLES = self.old_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = self.old_dynamic


def build_env(
    legacy_state: bool,
    n_sectors: int,
    sector_method: str,
    use_lidar_diff: bool,
    use_delta_yaw: bool,
    dynamic_speed_min: float,
    dynamic_speed_max: float,
    dynamic_patterns: Tuple[str, ...],
    dynamic_stop_prob: float,
) -> NavigationEnv:
    enhanced_cfg = {
        "n_sectors": int(n_sectors),
        "sector_method": str(sector_method),
        "use_lidar_diff": bool(use_lidar_diff),
        "use_delta_yaw": bool(use_delta_yaw),
    }
    return NavigationEnv(
        use_enhanced_state=(not legacy_state),
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=float(dynamic_speed_min),
        dynamic_speed_max=float(dynamic_speed_max),
        dynamic_patterns=tuple(dynamic_patterns),
        dynamic_stop_prob=float(dynamic_stop_prob),
    )


def _get_goal_xy(env: NavigationEnv) -> Tuple[float, float]:
    if hasattr(env, "goal"):
        try:
            gx, gy = env.goal
            return float(gx), float(gy)
        except Exception:
            pass
    gx = float(getattr(env, "goal_x"))
    gy = float(getattr(env, "goal_y"))
    return gx, gy


def _iter_obstacles(env: NavigationEnv):
    # env.obstacles 在本项目里是 Obstacle 对象列表；为了兼容也支持 dict
    for obs in getattr(env, "obstacles", []):
        if isinstance(obs, dict):
            x = float(obs.get("x", 0.0))
            y = float(obs.get("y", 0.0))
            r = float(obs.get("radius", 0.2))
            is_dyn = bool(obs.get("is_dynamic", False))
        else:
            x = float(getattr(obs, "x"))
            y = float(getattr(obs, "y"))
            r = float(getattr(obs, "radius"))
            is_dyn = bool(getattr(obs, "is_dynamic", False))
        yield x, y, r, is_dyn


def _get_dynamic_heading(obs, traj: List[Tuple[float, float]] | None = None) -> Tuple[float, float]:
    """返回动态障碍物朝向向量（单位向量）。

    优先使用障碍物自身速度 vx/vy；若速度过小，再退化到轨迹末两点估计方向。
    """
    vx = float(getattr(obs, "vx", 0.0))
    vy = float(getattr(obs, "vy", 0.0))
    norm = float(np.hypot(vx, vy))
    if norm > 1e-8:
        return vx / norm, vy / norm

    if traj is not None and len(traj) >= 2:
        dx = float(traj[-1][0] - traj[-2][0])
        dy = float(traj[-1][1] - traj[-2][1])
        norm = float(np.hypot(dx, dy))
        if norm > 1e-8:
            return dx / norm, dy / norm

    return 0.0, 0.0


def init_dynamic_trajs(env: NavigationEnv) -> dict[int, List[Tuple[float, float]]]:
    """初始化动态障碍物轨迹。

    约定：key 使用 env.obstacles 列表中的索引（idx），因为本项目环境中障碍物列表在 episode 内不会重排。
    """
    trajs: dict[int, List[Tuple[float, float]]] = {}
    for idx, obs in enumerate(getattr(env, "obstacles", [])):
        is_dyn = False
        if isinstance(obs, dict):
            is_dyn = bool(obs.get("is_dynamic", False))
            x = float(obs.get("x", 0.0))
            y = float(obs.get("y", 0.0))
        else:
            is_dyn = bool(getattr(obs, "is_dynamic", False))
            x = float(getattr(obs, "x"))
            y = float(getattr(obs, "y"))
        if is_dyn:
            trajs[idx] = [(x, y)]
    return trajs


def advance_dynamic_obstacles(env: NavigationEnv, trajs: dict[int, List[Tuple[float, float]]]) -> None:
    """只推进动态障碍物（不推进机器人），并更新轨迹。"""
    dt = float(EnvConfig.DT)
    w = float(EnvConfig.MAP_WIDTH)
    h = float(EnvConfig.MAP_HEIGHT)
    for idx, obs in enumerate(getattr(env, "obstacles", [])):
        if isinstance(obs, dict):
            # dict 障碍物无法调用 step（仅兼容读取）；若你的环境未来改为 dict，可在此扩展。
            continue
        if bool(getattr(obs, "is_dynamic", False)):
            obs.step(dt, w, h)
            if idx in trajs:
                trajs[idx].append((float(obs.x), float(obs.y)))



def draw_env(ax, env: NavigationEnv, title: str, dyn_trajs: dict[int, List[Tuple[float, float]]] | None = None) -> None:
    import matplotlib.pyplot as plt

    w, h = float(EnvConfig.MAP_WIDTH), float(EnvConfig.MAP_HEIGHT)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    # 边界
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], linewidth=1)

    # 目标
    gx, gy = _get_goal_xy(env)
    goal = plt.Circle((gx, gy), radius=float(EnvConfig.GOAL_RADIUS), fill=False, linewidth=2)
    ax.add_patch(goal)

    # 动态障碍物轨迹（红色折线）
    if dyn_trajs:
        for pts in dyn_trajs.values():
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, linewidth=1.2, alpha=0.9, color="red")

    # 障碍物：静态=灰色，动态=红色
    for x, y, r, is_dyn in _iter_obstacles(env):
        if is_dyn:
            circ = plt.Circle((x, y), radius=r, fill=True, alpha=0.35, facecolor="red", edgecolor="red", linewidth=1)
        else:
            circ = plt.Circle((x, y), radius=r, fill=True, alpha=0.55, facecolor="0.7", edgecolor="0.55", linewidth=1)
        ax.add_patch(circ)

    # 动态障碍物运动方向（箭头）
    for idx, obs in enumerate(getattr(env, "obstacles", [])):
        if isinstance(obs, dict) or (not bool(getattr(obs, "is_dynamic", False))):
            continue

        ux, uy = _get_dynamic_heading(obs, None if dyn_trajs is None else dyn_trajs.get(idx))
        if abs(ux) + abs(uy) < 1e-8:
            continue

        arrow_len = max(0.2, float(getattr(obs, "radius", 0.2)) * 2.0)
        ax.arrow(
            float(getattr(obs, "x")),
            float(getattr(obs, "y")),
            ux * arrow_len,
            uy * arrow_len,
            head_width=0.08,
            head_length=0.12,
            fc="darkred",
            ec="darkred",
            linewidth=1.2,
            alpha=0.95,
            length_includes_head=True,
            zorder=5,
        )

    # 机器人
    rx, ry = float(env.robot.x), float(env.robot.y)
    robot = plt.Circle((rx, ry), radius=float(env.robot.radius), fill=True, alpha=0.85, facecolor="#1f77b4")
    ax.add_patch(robot)


def fig_to_rgb(fig) -> np.ndarray:
    """将 matplotlib Figure 转为 RGB uint8(H,W,3)，兼容不同版本 canvas 接口。"""
    fig.canvas.draw()
    canvas = fig.canvas

    # 1) 新版/常见：buffer_rgba -> (H,W,4)
    if hasattr(canvas, "buffer_rgba"):
        buf = np.asarray(canvas.buffer_rgba())
        if buf.ndim == 3 and buf.shape[2] >= 3:
            return np.array(buf[:, :, :3], dtype=np.uint8)

    # 2) print_to_buffer -> (bytes, (w,h)) 但通常是 RGBA
    if hasattr(canvas, "print_to_buffer"):
        raw, (w, h) = canvas.print_to_buffer()
        arr = np.frombuffer(raw, dtype=np.uint8)
        # 经验上是 RGBA
        if arr.size == w * h * 4:
            arr = arr.reshape(h, w, 4)
            return arr[:, :, :3]
        if arr.size == w * h * 3:
            arr = arr.reshape(h, w, 3)
            return arr

    # 3) 老接口：tostring_rgb
    if hasattr(canvas, "tostring_rgb"):
        w, h = canvas.get_width_height()
        arr = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        return arr

    # 4) tostring_argb
    if hasattr(canvas, "tostring_argb"):
        w, h = canvas.get_width_height()
        arr = np.frombuffer(canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        # ARGB -> RGBA
        arr = arr[:, :, [1, 2, 3, 0]]
        return arr[:, :, :3]

    raise RuntimeError("无法从 Matplotlib canvas 提取像素数据（缺少 buffer_rgba/print_to_buffer/tostring_*）。")


def save_gif(frames: List[np.ndarray], out_path: str, fps: int = 15) -> None:
    try:
        import imageio.v2 as imageio  # type: ignore
    except Exception:
        import imageio  # type: ignore
    imageio.mimsave(out_path, frames, fps=int(fps))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)

    # 状态配置（仅用于构造环境，不影响“只展示场景”的核心）
    p.add_argument("--legacy_state", action="store_true")
    p.add_argument("--n_sectors", type=int, default=16, choices=[8, 16])
    p.add_argument("--sector_method", type=str, default="min", choices=["min", "mean"])
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    p.add_argument("--dynamic_speed_min", type=float, default=0.30)
    p.add_argument("--dynamic_speed_max", type=float, default=0.70)
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)

    # PNG
    p.add_argument("--save_png", action="store_true")
    p.add_argument("--out", type=str, default="two_scenarios.png")

    # GIF
    p.add_argument("--save_gif", action="store_true", help="保存 GIF（默认关闭）")
    p.add_argument("--gif_out", type=str, default="two_scenarios.gif")
    p.add_argument("--gif_steps", type=int, default=200, help="GIF 帧数/步数")
    p.add_argument("--gif_fps", type=int, default=15)
    p.add_argument("--gif_dpi", type=int, default=120)

    # 轨迹长度（PNG 时用于先推进若干步，再输出带轨迹的静态图）
    p.add_argument("--traj_steps", type=int, default=200, help="PNG 输出前推进的步数（用于绘制动态障碍物轨迹）")

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    dyn_patterns = tuple([s.strip() for s in str(args.dynamic_patterns).split(",") if s.strip()])

    base_static = int(EnvConfig.NUM_STATIC_OBSTACLES)
    base_dynamic = int(EnvConfig.NUM_DYNAMIC_OBSTACLES)

    env_train = build_env(
        legacy_state=args.legacy_state,
        n_sectors=args.n_sectors,
        sector_method=args.sector_method,
        use_lidar_diff=(not args.disable_lidar_diff),
        use_delta_yaw=(not args.disable_delta_yaw),
        dynamic_speed_min=args.dynamic_speed_min,
        dynamic_speed_max=args.dynamic_speed_max,
        dynamic_patterns=dyn_patterns,
        dynamic_stop_prob=args.dynamic_stop_prob,
    )
    env_train.reset()
    traj_train = init_dynamic_trajs(env_train)

    with TempObstacleCounts(base_static + 2, base_dynamic + 2):
        env_harder = build_env(
            legacy_state=args.legacy_state,
            n_sectors=args.n_sectors,
            sector_method=args.sector_method,
            use_lidar_diff=(not args.disable_lidar_diff),
            use_delta_yaw=(not args.disable_delta_yaw),
            dynamic_speed_min=args.dynamic_speed_min,
            dynamic_speed_max=args.dynamic_speed_max,
            dynamic_patterns=dyn_patterns,
            dynamic_stop_prob=args.dynamic_stop_prob,
        )
        env_harder.reset()
        traj_harder = init_dynamic_trajs(env_harder)

    if args.save_gif:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=int(args.gif_dpi))

        frames: List[np.ndarray] = []

        for t in range(int(args.gif_steps)):
            axes[0].cla()
            axes[1].cla()
            draw_env(axes[0], env_train, f"scene_train  t={t}  (S={base_static}, D={base_dynamic})", dyn_trajs=traj_train)
            draw_env(axes[1], env_harder, f"scene_harder  t={t}  (S={base_static+2}, D={base_dynamic+2})", dyn_trajs=traj_harder)
            plt.tight_layout()

            frames.append(fig_to_rgb(fig))

            # 只推进动态障碍物，让轨迹可视化更稳定（避免 env.step 触发 done/max_steps/reset 等副作用）
            advance_dynamic_obstacles(env_train, traj_train)
            advance_dynamic_obstacles(env_harder, traj_harder)

        save_gif(frames, args.gif_out, fps=int(args.gif_fps))
        plt.close(fig)
        print(f"Saved: {args.gif_out}")

        env_train.close()
        env_harder.close()
        return

    # PNG / show
    import matplotlib.pyplot as plt

    # PNG：先推进若干步以形成轨迹（仅推进动态障碍物）
    for _ in range(max(0, int(args.traj_steps) - 1)):
        advance_dynamic_obstacles(env_train, traj_train)
        advance_dynamic_obstacles(env_harder, traj_harder)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
    draw_env(axes[0], env_train, f"scene_train  t={max(0, int(args.traj_steps)-1)}  (S={base_static}, D={base_dynamic})", dyn_trajs=traj_train)
    draw_env(axes[1], env_harder, f"scene_harder  t={max(0, int(args.traj_steps)-1)}  (S={base_static+2}, D={base_dynamic+2})", dyn_trajs=traj_harder)
    plt.tight_layout()

    if args.save_png:
        plt.savefig(args.out)
        print(f"Saved: {args.out}")
    else:
        plt.show()

    env_train.close()
    env_harder.close()


if __name__ == "__main__":
    main()
