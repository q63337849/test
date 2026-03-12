#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在“自定义测试场景”上进行 DDPG vs LSTM-DDPG-Att 航迹规划对比。"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import EnvConfig
from environment import NavigationEnv, Obstacle


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def safe_torch_load(path: str, map_location: str = "cpu") -> Any:
    import torch
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def resolve_model_path(path: str, model_dir: str = "") -> str:
    if os.path.exists(path):
        return path
    if model_dir:
        alt = os.path.join(model_dir, os.path.basename(path))
        if os.path.exists(alt):
            return alt
    return path


def infer_ddpg_state_dim_from_ckpt(ckpt: Any) -> int:
    sd = ckpt.get("actor_local", ckpt) if isinstance(ckpt, dict) else None
    if not isinstance(sd, dict):
        raise RuntimeError("ddpg checkpoint 格式不支持")

    for k in ("fc1.weight", "linear1.weight", "actor.fc1.weight", "actor.linear1.weight", "mlp.0.weight", "net.0.weight"):
        w = sd.get(k)
        if getattr(w, "ndim", None) == 2:
            return int(w.shape[1])

    dims = [int(v.shape[1]) for v in sd.values() if getattr(v, "ndim", None) == 2]
    if dims:
        return min(dims)
    raise RuntimeError("无法从 DDPG checkpoint 推断 state_dim")


def infer_state_cfg_from_state_dim(state_dim: int) -> Dict[str, Any]:
    legacy_dim = EnvConfig.LIDAR_RAYS + 2 + 2 + 1 + 2
    if state_dim == legacy_dim:
        return dict(use_enhanced_state=False)

    def enhanced_dim(n: int, dis_diff: bool, dis_dyaw: bool) -> int:
        b = n + 3 + 2 + 2 + 1 + 1
        if not dis_diff:
            b += n
        if not dis_dyaw:
            b += 2
        return b

    cands = []
    for n in (8, 16):
        for dis_diff in (False, True):
            for dis_dyaw in (False, True):
                if enhanced_dim(n, dis_diff, dis_dyaw) == state_dim:
                    cands.append((n, dis_diff, dis_dyaw))
    if not cands:
        raise RuntimeError(f"无法从 state_dim={state_dim} 推断环境配置")

    cands.sort(key=lambda x: (1 if x[0] == 16 else 0, 1 if not x[1] else 0, 1 if not x[2] else 0), reverse=True)
    n, dis_diff, dis_dyaw = cands[0]
    return dict(
        use_enhanced_state=True,
        enhanced_state_config={
            "n_sectors": n,
            "sector_method": "min",
            "use_lidar_diff": (not dis_diff),
            "use_delta_yaw": (not dis_dyaw),
        },
    )


class DDPGPolicy:
    history_len = 1

    def __init__(self, model_path: str, state_dim: int):
        from ddpg import DDPGAgent
        kwargs = dict(state_dim=state_dim, action_dim=2, random_seed=0, batch_size=64, buffer_size=1000)
        sig = inspect.signature(DDPGAgent.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        self.agent = DDPGAgent(**kwargs)
        self.agent.load(model_path)

    def act(self, s: np.ndarray) -> np.ndarray:
        a = np.asarray(self.agent.act(s, step=0, add_noise=False), dtype=np.float32).reshape(-1)
        a[0] = np.clip(a[0], 0.0, EnvConfig.MAX_LINEAR_VEL)
        a[1] = np.clip(a[1], -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL)
        return a


class AttPolicy:
    def __init__(self, model_path: str, ckpt: dict, state_dim: int):
        from lstm_ddpg_att import LSTMDdpgAgent
        self.history_len = int(ckpt.get("history_len", 4))
        net_cfg = ckpt.get("net_cfg", {}) if isinstance(ckpt, dict) else {}
        kwargs = dict(
            state_dim=state_dim, action_dim=2, history_len=self.history_len,
            batch_size=64, buffer_size=1000,
            embed_dim=int(net_cfg.get("embed_dim", 64)),
            lstm_hidden_dim=int(net_cfg.get("lstm_hidden_dim", 64)),
            use_spatial_att=bool(net_cfg.get("use_spatial_att", True)),
            use_temporal_att=bool(net_cfg.get("use_temporal_att", True)),
            sector_model_dim=int(net_cfg.get("sector_model_dim", 64)),
            temporal_att_dim=int(net_cfg.get("temporal_att_dim", 64)),
            spatial_att_heads=int(net_cfg.get("spatial_att_heads", 2)),
            temporal_att_heads=int(net_cfg.get("temporal_att_heads", 2)),
        )
        sig = inspect.signature(LSTMDdpgAgent.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        self.agent = LSTMDdpgAgent(**kwargs)
        self.agent.load(model_path)

    def act(self, seq: np.ndarray) -> np.ndarray:
        a = np.asarray(self.agent.act(seq, step=0, add_noise=False), dtype=np.float32).reshape(-1)
        a[0] = np.clip(a[0], 0.0, EnvConfig.MAX_LINEAR_VEL)
        a[1] = np.clip(a[1], -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL)
        return a


def build_custom_scene(env: NavigationEnv, args: argparse.Namespace) -> Tuple[Tuple[float, float], List[Tuple[float, float, float]], List[int], List[float]]:
    """按示意图构造固定测试环境（左上起点、右中终点，静/动态障碍混合且半径不一）。"""
    env.step_count = 0
    env.episode_success = False
    env.episode_failure = False

    # 起点（蓝点）&终点（黑色空心圈）
    start_x, start_y, start_theta = 1.4, 8.4, -0.2
    env.robot.reset(start_x, start_y, start_theta)
    env.goal_x, env.goal_y = 8.8, 6.0

    # 参考图片布局，障碍物数量/大小/速度可调
    static_template = [
        (2.1, 7.2, 0.22),
        (3.6, 6.6, 0.16),
        (4.8, 2.8, 0.28),
        (6.9, 7.9, 0.25),
    ]
    dynamic_template = [
        (1.8, 1.6, 0.22, 0.55),
        (4.9, 4.2, 0.25, 0.60),
        (6.6, 5.4, 0.23, 0.58),
        (7.7, 7.3, 0.20, 0.62),
        (7.4, 4.0, 0.18, 0.52),
    ]

    static_n = max(0, min(int(args.num_static), len(static_template)))
    dynamic_n = max(0, min(int(args.num_dynamic), len(dynamic_template)))

    obstacles: List[Obstacle] = []
    for x, y, r in static_template[:static_n]:
        rr = float(np.clip(r * args.radius_scale, args.radius_min, args.radius_max))
        obstacles.append(Obstacle(x=x, y=y, radius=rr, is_dynamic=False))

    dyn_indices: List[int] = []
    dyn_radii: List[float] = []
    for x, y, r, base_speed in dynamic_template[:dynamic_n]:
        rr = float(np.clip(r * args.radius_scale, args.radius_min, args.radius_max))
        speed = float(np.clip(base_speed * args.speed_scale, args.dynamic_speed_min, args.dynamic_speed_max))
        ang = float(args.dynamic_heading_deg) * math.pi / 180.0
        vx, vy = speed * math.cos(ang), speed * math.sin(ang)
        obs = Obstacle(
            x=x, y=y, radius=rr, is_dynamic=True,
            vx=vx, vy=vy, pattern=args.dynamic_pattern,
            speed_min=args.dynamic_speed_min, speed_max=args.dynamic_speed_max,
            stop_prob=args.dynamic_stop_prob,
        )
        dyn_indices.append(len(obstacles))
        dyn_radii.append(rr)
        obstacles.append(obs)

    env.obstacles = obstacles
    env.prev_action = np.zeros(2, dtype=np.float32)
    if env._enhanced_state is not None:
        env._enhanced_state.reset()
    env.previous_distance = env._get_distance_to_goal()
    env.previous_heading = env._get_heading_to_goal()

    static_for_plot = [(o.x, o.y, o.radius) for o in obstacles if not o.is_dynamic]
    return (start_x, start_y), static_for_plot, dyn_indices, dyn_radii


def rollout(policy: Any, env_cfg: Dict[str, Any], args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    set_seed(seed)
    env = NavigationEnv(
        **env_cfg,
        dynamic_speed_min=args.dynamic_speed_min,
        dynamic_speed_max=args.dynamic_speed_max,
        dynamic_patterns=(args.dynamic_pattern,),
        dynamic_stop_prob=args.dynamic_stop_prob,
    )
    env.reset()

    start_xy, static_obs, dyn_indices, dyn_radii = build_custom_scene(env, args)

    state = env._get_state()
    q = None
    if getattr(policy, "history_len", 1) > 1:
        q = deque([state.copy() for _ in range(policy.history_len)], maxlen=policy.history_len)

    robot_traj = [(float(env.robot.x), float(env.robot.y))]
    dyn_trajs = [[(float(env.obstacles[idx].x), float(env.obstacles[idx].y))] for idx in dyn_indices]

    done = False
    info = {"reason": None}
    ret = 0.0
    steps = 0
    while (not done) and steps < int(args.max_steps):
        s_in = np.stack(list(q), axis=0) if q is not None else state
        a = policy.act(s_in)
        state, r, done, info = env.step(a)
        ret += float(r)
        steps += 1
        if q is not None:
            q.append(state.copy())
        robot_traj.append((float(env.robot.x), float(env.robot.y)))
        for k, idx in enumerate(dyn_indices):
            o = env.obstacles[idx]
            dyn_trajs[k].append((float(o.x), float(o.y)))

    out = dict(
        start=start_xy,
        goal=(float(env.goal_x), float(env.goal_y)),
        static_obs=static_obs,
        dyn_trajs=dyn_trajs,
        dyn_radii=dyn_radii,
        robot_traj=robot_traj,
        reason=str(info.get("reason", "max_steps")),
        steps=steps,
        ret=ret,
    )
    env.close()
    return out


def plot_compare(ddpg: Dict[str, Any], att: Dict[str, Any], out_path: str, seed: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(7.0, 6.4), dpi=140)
    w, h = float(EnvConfig.MAP_WIDTH), float(EnvConfig.MAP_HEIGHT)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color="black", linewidth=1.0)

    # 蓝色起点
    sx, sy = ddpg["start"]
    ax.scatter([sx], [sy], s=32, c="#1f77b4", marker="o", zorder=8)

    # 黑色空心终点
    gx, gy = ddpg["goal"]
    ax.scatter([gx], [gy], s=190, facecolors='none', edgecolors='black', linewidths=1.8, zorder=8)

    # 灰色静态障碍物（真实半径）
    for x, y, r in ddpg["static_obs"]:
        ax.add_patch(plt.Circle((x, y), radius=r, facecolor="0.7", edgecolor="0.55", alpha=0.75, linewidth=1.0, zorder=4))

    # 红色动态障碍物 + 轨迹 + 方向
    for traj, r in zip(ddpg["dyn_trajs"], ddpg["dyn_radii"]):
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        if len(traj) >= 2:
            ax.plot(xs, ys, color="#ff6b6b", linewidth=1.2, alpha=0.9, zorder=5)
            dx, dy = xs[-1] - xs[-2], ys[-1] - ys[-2]
            n = float(np.hypot(dx, dy))
            if n > 1e-8:
                ux, uy = dx / n, dy / n
                ax.arrow(xs[-1], ys[-1], ux * 0.24, uy * 0.24, head_width=0.06, head_length=0.08,
                         fc="#d62728", ec="#d62728", linewidth=1.0, alpha=0.95, length_includes_head=True, zorder=6)
        ax.add_patch(plt.Circle((xs[-1], ys[-1]), radius=r, facecolor="#d62728", edgecolor="#d62728", alpha=0.35, linewidth=1.0, zorder=6))

    # 航迹对比
    def draw_robot(traj, color, style, label):
        if len(traj) < 2:
            return
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color=color, linestyle=style, linewidth=2.0, label=label, zorder=7)

    draw_robot(ddpg["robot_traj"], "#1f77b4", "-", f"DDPG (steps={ddpg['steps']}, {ddpg['reason']})")
    draw_robot(att["robot_traj"], "#ff7f0e", "--", f"LSTM-DDPG-Att (steps={att['steps']}, {att['reason']})")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title(f"Custom Test Scene Trajectory Planning Compare (seed={seed})")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ddpg_model", type=str, required=True)
    p.add_argument("--att_model", type=str, required=True)
    p.add_argument("--model_dir", type=str, default="")
    p.add_argument("--out", type=str, default="ddpg_vs_att_custom_test_scene.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=int(getattr(EnvConfig, "MAX_STEPS", 500)))

    # 测试场景配置（数量、大小、速度可设置）
    p.add_argument("--num_static", type=int, default=4)
    p.add_argument("--num_dynamic", type=int, default=5)
    p.add_argument("--radius_min", type=float, default=float(EnvConfig.OBSTACLE_RADIUS_MIN))
    p.add_argument("--radius_max", type=float, default=float(EnvConfig.OBSTACLE_RADIUS_MAX))
    p.add_argument("--radius_scale", type=float, default=1.0)
    p.add_argument("--dynamic_speed_min", type=float, default=float(EnvConfig.DYNAMIC_OBS_VEL_MIN))
    p.add_argument("--dynamic_speed_max", type=float, default=float(EnvConfig.DYNAMIC_OBS_VEL_MAX))
    p.add_argument("--speed_scale", type=float, default=1.0)
    p.add_argument("--dynamic_pattern", type=str, default="bounce", choices=["bounce", "random_walk", "stop_and_go"])
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)
    p.add_argument("--dynamic_heading_deg", type=float, default=-35.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ddpg_path = resolve_model_path(args.ddpg_model, args.model_dir)
    att_path = resolve_model_path(args.att_model, args.model_dir)

    ddpg_ckpt = safe_torch_load(ddpg_path, map_location="cpu")
    att_ckpt = safe_torch_load(att_path, map_location="cpu")

    if not isinstance(att_ckpt, dict) or ("actor_local" not in att_ckpt):
        raise RuntimeError("att_model 不是预期 checkpoint(dict)，缺少 actor_local")

    state_dim = infer_ddpg_state_dim_from_ckpt(ddpg_ckpt)
    env_cfg = infer_state_cfg_from_state_dim(state_dim)

    ddpg_policy = DDPGPolicy(ddpg_path, state_dim)
    att_policy = AttPolicy(att_path, att_ckpt, state_dim)

    ddpg_roll = rollout(ddpg_policy, env_cfg, args, seed=args.seed)
    att_roll = rollout(att_policy, env_cfg, args, seed=args.seed)

    plot_compare(ddpg_roll, att_roll, args.out, seed=args.seed)
    print(f"Saved: {args.out}")
    print(f"DDPG: steps={ddpg_roll['steps']}, reason={ddpg_roll['reason']}, return={ddpg_roll['ret']:.2f}")
    print(f"ATT : steps={att_roll['steps']}, reason={att_roll['reason']}, return={att_roll['ret']:.2f}")


if __name__ == "__main__":
    main()
