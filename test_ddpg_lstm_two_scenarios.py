#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_ddpg_lstm_two_scenarios.py

评测 DDPG 与 LSTM-DDPG 在两个场景上的效果：

- 场景1：训练场景（EnvConfig 默认静态/动态障碍物数量）
- 场景2：在场景1基础上 +2 静态 +2 动态障碍物

输出（每个场景）：
- 成功率 / 碰撞率 / 超时率
- 平均步数
- 平均奖励（avg_return：每回合累计回报的均值）
- 实时决策时间（ms/step）：mean、p50、p90、p95

可选：--save_gif（默认关闭），保存每个算法每个场景前 N 个 episode 的 GIF。

注意：
- PyTorch 2.6+ 默认 torch.load(weights_only=True) 可能导致旧 checkpoint 无法加载。
  本脚本对“你自己训练产出的 checkpoint（可信）”使用 weights_only=False 加载。
- 本环境目标字段为 goal_x/goal_y（没有 env.goal）。脚本已兼容。
"""

from __future__ import annotations

import argparse
import os
import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from config import EnvConfig
from environment import NavigationEnv


# -----------------------------
# 通用：随机种子
# -----------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 通用：torch.load 兼容
# -----------------------------

def torch_load_compat(path: str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# -----------------------------
# 环境构造：state_cfg
# -----------------------------

@dataclass
class StateCfg:
    legacy_state: bool
    n_sectors: int = 16
    sector_method: str = "min"
    disable_lidar_diff: bool = False
    disable_delta_yaw: bool = False

    def to_env_kwargs(self) -> Dict[str, Any]:
        enhanced_cfg = {
            "n_sectors": int(self.n_sectors),
            "sector_method": str(self.sector_method),
            "use_lidar_diff": (not self.disable_lidar_diff),
            "use_delta_yaw": (not self.disable_delta_yaw),
        }
        return {
            "use_enhanced_state": (not self.legacy_state),
            "enhanced_state_config": enhanced_cfg,
        }


def build_env_with_cfg(
    cfg: StateCfg,
    dynamic_speed_min: float,
    dynamic_speed_max: float,
    dynamic_patterns: Tuple[str, ...],
    dynamic_stop_prob: float,
) -> NavigationEnv:
    kw = cfg.to_env_kwargs()
    env = NavigationEnv(
        use_enhanced_state=kw["use_enhanced_state"],
        enhanced_state_config=kw["enhanced_state_config"],
        dynamic_speed_min=float(dynamic_speed_min),
        dynamic_speed_max=float(dynamic_speed_max),
        dynamic_patterns=tuple(dynamic_patterns),
        dynamic_stop_prob=float(dynamic_stop_prob),
    )
    return env


def infer_state_cfg_by_state_dim(
    target_state_dim: int,
    dynamic_speed_min: float,
    dynamic_speed_max: float,
    dynamic_patterns: Tuple[str, ...],
    dynamic_stop_prob: float,
) -> StateCfg:
    # 1) legacy
    cfg_legacy = StateCfg(legacy_state=True)
    env = build_env_with_cfg(cfg_legacy, dynamic_speed_min, dynamic_speed_max, dynamic_patterns, dynamic_stop_prob)
    if int(env.state_dim) == int(target_state_dim):
        env.close()
        return cfg_legacy
    env.close()

    # 2) enhanced
    for n in (8, 16):
        for disable_diff in (True, False):
            for disable_dyaw in (True, False):
                cfg = StateCfg(
                    legacy_state=False,
                    n_sectors=n,
                    sector_method="min",
                    disable_lidar_diff=disable_diff,
                    disable_delta_yaw=disable_dyaw,
                )
                env = build_env_with_cfg(cfg, dynamic_speed_min, dynamic_speed_max, dynamic_patterns, dynamic_stop_prob)
                if int(env.state_dim) == int(target_state_dim):
                    env.close()
                    return cfg
                env.close()

    raise RuntimeError(
        f"无法根据 target_state_dim={target_state_dim} 自动匹配环境状态配置。"
        "请确认模型训练时使用的是 legacy_state 还是 enhanced_state，以及 n_sectors/diff/delta_yaw 开关。"
    )


# -----------------------------
# 障碍物数量临时修改（场景2）
# -----------------------------

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


# -----------------------------
# GIF 渲染（可选）：静态灰、动态红、动态轨迹红线、方向箭头（按速度方向）
# -----------------------------

def _get_goal_xy(env: NavigationEnv) -> Tuple[float, float]:
    # 本项目环境字段为 goal_x/goal_y
    if hasattr(env, "goal"):
        try:
            gx, gy = env.goal
            return float(gx), float(gy)
        except Exception:
            pass
    return float(getattr(env, "goal_x")), float(getattr(env, "goal_y"))


STATIC_COLOR = "#808080"  # 灰
DYN_COLOR = "#d62728"     # 红
ROBOT_COLOR = "#1f77b4"   # 蓝


def _iter_obstacles_with_vel(env: NavigationEnv):
    """yield: (idx, x, y, r, is_dyn, vx, vy)"""
    for idx, obs in enumerate(getattr(env, "obstacles", [])):
        if isinstance(obs, dict):
            x = float(obs.get("x", 0.0))
            y = float(obs.get("y", 0.0))
            r = float(obs.get("radius", 0.2))
            is_dyn = bool(obs.get("is_dynamic", False))
            vx = float(obs.get("vx", 0.0))
            vy = float(obs.get("vy", 0.0))
        else:
            x = float(getattr(obs, "x"))
            y = float(getattr(obs, "y"))
            r = float(getattr(obs, "radius"))
            is_dyn = bool(getattr(obs, "is_dynamic", False))
            vx = float(getattr(obs, "vx", 0.0))
            vy = float(getattr(obs, "vy", 0.0))
        yield idx, x, y, r, is_dyn, vx, vy


def _draw_arrow(ax, x: float, y: float, vx: float, vy: float, max_speed: float, color: str, alpha: float = 0.9):
    """按速度方向画箭头；速度过小时不画。"""
    import math

    speed = math.hypot(vx, vy)
    if speed < 1e-6:
        return
    ux, uy = vx / speed, vy / speed
    scale = min(1.0, speed / max(1e-6, max_speed))
    length = 0.25 + 0.65 * scale
    ax.arrow(
        x, y,
        ux * length, uy * length,
        head_width=0.18, head_length=0.22,
        fc=color, ec=color,
        linewidth=0.0,
        alpha=alpha,
        length_includes_head=True,
        zorder=6,
    )


def fig_to_rgb(fig) -> np.ndarray:
    """兼容不同 matplotlib 版本：返回 HxWx3 uint8"""
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba())
        return rgba[..., :3].copy()
    if hasattr(fig.canvas, "print_to_buffer"):
        buf, (w, h) = fig.canvas.print_to_buffer()
        rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        return rgba[..., :3].copy()
    if hasattr(fig.canvas, "tostring_argb"):
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8).reshape(h, w, 4)
        rgba = argb[..., [1, 2, 3, 0]]
        return rgba[..., :3].copy()
    raise RuntimeError("当前 matplotlib 后端不支持导出像素缓冲区")


def render_frame_rgb(
    env: NavigationEnv,
    dyn_trajs: Optional[Dict[int, List[Tuple[float, float]]]] = None,
    dpi: int = 100,
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    w, h = float(EnvConfig.MAP_WIDTH), float(EnvConfig.MAP_HEIGHT)

    fig = plt.figure(figsize=(4.5, 4.5), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(False)

    # 边界
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], linewidth=1)

    # 目标
    gx, gy = _get_goal_xy(env)
    goal = plt.Circle((gx, gy), radius=float(EnvConfig.GOAL_RADIUS), fill=False, linewidth=2)
    ax.add_patch(goal)

    # 动态轨迹（底层）
    if dyn_trajs:
        for _, pts in dyn_trajs.items():
            if len(pts) >= 2:
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, linewidth=1.2, alpha=0.85, color=DYN_COLOR, zorder=1)

    # 障碍物 + 动态箭头
    dyn_max = float(getattr(env, "dynamic_speed_max", 1.0))
    for idx, x, y, r, is_dyn, vx, vy in _iter_obstacles_with_vel(env):
        if is_dyn:
            circ = plt.Circle((x, y), radius=r, fill=True, alpha=0.35, facecolor=DYN_COLOR, edgecolor=DYN_COLOR, linewidth=0.5)
            ax.add_patch(circ)
            _draw_arrow(ax, x, y, vx, vy, max_speed=max(1e-6, dyn_max), color=DYN_COLOR, alpha=0.95)
        else:
            circ = plt.Circle((x, y), radius=r, fill=True, alpha=0.35, facecolor=STATIC_COLOR, edgecolor=STATIC_COLOR, linewidth=0.5)
            ax.add_patch(circ)

    # 机器人
    rx, ry = float(env.robot.x), float(env.robot.y)
    robot = plt.Circle((rx, ry), radius=float(env.robot.radius), fill=True, alpha=0.90, facecolor=ROBOT_COLOR, edgecolor=ROBOT_COLOR)
    ax.add_patch(robot)

    # robot 方向箭头（按速度方向；速度过小则不画）
    rvx, rvy = float(getattr(env.robot, "vx", 0.0)), float(getattr(env.robot, "vy", 0.0))
    _draw_arrow(ax, rx, ry, rvx, rvy, max_speed=float(getattr(EnvConfig, "MAX_LINEAR_VEL", 1.0)), color="black", alpha=0.9)

    img = fig_to_rgb(fig)
    plt.close(fig)
    return img


def save_gif(frames: List[np.ndarray], path: str, fps: int = 15) -> None:
    try:
        import imageio.v2 as imageio
    except Exception:
        import imageio  # type: ignore
    imageio.mimsave(path, frames, fps=int(fps))


# -----------------------------
# 评测主体
# -----------------------------

@dataclass
class EvalResult:
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    avg_steps: float
    avg_return: float
    decision_ms_mean: float
    decision_ms_p50: float
    decision_ms_p90: float
    decision_ms_p95: float


def classify_done(info: Dict[str, Any]) -> str:
    reason = str(info.get("reason", ""))
    if reason == "goal_reached":
        return "success"
    if reason in ("collision_obstacle", "collision_wall"):
        return "collision"
    return "timeout"


def percentile(a: List[float], q: float) -> float:
    if not a:
        return 0.0
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def evaluate_ddpg(
    model_path: str,
    env: NavigationEnv,
    episodes: int,
    seed: int,
    save_gif_flag: bool,
    gif_dir: str,
    gif_fps: int,
    gif_episodes: int,
    gif_max_steps: int,
) -> EvalResult:
    from ddpg import DDPGAgent

    ckpt = torch_load_compat(model_path, map_location="cpu")
    actor_sd = ckpt["actor_local"] if isinstance(ckpt, dict) and "actor_local" in ckpt else ckpt

    w = actor_sd.get("linear1.weight", None)
    if w is None:
        raise RuntimeError("无法从 DDPG checkpoint 中找到 linear1.weight，确认是否为 ddpg.py 保存的模型？")
    state_dim = int(w.shape[1])
    hidden_dim = int(w.shape[0])

    agent = DDPGAgent(state_dim=state_dim, action_dim=env.action_dim, hidden_dim=hidden_dim)
    agent.actor_local.load_state_dict(actor_sd, strict=True)
    agent.actor_local.eval()

    n_success = n_collision = n_timeout = 0
    steps_list: List[int] = []
    returns_list: List[float] = []
    decision_times_ms: List[float] = []

    os.makedirs(gif_dir, exist_ok=True)
    record_eps = set(range(min(int(gif_episodes), int(episodes)))) if save_gif_flag else set()

    for ep in range(int(episodes)):
        seed_everything(int(seed) + ep)
        s = env.reset()

        ep_return = 0.0

        frames: List[np.ndarray] = []
        dyn_trajs: Optional[Dict[int, List[Tuple[float, float]]]] = None
        if ep in record_eps:
            dyn_trajs = {}
            for idx, x, y, r0, is_dyn, vx, vy in _iter_obstacles_with_vel(env):
                if is_dyn:
                    dyn_trajs[idx] = [(x, y)]
            frames.append(render_frame_rgb(env, dyn_trajs=dyn_trajs))

        done_type = "timeout"
        max_steps = int(min(EnvConfig.MAX_STEPS, gif_max_steps if save_gif_flag else EnvConfig.MAX_STEPS))
        for t in range(max_steps):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            a = agent.act(s, step=0, add_noise=False).reshape(-1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decision_times_ms.append((time.perf_counter() - t0) * 1000.0)

            s2, r, done, info = env.step(a)

            ep_return += float(r)

            if ep in record_eps:
                if dyn_trajs is not None:
                    for idx, x, y, r0, is_dyn, vx, vy in _iter_obstacles_with_vel(env):
                        if is_dyn and idx in dyn_trajs:
                            dyn_trajs[idx].append((x, y))
                frames.append(render_frame_rgb(env, dyn_trajs=dyn_trajs))

            s = s2
            if done:
                done_type = classify_done(info)
                break

        steps_list.append(t + 1)
        returns_list.append(ep_return)
        if done_type == "success":
            n_success += 1
        elif done_type == "collision":
            n_collision += 1
        else:
            n_timeout += 1

        if ep in record_eps and frames:
            out_gif = os.path.join(gif_dir, f"ddpg_ep{ep+1:03d}.gif")
            save_gif(frames, out_gif, fps=gif_fps)

    return EvalResult(
        episodes=int(episodes),
        success_rate=n_success / episodes,
        collision_rate=n_collision / episodes,
        timeout_rate=n_timeout / episodes,
        avg_steps=float(np.mean(steps_list)) if steps_list else 0.0,
        avg_return=float(np.mean(returns_list)) if returns_list else 0.0,
        decision_ms_mean=float(np.mean(decision_times_ms)) if decision_times_ms else 0.0,
        decision_ms_p50=percentile(decision_times_ms, 50),
        decision_ms_p90=percentile(decision_times_ms, 90),
        decision_ms_p95=percentile(decision_times_ms, 95),
    )


def evaluate_lstm_ddpg(
    model_path: str,
    env: NavigationEnv,
    episodes: int,
    seed: int,
    save_gif_flag: bool,
    gif_dir: str,
    gif_fps: int,
    gif_episodes: int,
    gif_max_steps: int,
) -> EvalResult:
    from lstm_ddpg import LSTMDdpgAgent

    ckpt = torch_load_compat(model_path, map_location="cpu")

    if not isinstance(ckpt, dict) or "actor_local" not in ckpt:
        raise RuntimeError("LSTM-DDPG checkpoint 格式不符合 lstm_ddpg.py 的 save() 输出（缺少 actor_local）。")

    actor_sd = ckpt["actor_local"]
    net_cfg = ckpt.get("net_cfg", {}) or {}
    history_len = int(ckpt.get("history_len", net_cfg.get("history_len", 5)))

    embed_dim = int(net_cfg.get("embed_dim", 64))
    lstm_hidden_dim = int(net_cfg.get("lstm_hidden_dim", 64))
    mlp_hidden_dim = int(net_cfg.get("mlp_hidden_dim", 256))

    w = actor_sd.get("fc_in.weight", None)
    state_dim = int(w.shape[1]) if w is not None else int(net_cfg.get("state_dim", env.state_dim))

    agent = LSTMDdpgAgent(
        state_dim=state_dim,
        action_dim=env.action_dim,
        history_len=history_len,
        embed_dim=embed_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        hidden_dim=mlp_hidden_dim,
    )
    agent.actor_local.load_state_dict(actor_sd, strict=True)
    agent.actor_local.eval()

    n_success = n_collision = n_timeout = 0
    steps_list: List[int] = []
    returns_list: List[float] = []
    decision_times_ms: List[float] = []

    os.makedirs(gif_dir, exist_ok=True)
    record_eps = set(range(min(int(gif_episodes), int(episodes)))) if save_gif_flag else set()

    for ep in range(int(episodes)):
        seed_everything(int(seed) + ep)
        s = env.reset()

        q = deque([s.copy() for _ in range(history_len)], maxlen=history_len)

        ep_return = 0.0

        frames: List[np.ndarray] = []
        dyn_trajs: Optional[Dict[int, List[Tuple[float, float]]]] = None
        if ep in record_eps:
            dyn_trajs = {}
            for idx, x, y, r0, is_dyn, vx, vy in _iter_obstacles_with_vel(env):
                if is_dyn:
                    dyn_trajs[idx] = [(x, y)]
            frames.append(render_frame_rgb(env, dyn_trajs=dyn_trajs))

        done_type = "timeout"
        max_steps = int(min(EnvConfig.MAX_STEPS, gif_max_steps if save_gif_flag else EnvConfig.MAX_STEPS))
        for t in range(max_steps):
            state_seq = np.asarray(q, dtype=np.float32)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            a = agent.act(state_seq, step=0, add_noise=False).reshape(-1)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            decision_times_ms.append((time.perf_counter() - t0) * 1000.0)

            s2, r, done, info = env.step(a)

            ep_return += float(r)

            if ep in record_eps:
                if dyn_trajs is not None:
                    for idx, x, y, r0, is_dyn, vx, vy in _iter_obstacles_with_vel(env):
                        if is_dyn and idx in dyn_trajs:
                            dyn_trajs[idx].append((x, y))
                frames.append(render_frame_rgb(env, dyn_trajs=dyn_trajs))

            q.append(s2.copy())
            s = s2
            if done:
                done_type = classify_done(info)
                break

        steps_list.append(t + 1)
        returns_list.append(ep_return)
        if done_type == "success":
            n_success += 1
        elif done_type == "collision":
            n_collision += 1
        else:
            n_timeout += 1

        if ep in record_eps and frames:
            out_gif = os.path.join(gif_dir, f"lstm_ddpg_ep{ep+1:03d}.gif")
            save_gif(frames, out_gif, fps=gif_fps)

    return EvalResult(
        episodes=int(episodes),
        success_rate=n_success / episodes,
        collision_rate=n_collision / episodes,
        timeout_rate=n_timeout / episodes,
        avg_steps=float(np.mean(steps_list)) if steps_list else 0.0,
        avg_return=float(np.mean(returns_list)) if returns_list else 0.0,
        decision_ms_mean=float(np.mean(decision_times_ms)) if decision_times_ms else 0.0,
        decision_ms_p50=percentile(decision_times_ms, 50),
        decision_ms_p90=percentile(decision_times_ms, 90),
        decision_ms_p95=percentile(decision_times_ms, 95),
    )


def print_result(title: str, r: EvalResult) -> None:
    print(f"\n[{title}]")
    print(f"  episodes        : {r.episodes}")
    print(f"  success_rate    : {r.success_rate*100:.2f}%")
    print(f"  collision_rate  : {r.collision_rate*100:.2f}%")
    print(f"  timeout_rate    : {r.timeout_rate*100:.2f}%")
    print(f"  avg_steps       : {r.avg_steps:.2f}")
    print(f"  avg_return      : {r.avg_return:.2f}")
    print(
        f"  decision_time(ms): mean={r.decision_ms_mean:.3f}, p50={r.decision_ms_p50:.3f}, "
        f"p90={r.decision_ms_p90:.3f}, p95={r.decision_ms_p95:.3f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ddpg_model", type=str, default="", help="DDPG checkpoint 路径（可为空）")
    p.add_argument("--lstm_model", type=str, default="", help="LSTM-DDPG checkpoint 路径（可为空）")

    p.add_argument("--episodes", type=int, default=200, help="每个场景的测试回合数")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--dynamic_speed_min", type=float, default=0.50)
    p.add_argument("--dynamic_speed_max", type=float, default=0.70)
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)

    # GIF
    p.add_argument("--save_gif", action="store_true", help="保存 GIF（默认关闭）")
    p.add_argument("--gif_dir", type=str, default="gifs")
    p.add_argument("--gif_fps", type=int, default=15)
    p.add_argument("--gif_episodes", type=int, default=0, help="每个场景记录前 N 个 episode")
    p.add_argument("--gif_max_steps", type=int, default=500)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.ddpg_model and not args.lstm_model:
        raise SystemExit("请至少提供 --ddpg_model 或 --lstm_model 其中一个。")

    dyn_patterns = tuple([s.strip() for s in str(args.dynamic_patterns).split(",") if s.strip()])

    base_static = int(EnvConfig.NUM_STATIC_OBSTACLES)
    base_dynamic = int(EnvConfig.NUM_DYNAMIC_OBSTACLES)

    def run_for_algo(algo_name: str, model_path: str, evaluator_fn):
        if not os.path.exists(model_path):
            print(f"[{algo_name}] 模型不存在：{model_path}")
            return

        ckpt = torch_load_compat(model_path, map_location="cpu")

        # 推断 state_dim（不同算法层名不同）
        state_dim: Optional[int] = None
        if algo_name.lower() == "ddpg":
            sd = ckpt["actor_local"] if isinstance(ckpt, dict) and "actor_local" in ckpt else ckpt
            w = sd.get("linear1.weight", None)
            if w is not None:
                state_dim = int(w.shape[1])
        else:
            if isinstance(ckpt, dict):
                sd = ckpt.get("actor_local", {})
                w = sd.get("fc_in.weight", None)
                if w is not None:
                    state_dim = int(w.shape[1])
                else:
                    state_dim = int((ckpt.get("net_cfg", {}) or {}).get("state_dim", 0)) or None

        if state_dim is None:
            raise RuntimeError(f"[{algo_name}] 无法从 checkpoint 推断 state_dim。")

        st_cfg = infer_state_cfg_by_state_dim(
            target_state_dim=state_dim,
            dynamic_speed_min=args.dynamic_speed_min,
            dynamic_speed_max=args.dynamic_speed_max,
            dynamic_patterns=dyn_patterns,
            dynamic_stop_prob=args.dynamic_stop_prob,
        )

        print("\n" + "=" * 70)
        print(
            f"[{algo_name}] Using StateCfg: legacy={st_cfg.legacy_state}, n_sectors={st_cfg.n_sectors}, "
            f"disable_lidar_diff={st_cfg.disable_lidar_diff}, disable_delta_yaw={st_cfg.disable_delta_yaw}"
        )
        print(
            f"[{algo_name}] Dynamic: {args.dynamic_speed_min:.2f}~{args.dynamic_speed_max:.2f}, "
            f"patterns={dyn_patterns}, stop_prob={args.dynamic_stop_prob:.2f}"
        )
        print("=" * 70)

        # 场景1
        env1 = build_env_with_cfg(st_cfg, args.dynamic_speed_min, args.dynamic_speed_max, dyn_patterns, args.dynamic_stop_prob)
        res1 = evaluator_fn(
            model_path=model_path,
            env=env1,
            episodes=args.episodes,
            seed=args.seed,
            save_gif_flag=args.save_gif,
            gif_dir=os.path.join(args.gif_dir, algo_name, "scene_train"),
            gif_fps=args.gif_fps,
            gif_episodes=args.gif_episodes,
            gif_max_steps=args.gif_max_steps,
        )
        env1.close()

        # 场景2：+2S+2D
        with TempObstacleCounts(base_static + 2, base_dynamic + 2):
            env2 = build_env_with_cfg(st_cfg, args.dynamic_speed_min, args.dynamic_speed_max, dyn_patterns, args.dynamic_stop_prob)
            res2 = evaluator_fn(
                model_path=model_path,
                env=env2,
                episodes=args.episodes,
                seed=args.seed,
                save_gif_flag=args.save_gif,
                gif_dir=os.path.join(args.gif_dir, algo_name, "scene_harder"),
                gif_fps=args.gif_fps,
                gif_episodes=args.gif_episodes,
                gif_max_steps=args.gif_max_steps,
            )
            env2.close()

        print_result(f"{algo_name} - scene_train", res1)
        print_result(f"{algo_name} - scene_harder(+2S+2D)", res2)

    if args.ddpg_model:
        run_for_algo("DDPG", args.ddpg_model, evaluate_ddpg)
    if args.lstm_model:
        run_for_algo("LSTM_DDPG", args.lstm_model, evaluate_lstm_ddpg)


if __name__ == "__main__":
    main()
