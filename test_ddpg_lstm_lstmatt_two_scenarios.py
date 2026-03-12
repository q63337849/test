
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_ddpg_lstm_lstmatt_two_scenarios.py

同一脚本同时评测：DDPG / LSTM_DDPG / LSTM_DDPG_ATT
在两个场景：
  1) scene_train: 训练场景（默认障碍数量取 EnvConfig.NUM_STATIC_OBSTACLES / NUM_DYNAMIC_OBSTACLES）
  2) scene_harder: 在训练场景基础上 +2 静态 +2 动态

指标：
  - success_rate / collision_rate / timeout_rate
  - avg_steps / avg_reward
  - decision_time(ms): mean / p50 / p90 / p95

可选：为每个算法、每个场景保存 GIF（含：
  静态障碍灰色、动态障碍红色、动态障碍轨迹、无人机/动态障碍速度方向箭头）。

用法示例：
  python test_ddpg_lstm_lstmatt_two_scenarios.py \
    --ddpg_model models/ddpg_best.pth \
    --lstm_model models/lstm_ddpg_best.pth \
    --att_model  models/lstm_ddpg_att_best.pth \
    --episodes 200 --seed 0 \
    --dynamic_speed_min 0.3 --dynamic_speed_max 0.7 \
    --dynamic_patterns bounce,random_walk --dynamic_stop_prob 0.05 \
    --save_gif --gif_episodes 1 --gif_fps 15

说明：
  - DDPG checkpoint 通常不带 state_meta，本脚本会从网络第一层输入维度自动推断 state 配置。
  - LSTM_DDPG / LSTM_DDPG_ATT checkpoint 若带 state_meta，则直接使用 state_meta。
  - PyTorch >=2.6 的 torch.load 默认 weights_only=True，本脚本会显式使用 weights_only=False（仅用于你自己训练得到的 checkpoint）。
"""

from __future__ import annotations

import argparse
import os
import time
import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# 这几个模块均来自你的工程目录
from config import EnvConfig
from environment import NavigationEnv


# -----------------------------
# Utils
# -----------------------------

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def percentile(x: List[float], p: float) -> float:
    if len(x) == 0:
        return float("nan")
    arr = np.asarray(x, dtype=np.float64)
    return float(np.percentile(arr, p))


def safe_torch_load(path: str, map_location: str = "cpu") -> Any:
    """兼容 PyTorch 2.6+ weights_only 默认值变更。"""
    import torch
    try:
        # PyTorch 2.6+ 支持 weights_only 参数
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # 老版本 torch.load 没有 weights_only
        return torch.load(path, map_location=map_location)


def ensure_dir(p: str) -> None:
    if p and (not os.path.exists(p)):
        os.makedirs(p, exist_ok=True)


def parse_patterns(s: str) -> Tuple[str, ...]:
    items = [x.strip() for x in str(s).split(",") if x.strip()]
    return tuple(items)


def action_clip(a: np.ndarray) -> np.ndarray:
    a0 = float(np.clip(a[0], 0.0, EnvConfig.MAX_LINEAR_VEL))
    a1 = float(np.clip(a[1], -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL))
    return np.array([a0, a1], dtype=np.float32)


# -----------------------------
# State cfg inference
# -----------------------------

@dataclass
class StateCfg:
    legacy_state: bool
    n_sectors: int = 16
    sector_method: str = "min"  # min/mean
    disable_lidar_diff: bool = False
    disable_delta_yaw: bool = False

    def to_env_kwargs(self) -> Dict[str, Any]:
        if self.legacy_state:
            return dict(use_enhanced_state=False)
        enhanced_cfg = {
            "n_sectors": int(self.n_sectors),
            "sector_method": str(self.sector_method),
            "use_lidar_diff": (not self.disable_lidar_diff),
            "use_delta_yaw": (not self.disable_delta_yaw),
        }
        return dict(use_enhanced_state=True, enhanced_state_config=enhanced_cfg)


def enhanced_state_dim(n_sectors: int, disable_lidar_diff: bool, disable_delta_yaw: bool) -> int:
    # EnhancedSim2RealStateV2:
    #  sectors(n) + lidar_diff(n if enabled) + goal_rel(3) + vel_xy(2) + goal_dir(2)
    #  + min_lidar(1) + yaw(1) + delta_yaw(2 if enabled)
    base = n_sectors + 3 + 2 + 2 + 1 + 1
    if not disable_lidar_diff:
        base += n_sectors
    if not disable_delta_yaw:
        base += 2
    return int(base)


def infer_state_cfg_from_state_dim(state_dim: int) -> StateCfg:
    # legacy
    if state_dim == EnvConfig.LIDAR_RAYS + 2 + 2 + 1 + 2:
        return StateCfg(legacy_state=True)

    # enhanced candidates
    candidates: List[StateCfg] = []
    for n in (8, 16):
        for dis_diff in (False, True):
            for dis_dyaw in (False, True):
                if enhanced_state_dim(n, dis_diff, dis_dyaw) == state_dim:
                    candidates.append(StateCfg(
                        legacy_state=False,
                        n_sectors=n,
                        sector_method="min",
                        disable_lidar_diff=dis_diff,
                        disable_delta_yaw=dis_dyaw,
                    ))

    if not candidates:
        raise RuntimeError(
            f"无法根据 state_dim={state_dim} 推断状态配置。"
            f"已知 legacy_dim={EnvConfig.LIDAR_RAYS + 2 + 2 + 1 + 2}，"
            f"enhanced_dim(8/16, diff/dyaw) 不匹配。"
        )

    # 规则：优先 16 扇区，其次 diff=True，其次 dyaw=True（更贴近默认 enhanced 配置）
    def score(c: StateCfg) -> Tuple[int, int, int]:
        return (
            1 if c.n_sectors == 16 else 0,
            1 if (not c.disable_lidar_diff) else 0,
            1 if (not c.disable_delta_yaw) else 0,
        )

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def try_get_state_meta(ckpt: Any) -> Optional[Dict[str, Any]]:
    if isinstance(ckpt, dict) and ("state_meta" in ckpt) and isinstance(ckpt["state_meta"], dict):
        return ckpt["state_meta"]
    return None


def state_cfg_from_meta(meta: Dict[str, Any], algo: str) -> StateCfg:
    legacy = bool(meta.get("legacy_state", False))
    if legacy:
        return StateCfg(legacy_state=True)
    return StateCfg(
        legacy_state=False,
        n_sectors=int(meta.get("n_sectors", 16)),
        sector_method=str(meta.get("sector_method", "min")),
        disable_lidar_diff=bool(meta.get("disable_lidar_diff", False)),
        disable_delta_yaw=bool(meta.get("disable_delta_yaw", False)),
    )


# -----------------------------
# Environment builder
# -----------------------------

class _EnvCounts:
    """临时修改 EnvConfig.NUM_STATIC_OBSTACLES / NUM_DYNAMIC_OBSTACLES。"""

    def __init__(self, n_static: int, n_dynamic: int):
        self.n_static = int(n_static)
        self.n_dynamic = int(n_dynamic)
        self._old_s = None
        self._old_d = None

    def __enter__(self):
        self._old_s = getattr(EnvConfig, "NUM_STATIC_OBSTACLES", 0)
        self._old_d = getattr(EnvConfig, "NUM_DYNAMIC_OBSTACLES", 0)
        EnvConfig.NUM_STATIC_OBSTACLES = self.n_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = self.n_dynamic

    def __exit__(self, exc_type, exc, tb):
        if self._old_s is not None:
            EnvConfig.NUM_STATIC_OBSTACLES = self._old_s
        if self._old_d is not None:
            EnvConfig.NUM_DYNAMIC_OBSTACLES = self._old_d


def build_env(
    state_cfg: StateCfg,
    n_static: int,
    n_dynamic: int,
    dynamic_speed_min: float,
    dynamic_speed_max: float,
    dynamic_patterns: Tuple[str, ...],
    dynamic_stop_prob: float,
) -> NavigationEnv:
    with _EnvCounts(n_static=n_static, n_dynamic=n_dynamic):
        env = NavigationEnv(
            **state_cfg.to_env_kwargs(),
            dynamic_speed_min=float(dynamic_speed_min),
            dynamic_speed_max=float(dynamic_speed_max),
            dynamic_patterns=tuple(dynamic_patterns),
            dynamic_stop_prob=float(dynamic_stop_prob),
        )
    return env


# -----------------------------
# Rendering (for GIF)
# -----------------------------

def _get_canvas_rgb(fig) -> np.ndarray:
    """兼容 matplotlib 3.8+：使用 buffer_rgba。"""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # (H,W,4)
    rgb = np.asarray(buf[..., :3], dtype=np.uint8)
    return rgb


def render_frame(
    env: NavigationEnv,
    robot_traj: List[Tuple[float, float]],
    dyn_traj: List[List[Tuple[float, float]]],
    title: str = "",
) -> np.ndarray:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(5, 5), dpi=120)
    ax = fig.add_subplot(111)

    ax.set_xlim(0, EnvConfig.MAP_WIDTH)
    ax.set_ylim(0, EnvConfig.MAP_HEIGHT)
    ax.set_aspect("equal")
    ax.set_title(title)

    # 边界
    ax.plot([0, EnvConfig.MAP_WIDTH, EnvConfig.MAP_WIDTH, 0, 0],
            [0, 0, EnvConfig.MAP_HEIGHT, EnvConfig.MAP_HEIGHT, 0],
            linewidth=1)

    # 目标点
    ax.plot([env.goal_x], [env.goal_y], marker="*", markersize=10)

    # 静态障碍（灰色）& 动态障碍（红色）
    # env.obstacles: list[Obstacle]
    for obs in env.obstacles:
        if getattr(obs, "is_dynamic", False):
            c = "r"
            alpha = 0.9
        else:
            c = "0.5"  # gray
            alpha = 0.8
        circ = plt.Circle((obs.x, obs.y), obs.radius, color=c, alpha=alpha)
        ax.add_patch(circ)

    # 动态障碍轨迹
    for k, traj in enumerate(dyn_traj):
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, linewidth=1)

    # 无人机轨迹
    if len(robot_traj) >= 2:
        xs = [p[0] for p in robot_traj]
        ys = [p[1] for p in robot_traj]
        ax.plot(xs, ys, linewidth=1)

    # 无人机本体
    rb = env.robot
    circ_r = plt.Circle((rb.x, rb.y), rb.radius, alpha=0.9)
    ax.add_patch(circ_r)

    # 速度方向箭头（按 vx,vy）
    # 视觉上做一个缩放（否则箭头太短）
    arrow_scale = 0.8
    ax.arrow(rb.x, rb.y, rb.vx * arrow_scale, rb.vy * arrow_scale,
             width=0.03, length_includes_head=True)

    # 动态障碍箭头
    dyn_idx = 0
    for obs in env.obstacles:
        if getattr(obs, "is_dynamic", False):
            ax.arrow(obs.x, obs.y, obs.vx * arrow_scale, obs.vy * arrow_scale,
                     width=0.02, length_includes_head=True)
            dyn_idx += 1

    ax.set_xticks([])
    ax.set_yticks([])

    img = _get_canvas_rgb(fig)
    plt.close(fig)
    return img


def save_gif(frames: List[np.ndarray], out_path: str, fps: int = 15) -> None:
    ensure_dir(os.path.dirname(out_path))
    duration = 1.0 / max(1, int(fps))

    # imageio 优先
    try:
        import imageio.v2 as imageio
        imageio.mimsave(out_path, frames, duration=duration)
        return
    except Exception:
        pass

    # Pillow fallback
    try:
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(
            out_path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(duration * 1000),
            loop=0,
        )
    except Exception as e:
        raise RuntimeError(f"保存 GIF 失败：{e}")


# -----------------------------
# Policy wrappers
# -----------------------------

class PolicyBase:
    name: str
    history_len: int = 1

    def reset(self) -> None:
        pass

    def act(self, state_seq: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class DDPGPolicy(PolicyBase):
    def __init__(self, model_path: str, state_dim: int, device: str = "cpu"):
        self.name = "DDPG"
        self.history_len = 1
        self.model_path = model_path
        self.state_dim = int(state_dim)
        self.device = device

        from ddpg import DDPGAgent

        # 用很小的 buffer/batch 也无所谓（只评测 act）
        self.agent = DDPGAgent(
            state_dim=self.state_dim,
            action_dim=2,
            random_seed=0,
            batch_size=64,
            buffer_size=1000,
        )
        self.agent.load(model_path)

    def act(self, state_seq: np.ndarray) -> np.ndarray:
        # state_seq 这里是 (state_dim,)
        a = self.agent.act(state_seq, step=0, add_noise=False)
        return action_clip(np.asarray(a, dtype=np.float32))


class LSTMPolicy(PolicyBase):
    def __init__(self, model_path: str, ckpt: dict, state_dim: int, device: str = "cpu"):
        self.name = "LSTM_DDPG"
        self.model_path = model_path
        self.device = device
        self.history_len = int(ckpt.get("history_len", 4))

        from lstm_ddpg import LSTMDdpgAgent
        import inspect

        net_cfg = ckpt.get("net_cfg", {}) if isinstance(ckpt, dict) else {}
        kwargs = dict(
            state_dim=int(state_dim),
            action_dim=2,
            history_len=self.history_len,
            batch_size=64,
            buffer_size=1000,
            embed_dim=int(net_cfg.get("embed_dim", 64)),
            lstm_hidden_dim=int(net_cfg.get("lstm_hidden_dim", 64)),
        )

        # 按签名过滤参数，避免版本差异
        sig = inspect.signature(LSTMDdpgAgent.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        self.agent = LSTMDdpgAgent(**kwargs)
        self.agent.load(model_path)

    def act(self, state_seq: np.ndarray) -> np.ndarray:
        a = self.agent.act(state_seq, step=0, add_noise=False)
        return action_clip(np.asarray(a, dtype=np.float32))


class LSTMAttPolicy(PolicyBase):
    def __init__(self, model_path: str, ckpt: dict, state_dim: int, device: str = "cpu"):
        self.name = "LSTM_DDPG_ATT"
        self.model_path = model_path
        self.device = device
        self.history_len = int(ckpt.get("history_len", 4))

        import inspect
        from lstm_ddpg_att import LSTMDdpgAgent

        net_cfg = ckpt.get("net_cfg", {}) if isinstance(ckpt, dict) else {}
        meta = try_get_state_meta(ckpt) or {}

        kwargs = dict(
            state_dim=int(state_dim),
            action_dim=2,
            history_len=self.history_len,
            batch_size=64,
            buffer_size=1000,
            embed_dim=int(net_cfg.get("embed_dim", 64)),
            lstm_hidden_dim=int(net_cfg.get("lstm_hidden_dim", 64)),
            mlp_hidden_dim=int(net_cfg.get("mlp_hidden_dim", 256)),
            use_spatial_att=bool(net_cfg.get("use_spatial_att", meta.get("use_spatial_att", True))),
            use_temporal_att=bool(net_cfg.get("use_temporal_att", meta.get("use_temporal_att", False))),
            sector_model_dim=int(net_cfg.get("sector_model_dim", meta.get("sector_model_dim", 32))),
            temporal_att_dim=int(net_cfg.get("temporal_att_dim", meta.get("temporal_att_dim", 64))),
            att_dropout=float(net_cfg.get("att_dropout", meta.get("att_dropout", 0.0))),
            sp_gate_init=float(net_cfg.get("sp_gate_init", meta.get("sp_gate_init", -4.0))),
            temporal_gate_init=float(net_cfg.get("temporal_gate_init", meta.get("temporal_gate_init", -4.0))),
            # heads：如果 agent 支持就传，否则忽略（避免你本地版本差异）
            spatial_att_heads=int(meta.get("spatial_att_heads", 4)),
            temporal_att_heads=int(meta.get("temporal_att_heads", 2)),
        )

        sig = inspect.signature(LSTMDdpgAgent.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

        self.agent = LSTMDdpgAgent(**kwargs)
        self.agent.load(model_path)

    def act(self, state_seq: np.ndarray) -> np.ndarray:
        a = self.agent.act(state_seq, step=0, add_noise=False)
        return action_clip(np.asarray(a, dtype=np.float32))


# -----------------------------
# Evaluation
# -----------------------------

@dataclass
class EvalResult:
    algo: str
    scenario: str
    episodes: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    avg_steps: float
    avg_reward: float
    dt_mean_ms: float
    dt_p50_ms: float
    dt_p90_ms: float
    dt_p95_ms: float
    gif_path: str = ""


@dataclass
class TrajectoryRollout:
    algo: str
    scenario: str
    reason: str
    steps: int
    total_reward: float
    robot_traj: List[Tuple[float, float]]
    init_obstacles: List[Tuple[float, float, float, bool]]
    goal_xy: Tuple[float, float]


def evaluate_policy(
    policy: PolicyBase,
    env: NavigationEnv,
    episodes: int,
    seed: int,
    scenario_name: str,
    save_gif_flag: bool,
    gif_out: str,
    gif_episodes: int,
    gif_fps: int,
    gif_stride: int,
) -> EvalResult:
    success = 0
    collision = 0
    timeout = 0

    ep_rewards: List[float] = []
    ep_steps: List[int] = []
    decision_times_ms: List[float] = []

    frames: List[np.ndarray] = []
    record_left = int(gif_episodes) if save_gif_flag else 0

    for ep in range(int(episodes)):
        set_global_seed(int(seed) + ep)
        state = env.reset()

        # LSTM：用 deque 维护 history
        if policy.history_len > 1:
            from collections import deque
            q = deque([state.copy() for _ in range(policy.history_len)], maxlen=policy.history_len)

        robot_traj: List[Tuple[float, float]] = [(env.robot.x, env.robot.y)]

        # 动态障碍轨迹：按动态障碍的出现顺序存
        dyn_indices = [i for i, o in enumerate(env.obstacles) if getattr(o, "is_dynamic", False)]
        dyn_traj: List[List[Tuple[float, float]]] = [[] for _ in dyn_indices]
        for k, idx in enumerate(dyn_indices):
            o = env.obstacles[idx]
            dyn_traj[k].append((o.x, o.y))

        ep_ret = 0.0
        done = False
        info = {"reason": None}

        if record_left > 0:
            frames.append(render_frame(env, robot_traj, dyn_traj, title=f"{policy.name} | {scenario_name} | ep{ep}"))

        while not done:
            if policy.history_len > 1:
                s_in = np.stack(list(q), axis=0)  # (H, state_dim)
            else:
                s_in = state

            t0 = time.perf_counter()
            a = policy.act(s_in)
            t1 = time.perf_counter()
            decision_times_ms.append((t1 - t0) * 1000.0)

            next_state, r, done, info = env.step(a)
            ep_ret += float(r)

            state = next_state
            if policy.history_len > 1:
                q.append(state.copy())

            robot_traj.append((env.robot.x, env.robot.y))
            for k, idx in enumerate(dyn_indices):
                o = env.obstacles[idx]
                dyn_traj[k].append((o.x, o.y))

            if record_left > 0 and (len(robot_traj) % max(1, int(gif_stride)) == 0):
                frames.append(render_frame(env, robot_traj, dyn_traj, title=f"{policy.name} | {scenario_name} | ep{ep}"))

        reason = info.get("reason", None)
        if reason == "goal_reached":
            success += 1
        elif reason in ("collision_obstacle", "collision_wall"):
            collision += 1
        elif reason == "max_steps":
            timeout += 1

        ep_rewards.append(ep_ret)
        ep_steps.append(int(env.step_count))

        if record_left > 0:
            record_left -= 1
            # 每个 episode 结束时补一帧
            frames.append(render_frame(env, robot_traj, dyn_traj, title=f"{policy.name} | {scenario_name} | ep{ep} (done:{reason})"))

    if save_gif_flag and len(frames) > 0:
        save_gif(frames, gif_out, fps=int(gif_fps))

    n = float(max(1, int(episodes)))
    res = EvalResult(
        algo=policy.name,
        scenario=scenario_name,
        episodes=int(episodes),
        success_rate=success / n,
        collision_rate=collision / n,
        timeout_rate=timeout / n,
        avg_steps=float(np.mean(ep_steps)) if ep_steps else float("nan"),
        avg_reward=float(np.mean(ep_rewards)) if ep_rewards else float("nan"),
        dt_mean_ms=float(np.mean(decision_times_ms)) if decision_times_ms else float("nan"),
        dt_p50_ms=percentile(decision_times_ms, 50),
        dt_p90_ms=percentile(decision_times_ms, 90),
        dt_p95_ms=percentile(decision_times_ms, 95),
        gif_path=gif_out if save_gif_flag else "",
    )
    return res


def rollout_single_episode(
    policy: PolicyBase,
    cfg: StateCfg,
    scenario_name: str,
    seed: int,
    n_static: int,
    n_dynamic: int,
    dynamic_speed_min: float,
    dynamic_speed_max: float,
    dynamic_patterns: Tuple[str, ...],
    dynamic_stop_prob: float,
    max_steps: int,
) -> TrajectoryRollout:
    """在指定场景跑 1 个 episode，返回机器人轨迹用于可视化对比。"""
    set_global_seed(int(seed))
    env = build_env(
        cfg,
        n_static=n_static,
        n_dynamic=n_dynamic,
        dynamic_speed_min=dynamic_speed_min,
        dynamic_speed_max=dynamic_speed_max,
        dynamic_patterns=dynamic_patterns,
        dynamic_stop_prob=dynamic_stop_prob,
    )

    state = env.reset()
    policy.reset()

    if policy.history_len > 1:
        from collections import deque
        q = deque([state.copy() for _ in range(policy.history_len)], maxlen=policy.history_len)

    robot_traj: List[Tuple[float, float]] = [(float(env.robot.x), float(env.robot.y))]
    init_obstacles: List[Tuple[float, float, float, bool]] = []
    for _, x, y, r, is_dyn, _, _ in _iter_obstacles_with_vel(env):
        init_obstacles.append((x, y, r, is_dyn))

    total_reward = 0.0
    done = False
    info: Dict[str, Any] = {"reason": None}

    step = 0
    while (not done) and (step < int(max_steps)):
        if policy.history_len > 1:
            s_in = np.stack(list(q), axis=0)
        else:
            s_in = state

        a = policy.act(s_in)
        next_state, reward, done, info = env.step(a)
        total_reward += float(reward)
        state = next_state

        if policy.history_len > 1:
            q.append(state.copy())

        robot_traj.append((float(env.robot.x), float(env.robot.y)))
        step += 1

    reason = str(info.get("reason", "max_steps"))
    if (not done) and step >= int(max_steps):
        reason = "max_steps"

    result = TrajectoryRollout(
        algo=policy.name,
        scenario=scenario_name,
        reason=reason,
        steps=step,
        total_reward=float(total_reward),
        robot_traj=robot_traj,
        init_obstacles=init_obstacles,
        goal_xy=(float(env.goal_x), float(env.goal_y)),
    )
    env.close()
    return result


def save_train_traj_compare_png(
    ddpg_rollout: TrajectoryRollout,
    att_rollout: TrajectoryRollout,
    out_path: str,
) -> None:
    import matplotlib.pyplot as plt

    w, h = float(EnvConfig.MAP_WIDTH), float(EnvConfig.MAP_HEIGHT)
    fig, ax = plt.subplots(1, 1, figsize=(6.6, 6.0), dpi=130)

    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], linewidth=1.0, color="black")

    gx, gy = ddpg_rollout.goal_xy
    goal = plt.Circle((gx, gy), radius=float(EnvConfig.GOAL_RADIUS), fill=False, linewidth=2, edgecolor="green")
    ax.add_patch(goal)

    for x, y, r, is_dyn in ddpg_rollout.init_obstacles:
        if is_dyn:
            circ = plt.Circle((x, y), radius=r, fill=True, alpha=0.30, facecolor="#ff7f7f", edgecolor="#d62728", linewidth=1)
        else:
            circ = plt.Circle((x, y), radius=r, fill=True, alpha=0.50, facecolor="0.7", edgecolor="0.55", linewidth=1)
        ax.add_patch(circ)

    def _plot_traj(traj: List[Tuple[float, float]], color: str, label: str):
        if len(traj) >= 2:
            xs = [p[0] for p in traj]
            ys = [p[1] for p in traj]
            ax.plot(xs, ys, color=color, linewidth=2.0, label=label)
            ax.scatter([xs[0]], [ys[0]], marker="o", s=20, color=color, alpha=0.9)
            ax.scatter([xs[-1]], [ys[-1]], marker="x", s=36, color=color, alpha=0.95)

    _plot_traj(
        ddpg_rollout.robot_traj,
        "#1f77b4",
        f"DDPG (steps={ddpg_rollout.steps}, reason={ddpg_rollout.reason})",
    )
    _plot_traj(
        att_rollout.robot_traj,
        "#ff7f0e",
        f"LSTM-DDPG-Attention (steps={att_rollout.steps}, reason={att_rollout.reason})",
    )

    ax.set_title("Training Scene Trajectory Compare: DDPG vs LSTM-DDPG-Attention")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def print_table(results: List[EvalResult]) -> None:
    # 统一对齐输出
    headers = [
        "Algo",
        "Scenario",
        "Succ%",
        "Coll%",
        "Tout%",
        "AvgSteps",
        "AvgReward",
        "DT(ms) mean/p50/p90/p95",
        "GIF",
    ]
    rows = []
    for r in results:
        rows.append([
            r.algo,
            r.scenario,
            f"{r.success_rate*100:6.2f}",
            f"{r.collision_rate*100:6.2f}",
            f"{r.timeout_rate*100:6.2f}",
            f"{r.avg_steps:8.2f}",
            f"{r.avg_reward:9.2f}",
            f"{r.dt_mean_ms:.3f}/{r.dt_p50_ms:.3f}/{r.dt_p90_ms:.3f}/{r.dt_p95_ms:.3f}",
            (r.gif_path if r.gif_path else "-"),
        ])

    colw = [max(len(str(x)) for x in col) for col in zip(headers, *rows)]

    def fmt_row(row):
        return " | ".join(str(v).ljust(w) for v, w in zip(row, colw))

    print("\n" + fmt_row(headers))
    print("-+-".join("-" * w for w in colw))
    for row in rows:
        print(fmt_row(row))
    print("")


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--ddpg_model", type=str, required=True)
    parser.add_argument("--lstm_model", type=str, required=True)
    parser.add_argument("--att_model", type=str, required=True)

    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)

    # 动态障碍参数（评测时统一使用该组参数）
    parser.add_argument("--dynamic_speed_min", type=float, default=0.3)
    parser.add_argument("--dynamic_speed_max", type=float, default=0.7)
    parser.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    parser.add_argument("--dynamic_stop_prob", type=float, default=0.05)

    # 场景障碍数量
    parser.add_argument("--base_static", type=int, default=int(getattr(EnvConfig, "NUM_STATIC_OBSTACLES", 4)))
    parser.add_argument("--base_dynamic", type=int, default=int(getattr(EnvConfig, "NUM_DYNAMIC_OBSTACLES", 2)))
    parser.add_argument("--extra_static", type=int, default=2)
    parser.add_argument("--extra_dynamic", type=int, default=2)

    # GIF
    parser.add_argument("--save_gif", action="store_true")
    parser.add_argument("--gif_dir", type=str, default="gifs")
    parser.add_argument("--gif_episodes", type=int, default=1)
    parser.add_argument("--gif_fps", type=int, default=15)
    parser.add_argument("--gif_stride", type=int, default=1)

    # 训练场景单回合轨迹对比（DDPG vs LSTM-DDPG-Attention）
    parser.add_argument("--save_train_traj_compare", action="store_true")
    parser.add_argument("--train_traj_out", type=str, default="ddpg_vs_lstmatt_train_traj.png")
    parser.add_argument("--train_traj_seed", type=int, default=None, help="轨迹对比使用的随机种子，默认沿用 --seed")
    parser.add_argument("--train_traj_max_steps", type=int, default=int(getattr(EnvConfig, "MAX_STEPS", 500)))
    parser.add_argument("--only_train_traj_compare", action="store_true", help="仅生成训练场景轨迹对比图，不跑完整评测")

    args = parser.parse_args()

    ensure_dir(args.gif_dir)

    dyn_patterns = parse_patterns(args.dynamic_patterns)

    print("=" * 70)
    print(
        f"Dynamic: {args.dynamic_speed_min:.2f}~{args.dynamic_speed_max:.2f}, "
        f"patterns={dyn_patterns}, stop_prob={args.dynamic_stop_prob}"
    )
    print(
        f"Base obstacles: static={args.base_static}, dynamic={args.base_dynamic}\n"
        f"Hard obstacles: static={args.base_static + args.extra_static}, dynamic={args.base_dynamic + args.extra_dynamic} (+{args.extra_static}S +{args.extra_dynamic}D)"
    )
    print("=" * 70)

    # ----------------- load ckpts -----------------
    ddpg_ckpt = safe_torch_load(args.ddpg_model, map_location="cpu")

    lstm_ckpt = safe_torch_load(args.lstm_model, map_location="cpu")
    if not isinstance(lstm_ckpt, dict) or ("actor_local" not in lstm_ckpt):
        raise RuntimeError("lstm_model 不是预期的 checkpoint(dict)，缺少 actor_local。")

    att_ckpt = safe_torch_load(args.att_model, map_location="cpu")
    if not isinstance(att_ckpt, dict) or ("actor_local" not in att_ckpt):
        raise RuntimeError("att_model 不是预期的 checkpoint(dict)，缺少 actor_local。")

    # ----------------- infer state cfg -----------------
    # DDPG：从 actor_local 第一层输入维度推断
    if isinstance(ddpg_ckpt, dict) and ("actor_local" in ddpg_ckpt):
        w = ddpg_ckpt["actor_local"]["fc1.weight"]
        ddpg_state_dim = int(w.shape[1])
    else:
        # 也可能是直接保存的 state_dict
        if isinstance(ddpg_ckpt, dict) and ("fc1.weight" in ddpg_ckpt):
            ddpg_state_dim = int(ddpg_ckpt["fc1.weight"].shape[1])
        else:
            raise RuntimeError("无法从 ddpg_model 推断 state_dim：未找到 fc1.weight。")

    ddpg_cfg = infer_state_cfg_from_state_dim(ddpg_state_dim)
    print(f"[DDPG] inferred state_dim={ddpg_state_dim} -> {ddpg_cfg}")

    # LSTM：优先 state_meta
    lstm_meta = try_get_state_meta(lstm_ckpt) or {}
    lstm_cfg = state_cfg_from_meta(lstm_meta, "LSTM_DDPG")
    lstm_state_dim = int(lstm_ckpt.get("net_cfg", {}).get("state_dim", 0))
    print(f"[LSTM_DDPG] Using StateCfg(from meta): {lstm_cfg}, net_state_dim={lstm_state_dim}")

    # ATT：优先 state_meta
    att_meta = try_get_state_meta(att_ckpt) or {}
    att_cfg = state_cfg_from_meta(att_meta, "LSTM_DDPG_ATT")
    att_state_dim = int(att_ckpt.get("net_cfg", {}).get("state_dim", 0))
    print(f"[LSTM_DDPG_ATT] Using StateCfg(from meta): {att_cfg}, net_state_dim={att_state_dim}")

    # sanity：确保 env.state_dim 和网络 state_dim 匹配（LSTM/ATT）
    env_tmp = build_env(
        lstm_cfg,
        n_static=args.base_static,
        n_dynamic=args.base_dynamic,
        dynamic_speed_min=args.dynamic_speed_min,
        dynamic_speed_max=args.dynamic_speed_max,
        dynamic_patterns=dyn_patterns,
        dynamic_stop_prob=args.dynamic_stop_prob,
    )
    if lstm_state_dim and env_tmp.state_dim != lstm_state_dim:
        print(
            f"[WARN] LSTM env.state_dim={env_tmp.state_dim} != ckpt net_cfg.state_dim={lstm_state_dim}. "
            f"如果加载或表现异常，请核对训练时的 state 配置。"
        )

    env_tmp2 = build_env(
        att_cfg,
        n_static=args.base_static,
        n_dynamic=args.base_dynamic,
        dynamic_speed_min=args.dynamic_speed_min,
        dynamic_speed_max=args.dynamic_speed_max,
        dynamic_patterns=dyn_patterns,
        dynamic_stop_prob=args.dynamic_stop_prob,
    )
    if att_state_dim and env_tmp2.state_dim != att_state_dim:
        print(
            f"[WARN] ATT env.state_dim={env_tmp2.state_dim} != ckpt net_cfg.state_dim={att_state_dim}. "
            f"如果加载或表现异常，请核对训练时的 state 配置。"
        )

    # ----------------- build policies -----------------
    # 注意：这些 agent 内部会创建 torch 模型；如果你想强制 GPU，可在各自模块里改 device
    ddpg_policy = DDPGPolicy(args.ddpg_model, state_dim=ddpg_state_dim)
    lstm_policy = LSTMPolicy(args.lstm_model, ckpt=lstm_ckpt, state_dim=env_tmp.state_dim)
    att_policy = LSTMAttPolicy(args.att_model, ckpt=att_ckpt, state_dim=env_tmp2.state_dim)

    policies: List[Tuple[PolicyBase, StateCfg]] = [
        (ddpg_policy, ddpg_cfg),
        (lstm_policy, lstm_cfg),
        (att_policy, att_cfg),
    ]

    # ----------------- optional: training-scene trajectory compare (DDPG vs ATT) -----------------
    if args.save_train_traj_compare or args.only_train_traj_compare:
        traj_seed = int(args.seed if args.train_traj_seed is None else args.train_traj_seed)
        print("\n" + "=" * 70)
        print(f"Trajectory compare on training scene (seed={traj_seed})")
        print("=" * 70)

        ddpg_rollout = rollout_single_episode(
            policy=ddpg_policy,
            cfg=ddpg_cfg,
            scenario_name="scene_train",
            seed=traj_seed,
            n_static=args.base_static,
            n_dynamic=args.base_dynamic,
            dynamic_speed_min=args.dynamic_speed_min,
            dynamic_speed_max=args.dynamic_speed_max,
            dynamic_patterns=dyn_patterns,
            dynamic_stop_prob=args.dynamic_stop_prob,
            max_steps=args.train_traj_max_steps,
        )
        att_rollout = rollout_single_episode(
            policy=att_policy,
            cfg=att_cfg,
            scenario_name="scene_train",
            seed=traj_seed,
            n_static=args.base_static,
            n_dynamic=args.base_dynamic,
            dynamic_speed_min=args.dynamic_speed_min,
            dynamic_speed_max=args.dynamic_speed_max,
            dynamic_patterns=dyn_patterns,
            dynamic_stop_prob=args.dynamic_stop_prob,
            max_steps=args.train_traj_max_steps,
        )

        save_train_traj_compare_png(ddpg_rollout, att_rollout, args.train_traj_out)
        print(f"Saved trajectory compare: {args.train_traj_out}")
        print(f"  DDPG: steps={ddpg_rollout.steps}, reason={ddpg_rollout.reason}, return={ddpg_rollout.total_reward:.2f}")
        print(f"  ATT : steps={att_rollout.steps}, reason={att_rollout.reason}, return={att_rollout.total_reward:.2f}")

        if args.only_train_traj_compare:
            return

    results: List[EvalResult] = []

    # ----------------- evaluate two scenarios -----------------
    scenarios = [
        ("scene_train", args.base_static, args.base_dynamic),
        ("scene_harder(+2S+2D)", args.base_static + args.extra_static, args.base_dynamic + args.extra_dynamic),
    ]

    for policy, cfg in policies:
        print("\n" + "=" * 70)
        print(f"Evaluating: {policy.name}")
        print("=" * 70)

        for scene_name, n_static, n_dynamic in scenarios:
            env = build_env(
                cfg,
                n_static=n_static,
                n_dynamic=n_dynamic,
                dynamic_speed_min=args.dynamic_speed_min,
                dynamic_speed_max=args.dynamic_speed_max,
                dynamic_patterns=dyn_patterns,
                dynamic_stop_prob=args.dynamic_stop_prob,
            )

            gif_path = ""
            if args.save_gif:
                safe_algo = policy.name.lower().replace("/", "_")
                safe_scene = scene_name.replace(" ", "").replace("(", "").replace(")", "").replace("+", "plus")
                gif_path = os.path.join(args.gif_dir, f"{safe_algo}_{safe_scene}.gif")

            r = evaluate_policy(
                policy=policy,
                env=env,
                episodes=args.episodes,
                seed=args.seed,
                scenario_name=scene_name,
                save_gif_flag=bool(args.save_gif),
                gif_out=gif_path,
                gif_episodes=args.gif_episodes,
                gif_fps=args.gif_fps,
                gif_stride=args.gif_stride,
            )
            results.append(r)

            # 逐场景即时打印
            print(f"[{policy.name} - {scene_name}]")
            print(f"  episodes         : {r.episodes}")
            print(f"  success_rate     : {r.success_rate*100:.2f}%")
            print(f"  collision_rate   : {r.collision_rate*100:.2f}%")
            print(f"  timeout_rate     : {r.timeout_rate*100:.2f}%")
            print(f"  avg_steps        : {r.avg_steps:.2f}")
            print(f"  avg_reward       : {r.avg_reward:.2f}")
            print(
                f"  decision_time(ms): mean={r.dt_mean_ms:.3f}, "
                f"p50={r.dt_p50_ms:.3f}, p90={r.dt_p90_ms:.3f}, p95={r.dt_p95_ms:.3f}"
            )
            if args.save_gif:
                print(f"  gif              : {gif_path}")
            print("")

    # ----------------- unified table -----------------
    print_table(results)


if __name__ == "__main__":
    main()
