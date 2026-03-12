#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_random_scenario.py

随机测试场景下 DDPG vs LSTM-DDPG-ATT 航迹规划对比脚本。

功能：
  - 起点、终点随机（每 episode 重新采样）
  - 障碍物大小、速度随机
  - 障碍物数量可通过命令行自定义（--n_static / --n_dynamic）
  - 颜色与 visualize_two_scenarios.py 完全一致
  - 动态障碍物运动方向箭头 + 历史轨迹渐变折线
  - 输出：控制台指标表 + 航迹对比图(PNG) + 指标柱状图(PNG)

用法：
  # 仅展示随机场景（不跑模型）
  python test_random_scenario.py --n_static 5 --n_dynamic 4 --show_only

  # 完整对比（50 episodes）
  python test_random_scenario.py \\
    --model_dir models \\
    --ddpg_model ddpg_best.pth \\
    --att_model  lstm_ddpg_att_best.pth \\
    --n_static 5 --n_dynamic 4 \\
    --episodes 50 --seed 42 \\
    --save_fig results/compare.png

  # 单跑 DDPG（不需要 att 模型）
  python test_random_scenario.py \\
    --model_dir models \\
    --ddpg_model ddpg_best.pth \\
    --n_static 5 --n_dynamic 4 --episodes 30
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.collections as mc
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import numpy as np

from config import EnvConfig
from environment import NavigationEnv


# ─── 颜色常量（与 visualize_two_scenarios.py 完全一致） ──────────────────────
COLOR_STATIC_FACE   = "0.7"        # 灰色填充
COLOR_STATIC_EDGE   = "0.55"       # 灰色边框
COLOR_DYNAMIC_FACE  = "red"        # 红色填充
COLOR_DYNAMIC_EDGE  = "red"        # 红色边框
COLOR_DYN_TRAJ      = "red"        # 动态障碍物轨迹颜色
COLOR_DYN_ARROW_FC  = "darkred"    # 动态障碍物方向箭头颜色
COLOR_ROBOT         = "#1f77b4"    # 蓝色机器人
COLOR_GOAL_FACE     = "#2ca02c"    # 绿色目标
COLOR_BOUNDARY      = "black"      # 边界框

# 机器人航迹颜色（两种算法区分）
COLOR_TRAJ_DDPG     = "#ff7f0e"    # 橙色  DDPG
COLOR_TRAJ_ATT      = "#9467bd"    # 紫色  LSTM-DDPG-ATT

DYNAMIC_TRAJ_KEEP   = 60           # 动态障碍物轨迹保留步数


# ─── 随机种子 ────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


# ─── 环境构建 ────────────────────────────────────────────────────────────────

class TempObstacleCounts:
    def __init__(self, n_static: int, n_dynamic: int):
        self.ns, self.nd = int(n_static), int(n_dynamic)
        self.os = int(EnvConfig.NUM_STATIC_OBSTACLES)
        self.od = int(EnvConfig.NUM_DYNAMIC_OBSTACLES)

    def __enter__(self):
        EnvConfig.NUM_STATIC_OBSTACLES  = self.ns
        EnvConfig.NUM_DYNAMIC_OBSTACLES = self.nd
        return self

    def __exit__(self, *_):
        EnvConfig.NUM_STATIC_OBSTACLES  = self.os
        EnvConfig.NUM_DYNAMIC_OBSTACLES = self.od


def _build_env(args, enhanced: bool) -> NavigationEnv:
    enhanced_cfg = {
        "n_sectors":     int(args.n_sectors),
        "sector_method": "min",
        "use_lidar_diff": not args.no_lidar_diff,
        "use_delta_yaw":  not args.no_delta_yaw,
    }
    return NavigationEnv(
        use_enhanced_state=enhanced,
        enhanced_state_config=enhanced_cfg if enhanced else {},
        dynamic_speed_min=float(args.dynamic_speed_min),
        dynamic_speed_max=float(args.dynamic_speed_max),
        dynamic_patterns=tuple(
            s.strip() for s in args.dynamic_patterns.split(",") if s.strip()
        ),
        dynamic_stop_prob=float(args.dynamic_stop_prob),
    )


def build_envs(args):
    """构建两个环境（DDPG 用 legacy state，ATT 用 enhanced state），
    数量由 args.n_static / n_dynamic 控制。"""
    ns = int(args.n_static)
    nd = int(args.n_dynamic)
    with TempObstacleCounts(ns, nd):
        env_ddpg = _build_env(args, enhanced=False)
        env_att  = _build_env(args, enhanced=True)
    return env_ddpg, env_att


# ─── 障碍物工具 ──────────────────────────────────────────────────────────────

def _iter_obstacles(env: NavigationEnv):
    for obs in getattr(env, "obstacles", []):
        if isinstance(obs, dict):
            yield (float(obs.get("x", 0)), float(obs.get("y", 0)),
                   float(obs.get("radius", 0.2)), bool(obs.get("is_dynamic", False)))
        else:
            yield (float(obs.x), float(obs.y), float(obs.radius),
                   bool(getattr(obs, "is_dynamic", False)))


def _get_goal_xy(env: NavigationEnv) -> Tuple[float, float]:
    if hasattr(env, "goal"):
        try:
            gx, gy = env.goal
            return float(gx), float(gy)
        except Exception:
            pass
    return float(getattr(env, "goal_x")), float(getattr(env, "goal_y"))


def _get_heading(obs, traj: Optional[List] = None) -> Tuple[float, float]:
    vx = float(getattr(obs, "vx", 0.0))
    vy = float(getattr(obs, "vy", 0.0))
    norm = np.hypot(vx, vy)
    if norm > 1e-8:
        return vx / norm, vy / norm
    if traj and len(traj) >= 2:
        dx, dy = traj[-1][0] - traj[-2][0], traj[-1][1] - traj[-2][1]
        norm = np.hypot(dx, dy)
        if norm > 1e-8:
            return dx / norm, dy / norm
    return 0.0, 0.0


def _init_dyn_trajs(env: NavigationEnv) -> Dict[int, List]:
    trajs: Dict[int, List] = {}
    for idx, obs in enumerate(getattr(env, "obstacles", [])):
        is_dyn = bool(getattr(obs, "is_dynamic", False)) if not isinstance(obs, dict) \
                 else bool(obs.get("is_dynamic", False))
        if is_dyn:
            x = float(obs.x if not isinstance(obs, dict) else obs.get("x", 0))
            y = float(obs.y if not isinstance(obs, dict) else obs.get("y", 0))
            trajs[idx] = [(x, y)]
    return trajs


# ─── 绘图核心 ────────────────────────────────────────────────────────────────

def draw_env(
    ax,
    env: NavigationEnv,
    title: str,
    dyn_trajs: Optional[Dict[int, List]] = None,
    robot_traj: Optional[List[Tuple[float, float]]] = None,
    traj_color: str = COLOR_TRAJ_DDPG,
    outcome: str = "",
) -> None:
    """绘制场景底图、动态轨迹、机器人航迹。"""
    W = float(EnvConfig.MAP_WIDTH)
    H = float(EnvConfig.MAP_HEIGHT)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")
    ax.set_facecolor("#f9f9f9")
    ax.set_xticks([])
    ax.set_yticks([])

    # ── 标题颜色反映结果 ────────────────────────────────────────────────────
    title_color = {"success": "#2ca02c", "collision": "#d62728",
                   "timeout": "#ff7f0e"}.get(outcome, "black")
    ax.set_title(title, fontsize=9, color=title_color, fontweight="bold")

    # ── 边界框 ──────────────────────────────────────────────────────────────
    ax.plot([0, W, W, 0, 0], [0, 0, H, H, 0],
            color=COLOR_BOUNDARY, linewidth=1.5, zorder=0)

    # ── 目标（绿色空心圆 + 星标） ────────────────────────────────────────────
    gx, gy = _get_goal_xy(env)
    goal_r = float(EnvConfig.GOAL_RADIUS)
    ax.add_patch(Circle((gx, gy), goal_r,
                         fill=True, facecolor=COLOR_GOAL_FACE, edgecolor=COLOR_GOAL_FACE,
                         alpha=0.20, linewidth=2, zorder=3))
    ax.add_patch(Circle((gx, gy), goal_r,
                         fill=False, edgecolor=COLOR_GOAL_FACE,
                         linewidth=2, zorder=4))
    ax.plot(gx, gy, marker="*", color=COLOR_GOAL_FACE,
            markersize=12, zorder=5, linestyle="None")

    # ── 动态障碍物历史轨迹（alpha 渐变红色折线段） ──────────────────────────
    if dyn_trajs:
        for pts in dyn_trajs.values():
            if len(pts) < 2:
                continue
            n_seg = len(pts) - 1
            segs = [[(pts[i][0], pts[i][1]), (pts[i+1][0], pts[i+1][1])]
                    for i in range(n_seg)]
            alphas = np.linspace(0.06, 0.80, n_seg)
            colors_rgba = [mcolors.to_rgba(COLOR_DYN_TRAJ, alpha=float(a))
                           for a in alphas]
            lc = mc.LineCollection(segs, colors=colors_rgba, linewidths=1.4, zorder=2)
            ax.add_collection(lc)

    # ── 障碍物（静态=灰色，动态=红色） ──────────────────────────────────────
    for x, y, r, is_dyn in _iter_obstacles(env):
        if is_dyn:
            circ = Circle((x, y), r, fill=True, alpha=0.35,
                           facecolor=COLOR_DYNAMIC_FACE, edgecolor=COLOR_DYNAMIC_EDGE,
                           linewidth=1.2, zorder=3)
        else:
            circ = Circle((x, y), r, fill=True, alpha=0.55,
                           facecolor=COLOR_STATIC_FACE, edgecolor=COLOR_STATIC_EDGE,
                           linewidth=1.2, zorder=3)
        ax.add_patch(circ)

    # ── 动态障碍物运动方向箭头（darkred） ────────────────────────────────────
    for idx, obs in enumerate(getattr(env, "obstacles", [])):
        if isinstance(obs, dict) or not bool(getattr(obs, "is_dynamic", False)):
            continue
        ux, uy = _get_heading(obs, None if dyn_trajs is None else dyn_trajs.get(idx))
        if abs(ux) + abs(uy) < 1e-8:
            continue
        obs_r = float(getattr(obs, "radius", 0.2))
        arrow_len = max(0.25, obs_r * 2.2)
        ox, oy = float(obs.x), float(obs.y)
        ax.annotate("",
                    xy=(ox + ux * arrow_len, oy + uy * arrow_len),
                    xytext=(ox, oy),
                    arrowprops=dict(arrowstyle="-|>", color=COLOR_DYN_ARROW_FC,
                                    lw=1.5, mutation_scale=11),
                    zorder=6)

    # ── 机器人航迹 ───────────────────────────────────────────────────────────
    if robot_traj and len(robot_traj) >= 2:
        xs, ys = zip(*robot_traj)
        ax.plot(xs, ys, "-", color=traj_color, linewidth=2.0,
                alpha=0.92, zorder=7)
        ax.plot(xs[0],  ys[0],  "o", color=traj_color, markersize=7,
                markeredgecolor="white", markeredgewidth=0.8, zorder=8)
        ax.plot(xs[-1], ys[-1], "s", color=traj_color, markersize=8,
                markeredgecolor="k",     markeredgewidth=0.5, zorder=8)

    # ── 机器人当前位置 ───────────────────────────────────────────────────────
    rx, ry = float(env.robot.x), float(env.robot.y)
    rr = float(env.robot.radius)
    ax.add_patch(Circle((rx, ry), rr, fill=True, alpha=0.90,
                          facecolor=COLOR_ROBOT, edgecolor="white",
                          linewidth=1.0, zorder=9))
    heading = float(getattr(env.robot, "heading", 0.0))
    hx = rx + rr * 1.8 * math.cos(heading)
    hy = ry + rr * 1.8 * math.sin(heading)
    ax.plot([rx, hx], [ry, hy], color="white", linewidth=1.2, zorder=10)


# ─── 数据结构 ────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    algo:        str
    outcome:     str        # success / collision / timeout
    steps:       int
    total_reward: float
    dt_mean_ms:  float
    robot_traj:  List[Tuple[float, float]] = field(default_factory=list)
    dyn_trajs:   Dict[int, List]           = field(default_factory=dict)
    env_snapshot: Optional[object]         = field(default=None, repr=False)


@dataclass
class EvalSummary:
    algo:           str
    n:              int
    success_rate:   float
    collision_rate: float
    timeout_rate:   float
    avg_steps:      float
    avg_reward:     float
    dt_mean_ms:     float
    dt_p50_ms:      float
    dt_p90_ms:      float


# ─── 公共推理辅助 ────────────────────────────────────────────────────────────

def _torch_load(path: str):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _classify(info: dict) -> str:
    r = info.get("reason", "")
    if r == "goal_reached":   return "success"
    if "collision" in r:      return "collision"
    return "timeout"


def _record_step(env, robot_traj, dyn_trajs, dyn_indices):
    robot_traj.append((float(env.robot.x), float(env.robot.y)))
    for k, idx in enumerate(dyn_indices):
        obs = env.obstacles[idx]
        pt  = (float(obs.x), float(obs.y))
        dyn_trajs[k].append(pt)
        if len(dyn_trajs[k]) > DYNAMIC_TRAJ_KEEP:
            dyn_trajs[k] = dyn_trajs[k][-DYNAMIC_TRAJ_KEEP:]


# ─── DDPG 评估 ───────────────────────────────────────────────────────────────

def eval_ddpg(
    model_path: str,
    env: NavigationEnv,
    n_episodes: int,
    seed: int,
    record_n: int = 3,
) -> Tuple[EvalSummary, List[RunResult]]:
    from ddpg import DDPGAgent

    ckpt     = _torch_load(model_path)
    actor_sd = ckpt.get("actor_local", ckpt)
    state_dim  = int(actor_sd["linear1.weight"].shape[1])
    hidden_dim = int(actor_sd["linear1.weight"].shape[0])

    agent = DDPGAgent(state_dim=state_dim, action_dim=2, hidden_dim=hidden_dim)
    agent.actor_local.load_state_dict(actor_sd, strict=True)
    agent.actor_local.eval()

    outcomes, steps_list, rewards, dt_all = [], [], [], []
    results: List[RunResult] = []

    for ep in range(n_episodes):
        set_seed(seed + ep)
        state   = env.reset()
        dyn_idx = [i for i, o in enumerate(env.obstacles) if getattr(o, "is_dynamic", False)]
        r_traj  = [(float(env.robot.x), float(env.robot.y))]
        d_trajs = {k: [(float(env.obstacles[i].x), float(env.obstacles[i].y))]
                   for k, i in enumerate(dyn_idx)}
        ep_ret = 0.0; done = False; info = {}

        for step in range(EnvConfig.MAX_STEPS):
            t0 = time.perf_counter()
            a  = agent.act(state, step=0, add_noise=False).reshape(-1)
            dt_all.append((time.perf_counter() - t0) * 1000.0)

            state, r, done, info = env.step(a)
            ep_ret += r
            _record_step(env, r_traj, d_trajs, dyn_idx)
            if done:
                break

        out = _classify(info)
        outcomes.append(out); steps_list.append(step + 1); rewards.append(ep_ret)

        if ep < record_n:
            # 拷贝动态轨迹（转 dict indexed by int）
            dyn_save = {idx: list(pts) for idx, pts in d_trajs.items()}
            results.append(RunResult(
                algo="DDPG", outcome=out,
                steps=step + 1, total_reward=ep_ret,
                dt_mean_ms=float(np.mean(dt_all)) if dt_all else 0.0,
                robot_traj=list(r_traj),
                dyn_trajs=dyn_save,
                env_snapshot=env,
            ))

    n = n_episodes
    return EvalSummary(
        algo="DDPG", n=n,
        success_rate=outcomes.count("success") / n,
        collision_rate=outcomes.count("collision") / n,
        timeout_rate=outcomes.count("timeout") / n,
        avg_steps=float(np.mean(steps_list)),
        avg_reward=float(np.mean(rewards)),
        dt_mean_ms=float(np.mean(dt_all)),
        dt_p50_ms=float(np.percentile(dt_all, 50)),
        dt_p90_ms=float(np.percentile(dt_all, 90)),
    ), results


# ─── LSTM-DDPG-ATT 评估 ──────────────────────────────────────────────────────

def _try_load_att_agent(model_path, env, module_name):
    """尝试用指定模块加载 ATT agent，成功返回 (agent, hist_len)，失败抛异常。"""
    import importlib, inspect
    mod           = importlib.import_module(module_name)
    LSTMDdpgAgent = mod.LSTMDdpgAgent

    ckpt     = _torch_load(model_path)
    net_cfg  = ckpt.get("net_cfg", {}) or {}
    hist_len = int(ckpt.get("history_len", net_cfg.get("history_len", 3)))

    sig    = inspect.signature(LSTMDdpgAgent.__init__)
    kwargs = dict(
        state_dim         = int(net_cfg.get("state_dim", env.state_dim)),
        action_dim        = 2,
        history_len       = hist_len,
        batch_size        = 64,
        buffer_size       = 1000,
        embed_dim         = int(net_cfg.get("embed_dim",          64)),
        lstm_hidden_dim   = int(net_cfg.get("lstm_hidden_dim",    64)),
        use_spatial_att   = bool(net_cfg.get("use_spatial_att",   True)),
        use_temporal_att  = bool(net_cfg.get("use_temporal_att",  True)),
        sector_model_dim  = int(net_cfg.get("sector_model_dim",   32)),
        temporal_att_dim  = int(net_cfg.get("temporal_att_dim",   64)),
        spatial_att_heads = int(net_cfg.get("spatial_att_heads",   4)),
        temporal_att_heads= int(net_cfg.get("temporal_att_heads",  4)),
    )
    kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    agent = LSTMDdpgAgent(**kwargs)
    agent.load(model_path, load_optimizers=False)  # strict=True，失败会抛 RuntimeError
    agent.actor_local.eval()
    return agent, hist_len


def eval_lstm_att(
    model_path: str,
    env: NavigationEnv,
    n_episodes: int,
    seed: int,
    record_n: int = 3,
    att_module: str = "lstm_ddpg_att1",
) -> Tuple[EvalSummary, List[RunResult]]:
    # 优先用命令行指定模块；若 strict load 失败，自动尝试另一个版本
    candidates = [att_module] + [m for m in ["lstm_ddpg_att1", "lstm_ddpg_att"]
                                  if m != att_module]
    agent, hist_len, last_err = None, 3, None
    for mod_name in candidates:
        try:
            agent, hist_len = _try_load_att_agent(model_path, env, mod_name)
            print(f"  [ATT] 加载成功，使用模块: {mod_name}  history_len={hist_len}")
            break
        except Exception as e:
            last_err = e
            print(f"  [ATT] {mod_name} 不匹配: {type(e).__name__}: {str(e)[:100]}…")
    if agent is None:
        raise RuntimeError(f"所有模块均无法加载 {model_path}，最后错误: {last_err}")

    outcomes, steps_list, rewards, dt_all = [], [], [], []
    results: List[RunResult] = []

    for ep in range(n_episodes):
        set_seed(seed + ep)
        state  = env.reset()
        q      = deque([state.copy() for _ in range(hist_len)], maxlen=hist_len)
        dyn_idx = [i for i, o in enumerate(env.obstacles) if getattr(o, "is_dynamic", False)]
        r_traj  = [(float(env.robot.x), float(env.robot.y))]
        d_trajs = {k: [(float(env.obstacles[i].x), float(env.obstacles[i].y))]
                   for k, i in enumerate(dyn_idx)}
        ep_ret = 0.0; done = False; info = {}

        for step in range(EnvConfig.MAX_STEPS):
            state_seq = np.stack(list(q), axis=0)
            t0 = time.perf_counter()
            a  = agent.act(state_seq, step=0, add_noise=False).reshape(-1)
            dt_all.append((time.perf_counter() - t0) * 1000.0)

            state, r, done, info = env.step(a)
            q.append(state.copy())
            ep_ret += r
            _record_step(env, r_traj, d_trajs, dyn_idx)
            if done:
                break

        out = _classify(info)
        outcomes.append(out); steps_list.append(step + 1); rewards.append(ep_ret)

        if ep < record_n:
            dyn_save = {idx: list(pts) for idx, pts in d_trajs.items()}
            results.append(RunResult(
                algo="LSTM-DDPG-ATT", outcome=out,
                steps=step + 1, total_reward=ep_ret,
                dt_mean_ms=float(np.mean(dt_all)) if dt_all else 0.0,
                robot_traj=list(r_traj),
                dyn_trajs=dyn_save,
                env_snapshot=env,
            ))

    n = n_episodes
    return EvalSummary(
        algo="LSTM-DDPG-ATT", n=n,
        success_rate=outcomes.count("success") / n,
        collision_rate=outcomes.count("collision") / n,
        timeout_rate=outcomes.count("timeout") / n,
        avg_steps=float(np.mean(steps_list)),
        avg_reward=float(np.mean(rewards)),
        dt_mean_ms=float(np.mean(dt_all)),
        dt_p50_ms=float(np.percentile(dt_all, 50)),
        dt_p90_ms=float(np.percentile(dt_all, 90)),
    ), results


# ─── 可视化输出 ──────────────────────────────────────────────────────────────

OUTCOME_LABEL = {
    "success":   "✔ 成功",
    "collision": "✖ 碰撞",
    "timeout":   "⏱ 超时",
}
OUTCOME_COLOR = {
    "success":   "#2ca02c",
    "collision": "#d62728",
    "timeout":   "#ff7f0e",
}


def plot_trajectory_comparison(
    runs_ddpg: List[RunResult],
    runs_att:  List[RunResult],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """2×N 子图：每列一个 episode，上行 DDPG，下行 LSTM-DDPG-ATT。"""
    n_col = min(len(runs_ddpg), len(runs_att), 3)
    if n_col == 0:
        print("[WARNING] 没有足够的运行结果，跳过航迹对比图。")
        return None

    fig, axes = plt.subplots(2, n_col, figsize=(5.5 * n_col, 11), dpi=120)
    if n_col == 1:
        axes = axes.reshape(2, 1)

    algo_rows = [
        ("DDPG",          runs_ddpg, COLOR_TRAJ_DDPG),
        ("LSTM-DDPG-ATT", runs_att,  COLOR_TRAJ_ATT),
    ]

    for row, (algo_name, runs, traj_clr) in enumerate(algo_rows):
        for col in range(n_col):
            ax  = axes[row][col]
            run = runs[col]

            out_label = OUTCOME_LABEL.get(run.outcome, run.outcome)
            title = (f"{algo_name}  Episode {col + 1}\n"
                     f"{out_label}  |  步数={run.steps}  |  奖励={run.total_reward:.1f}")

            # 使用 run.env_snapshot 绘制场景；若无快照则仅绘航迹
            if run.env_snapshot is not None:
                draw_env(
                    ax, run.env_snapshot,
                    title=title,
                    dyn_trajs={i: list(pts) for i, pts in run.dyn_trajs.items()},
                    robot_traj=run.robot_traj,
                    traj_color=traj_clr,
                    outcome=run.outcome,
                )
            else:
                ax.set_title(title, fontsize=9, fontweight="bold",
                             color=OUTCOME_COLOR.get(run.outcome, "black"))

            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            ax.tick_params(labelsize=7)

    # ── 图例 ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=COLOR_STATIC_FACE,  edgecolor=COLOR_STATIC_EDGE,
                       label="静态障碍物"),
        mpatches.Patch(facecolor=COLOR_DYNAMIC_FACE, edgecolor=COLOR_DYNAMIC_EDGE,
                       alpha=0.4, label="动态障碍物"),
        mpatches.Patch(facecolor=COLOR_GOAL_FACE,    alpha=0.3,  label="目标区域"),
        mpatches.Patch(facecolor=COLOR_ROBOT,        label="机器人"),
        plt.Line2D([0], [0], color=COLOR_TRAJ_DDPG, linewidth=2, label="DDPG 航迹"),
        plt.Line2D([0], [0], color=COLOR_TRAJ_ATT,  linewidth=2, label="LSTM-DDPG-ATT 航迹"),
        plt.Line2D([0], [0], color=COLOR_DYN_TRAJ,  linewidth=1.5,
                   alpha=0.7, label="动态障碍物轨迹"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               fontsize=9, framealpha=0.90, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("随机测试场景航迹规划对比：DDPG vs LSTM-DDPG-ATT",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[图] 航迹对比图 → {save_path}")
    return fig


def plot_metrics_bar(
    summ_ddpg: EvalSummary,
    summ_att:  EvalSummary,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """指标柱状图：任务结果率 + 步数/奖励/决策延迟。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=120)
    w = 0.35

    # ── 左：成功/碰撞/超时率 ──────────────────────────────────────────────
    labels = ["成功率", "碰撞率", "超时率"]
    ddpg_v = [summ_ddpg.success_rate * 100,
               summ_ddpg.collision_rate * 100,
               summ_ddpg.timeout_rate * 100]
    att_v  = [summ_att.success_rate * 100,
               summ_att.collision_rate * 100,
               summ_att.timeout_rate * 100]
    x = np.arange(len(labels))

    b1 = ax1.bar(x - w / 2, ddpg_v, w, label="DDPG",          color=COLOR_TRAJ_DDPG, alpha=0.85)
    b2 = ax1.bar(x + w / 2, att_v,  w, label="LSTM-DDPG-ATT", color=COLOR_TRAJ_ATT,  alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_ylabel("比率 (%)", fontsize=10)
    ax1.set_ylim(0, 115)
    ax1.legend(fontsize=10)
    ax1.set_title("任务结果对比", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 1.5,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    # ── 右：平均步数 / 平均奖励 / 决策延迟 ───────────────────────────────
    labels2   = ["平均步数", "平均奖励", "决策延迟(ms)\n×10"]
    ddpg_m = [summ_ddpg.avg_steps, summ_ddpg.avg_reward, summ_ddpg.dt_mean_ms * 10]
    att_m  = [summ_att.avg_steps,  summ_att.avg_reward,  summ_att.dt_mean_ms  * 10]
    x2 = np.arange(len(labels2))

    b3 = ax2.bar(x2 - w / 2, ddpg_m, w, label="DDPG",          color=COLOR_TRAJ_DDPG, alpha=0.85)
    b4 = ax2.bar(x2 + w / 2, att_m,  w, label="LSTM-DDPG-ATT", color=COLOR_TRAJ_ATT,  alpha=0.85)
    ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.set_title("性能指标对比", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)

    # 在柱子上标真实值（延迟要 /10）
    for bar, (d_val, a_val) in zip(list(b3) + list(b4),
                                    [(ddpg_m[i], att_m[i]) for i in range(3)] * 2):
        h   = bar.get_height()
        idx = list(b3).index(bar) if bar in b3 else list(b4).index(bar)
        scale = 0.1 if idx == 2 else 1.0
        real  = h * scale
        fmt   = f"{real:.2f}" if scale < 1 else f"{real:.1f}"
        ax2.text(bar.get_x() + bar.get_width() / 2, h + max(abs(h) * 0.01, 0.5),
                 fmt, ha="center", va="bottom", fontsize=8)

    # 额外标注 p50/p90
    info_text = (f"DDPG  mean={summ_ddpg.dt_mean_ms:.2f}ms  "
                 f"p50={summ_ddpg.dt_p50_ms:.2f}ms  p90={summ_ddpg.dt_p90_ms:.2f}ms\n"
                 f"ATT   mean={summ_att.dt_mean_ms:.2f}ms   "
                 f"p50={summ_att.dt_p50_ms:.2f}ms  p90={summ_att.dt_p90_ms:.2f}ms")
    ax2.text(0.5, -0.18, info_text, transform=ax2.transAxes,
             ha="center", fontsize=8, color="#444",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    fig.suptitle(f"随机场景性能汇总  (n={summ_ddpg.n} episodes/algo)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save_path:
        bar_path = save_path.replace(".png", "_metrics.png")
        os.makedirs(os.path.dirname(bar_path) if os.path.dirname(bar_path) else ".", exist_ok=True)
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        print(f"[图] 指标对比图 → {bar_path}")
    return fig


def plot_scene_only(env: NavigationEnv, args) -> plt.Figure:
    """仅展示随机场景（无航迹）。"""
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    draw_env(ax, env,
             title=f"随机测试场景  (Static={args.n_static}, Dynamic={args.n_dynamic})")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")

    legend_handles = [
        mpatches.Patch(facecolor=COLOR_ROBOT,        label="机器人（随机起点）"),
        mpatches.Patch(facecolor=COLOR_STATIC_FACE,  edgecolor=COLOR_STATIC_EDGE,
                       label=f"静态障碍物 ({args.n_static}个)"),
        mpatches.Patch(facecolor=COLOR_DYNAMIC_FACE, edgecolor=COLOR_DYNAMIC_EDGE,
                       alpha=0.4, label=f"动态障碍物 ({args.n_dynamic}个)"),
        mpatches.Patch(facecolor=COLOR_GOAL_FACE, alpha=0.4, label="目标（随机位置）"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    return fig


def print_summary_table(summaries: List[EvalSummary]) -> None:
    print("\n" + "=" * 82)
    print(f"{'算法':<20} {'成功率':>7} {'碰撞率':>7} {'超时率':>7} "
          f"{'平均步数':>9} {'平均奖励':>9} {'决策mean(ms)':>13} {'p50':>7} {'p90':>7}")
    print("-" * 82)
    for s in summaries:
        print(f"{s.algo:<20} "
              f"{s.success_rate*100:6.1f}% "
              f"{s.collision_rate*100:6.1f}% "
              f"{s.timeout_rate*100:6.1f}%  "
              f"{s.avg_steps:8.1f}  "
              f"{s.avg_reward:8.1f}  "
              f"{s.dt_mean_ms:12.3f}  "
              f"{s.dt_p50_ms:6.3f}  "
              f"{s.dt_p90_ms:6.3f}")
    print("=" * 82 + "\n")


# ─── 命令行 ──────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="随机测试场景下 DDPG vs LSTM-DDPG-ATT 航迹规划对比"
    )
    # ── 模型路径 ──
    p.add_argument("--model_dir",   type=str, default="models",
                   help="模型文件目录（默认 models/）")
    p.add_argument("--ddpg_model",  type=str, default="ddpg_best.pth",
                   help="DDPG 模型文件名（相对 model_dir）")
    p.add_argument("--att_model",   type=str, default="lstm_ddpg_att_best.pth",
                   help="LSTM-DDPG-ATT 模型文件名（相对 model_dir）")
    p.add_argument("--att_module",  type=str, default="lstm_ddpg_att1",
                   choices=["lstm_ddpg_att", "lstm_ddpg_att1"],
                   help="LSTM-DDPG-ATT Python 模块名")

    # ── 场景参数 ──
    p.add_argument("--n_static",  type=int, default=5,
                   help="静态障碍物数量（默认 5）")
    p.add_argument("--n_dynamic", type=int, default=4,
                   help="动态障碍物数量（默认 4）")
    p.add_argument("--dynamic_speed_min", type=float, default=0.30)
    p.add_argument("--dynamic_speed_max", type=float, default=0.70)
    p.add_argument("--dynamic_patterns",  type=str,   default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)

    # ── 评估参数 ──
    p.add_argument("--episodes",   type=int, default=50)
    p.add_argument("--seed",       type=int, default=465)
    p.add_argument("--record_n",   type=int, default=3,
                   help="记录前 N 个 episode 用于航迹图（默认 3）")

    # ── 状态维度 ──
    p.add_argument("--n_sectors",    type=int, default=16, choices=[8, 16])
    p.add_argument("--no_lidar_diff", action="store_true")
    p.add_argument("--no_delta_yaw",  action="store_true")

    # ── 输出 ──
    p.add_argument("--show_only", action="store_true",
                   help="仅展示随机场景，不运行算法")
    p.add_argument("--save_fig",  type=str, default="results/compare.png",
                   help="对比图保存路径")

    return p.parse_args()


# ─── 主程序 ──────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # 拼接完整模型路径
    ddpg_path = os.path.join(args.model_dir, args.ddpg_model)
    att_path  = os.path.join(args.model_dir, args.att_model)

    print(f"[配置] Static={args.n_static}  Dynamic={args.n_dynamic}  "
          f"Episodes={args.episodes}  Seed={args.seed}")
    print(f"[模型] DDPG → {ddpg_path}")
    print(f"[模型] ATT  → {att_path}")

    # ── 仅展示场景 ──────────────────────────────────────────────────────────
    if args.show_only:
        with TempObstacleCounts(args.n_static, args.n_dynamic):
            env = _build_env(args, enhanced=True)
        env.reset()
        fig = plot_scene_only(env, args)
        out = args.save_fig.replace(".png", "_scene.png") if args.save_fig else "results/scene.png"
        os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"[图] 场景图 → {out}")
        plt.close(fig)
        env.close()
        return

    # ── 自动检测 DDPG 模型的 state_dim，决定用哪种环境 ─────────────────────
    ddpg_enhanced = True   # 默认先假设 enhanced（43维）
    if os.path.exists(ddpg_path):
        try:
            _ckpt    = _torch_load(ddpg_path)
            _actor   = _ckpt.get("actor_local", _ckpt)
            _sdim    = int(_actor["linear1.weight"].shape[1])
            # 若 state_dim == 39 则是 legacy（无 enhanced），否则用 enhanced
            ddpg_enhanced = (_sdim != 39)
            print(f"[DDPG] 检测到 state_dim={_sdim}  "
                  f"→ {'enhanced' if ddpg_enhanced else 'legacy'} state")
        except Exception as e:
            print(f"[DDPG] 检测 state_dim 失败（{e}），默认使用 enhanced state")

    # ── 构建环境 ────────────────────────────────────────────────────────────
    with TempObstacleCounts(args.n_static, args.n_dynamic):
        env_ddpg = _build_env(args, enhanced=ddpg_enhanced)
        env_att  = _build_env(args, enhanced=True)

    print(f"[环境] DDPG state_dim={env_ddpg.state_dim}  "
          f"ATT state_dim={env_att.state_dim}")

    summaries: List[EvalSummary] = []
    all_runs: Dict[str, List[RunResult]] = {}

    # ── DDPG ────────────────────────────────────────────────────────────────
    if os.path.exists(ddpg_path):
        print(f"\n[DDPG] 开始评估 ({args.episodes} episodes)…")
        t0 = time.time()
        summ, runs = eval_ddpg(
            ddpg_path, env_ddpg,
            n_episodes=args.episodes,
            seed=args.seed,
            record_n=args.record_n,
        )
        print(f"  完成，耗时 {time.time()-t0:.1f}s  "
              f"成功率={summ.success_rate*100:.1f}%  "
              f"碰撞率={summ.collision_rate*100:.1f}%")
        summaries.append(summ)
        all_runs["DDPG"] = runs
    else:
        print(f"[DDPG] 模型文件不存在: {ddpg_path}，跳过。")

    env_ddpg.close()

    # ── LSTM-DDPG-ATT ───────────────────────────────────────────────────────
    if os.path.exists(att_path):
        print(f"\n[LSTM-DDPG-ATT] 开始评估 ({args.episodes} episodes)…")
        t0 = time.time()
        summ, runs = eval_lstm_att(
            att_path, env_att,
            n_episodes=args.episodes,
            seed=args.seed,
            record_n=args.record_n,
            att_module=args.att_module,
        )
        print(f"  完成，耗时 {time.time()-t0:.1f}s  "
              f"成功率={summ.success_rate*100:.1f}%  "
              f"碰撞率={summ.collision_rate*100:.1f}%")
        summaries.append(summ)
        all_runs["LSTM-DDPG-ATT"] = runs
    else:
        print(f"[LSTM-DDPG-ATT] 模型文件不存在: {att_path}，跳过。")

    env_att.close()

    # ── 打印汇总表 ──────────────────────────────────────────────────────────
    if summaries:
        print_summary_table(summaries)

    # ── 可视化输出 ──────────────────────────────────────────────────────────
    if "DDPG" in all_runs and "LSTM-DDPG-ATT" in all_runs:
        fig1 = plot_trajectory_comparison(
            all_runs["DDPG"], all_runs["LSTM-DDPG-ATT"],
            save_path=args.save_fig)
        if fig1:
            plt.close(fig1)

        if len(summaries) == 2:
            fig2 = plot_metrics_bar(summaries[0], summaries[1],
                                     save_path=args.save_fig)
            if fig2:
                plt.close(fig2)

    elif summaries:
        # 只有一种算法时，输出单算法航迹图
        algo = list(all_runs.keys())[0]
        run  = all_runs[algo][0]
        if run.env_snapshot is not None:
            fig, ax = plt.subplots(figsize=(6, 6), dpi=130)
            clr = COLOR_TRAJ_DDPG if algo == "DDPG" else COLOR_TRAJ_ATT
            draw_env(ax, run.env_snapshot,
                     title=f"{algo}  Ep1 | {OUTCOME_LABEL.get(run.outcome, run.outcome)}",
                     dyn_trajs=run.dyn_trajs,
                     robot_traj=run.robot_traj,
                     traj_color=clr,
                     outcome=run.outcome)
            plt.tight_layout()
            out = args.save_fig or f"results/{algo.lower()}_traj.png"
            os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"[图] 单算法航迹图 → {out}")
            plt.close()

    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()