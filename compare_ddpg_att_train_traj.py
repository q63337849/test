#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立轨迹对比脚本（不依赖/不修改现有测试脚本）

功能：
- 在训练场景中，对 DDPG 与 LSTM-DDPG-Attention 各跑 1 个 episode
- 保存同图轨迹对比 PNG
- 支持模型放在新目录（--model_dir）

用法：
python compare_ddpg_att_train_traj.py \
  --ddpg_model ddpg_best.pth \
  --att_model lstm_ddpg_att_best.pth \
  --model_dir models \
  --out ddpg_vs_att_train_traj.png
"""

from __future__ import annotations

import argparse
import inspect
import os
import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import EnvConfig
from environment import NavigationEnv


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


@dataclass
class StateCfg:
    legacy_state: bool
    n_sectors: int = 16
    sector_method: str = "min"
    disable_lidar_diff: bool = False
    disable_delta_yaw: bool = False

    def to_env_kwargs(self) -> Dict[str, Any]:
        if self.legacy_state:
            return dict(use_enhanced_state=False)
        return dict(
            use_enhanced_state=True,
            enhanced_state_config={
                "n_sectors": int(self.n_sectors),
                "sector_method": str(self.sector_method),
                "use_lidar_diff": (not self.disable_lidar_diff),
                "use_delta_yaw": (not self.disable_delta_yaw),
            },
        )


def enhanced_state_dim(n_sectors: int, disable_lidar_diff: bool, disable_delta_yaw: bool) -> int:
    base = n_sectors + 3 + 2 + 2 + 1 + 1
    if not disable_lidar_diff:
        base += n_sectors
    if not disable_delta_yaw:
        base += 2
    return int(base)


def infer_state_cfg_from_state_dim(state_dim: int) -> StateCfg:
    legacy_dim = EnvConfig.LIDAR_RAYS + 2 + 2 + 1 + 2
    if state_dim == legacy_dim:
        return StateCfg(legacy_state=True)

    cands: List[StateCfg] = []
    for n in (8, 16):
        for dis_diff in (False, True):
            for dis_dyaw in (False, True):
                if enhanced_state_dim(n, dis_diff, dis_dyaw) == state_dim:
                    cands.append(StateCfg(False, n, "min", dis_diff, dis_dyaw))
    if not cands:
        raise RuntimeError(f"无法从 state_dim={state_dim} 推断状态配置")
    cands.sort(key=lambda c: (1 if c.n_sectors == 16 else 0, 1 if not c.disable_lidar_diff else 0, 1 if not c.disable_delta_yaw else 0), reverse=True)
    return cands[0]


def infer_ddpg_state_dim_from_ckpt(ckpt: Any) -> int:
    def actor_sd(obj: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        if isinstance(obj.get("actor_local"), dict):
            return obj["actor_local"]
        return obj

    sd = actor_sd(ckpt)
    if not isinstance(sd, dict):
        raise RuntimeError("ddpg checkpoint 格式不支持")

    preferred = ["fc1.weight", "linear1.weight", "actor.fc1.weight", "actor.linear1.weight", "mlp.0.weight", "net.0.weight"]
    for k in preferred:
        w = sd.get(k)
        if getattr(w, "ndim", None) == 2:
            return int(w.shape[1])

    dims: List[int] = []
    for v in sd.values():
        if getattr(v, "ndim", None) == 2:
            dims.append(int(v.shape[1]))
    if dims:
        return min(dims)
    raise RuntimeError("无法从 DDPG checkpoint 推断 state_dim")


def state_cfg_from_meta(meta: Dict[str, Any]) -> StateCfg:
    legacy = bool(meta.get("legacy_state", False))
    if legacy:
        return StateCfg(True)
    return StateCfg(
        False,
        n_sectors=int(meta.get("n_sectors", 16)),
        sector_method=str(meta.get("sector_method", "min")),
        disable_lidar_diff=bool(meta.get("disable_lidar_diff", False)),
        disable_delta_yaw=bool(meta.get("disable_delta_yaw", False)),
    )


class DDPGPolicy:
    history_len = 1
    name = "DDPG"

    def __init__(self, model_path: str, state_dim: int):
        from ddpg import DDPGAgent

        kwargs = dict(state_dim=int(state_dim), action_dim=2, random_seed=0, batch_size=64, buffer_size=1000)
        sig = inspect.signature(DDPGAgent.__init__)
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
        self.agent = DDPGAgent(**kwargs)
        self.agent.load(model_path)

    def act(self, state_or_seq: np.ndarray) -> np.ndarray:
        a = self.agent.act(state_or_seq, step=0, add_noise=False)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        a[0] = np.clip(a[0], 0.0, EnvConfig.MAX_LINEAR_VEL)
        a[1] = np.clip(a[1], -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL)
        return a


class AttPolicy:
    name = "LSTM-DDPG-Attention"

    def __init__(self, model_path: str, ckpt: dict, state_dim: int):
        from lstm_ddpg_att import LSTMDdpgAgent

        self.history_len = int(ckpt.get("history_len", 4))
        net_cfg = ckpt.get("net_cfg", {}) if isinstance(ckpt, dict) else {}
        kwargs = dict(
            state_dim=int(state_dim),
            action_dim=2,
            history_len=self.history_len,
            batch_size=64,
            buffer_size=1000,
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

    def act(self, state_or_seq: np.ndarray) -> np.ndarray:
        a = self.agent.act(state_or_seq, step=0, add_noise=False)
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        a[0] = np.clip(a[0], 0.0, EnvConfig.MAX_LINEAR_VEL)
        a[1] = np.clip(a[1], -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL)
        return a


@dataclass
class Rollout:
    traj: List[Tuple[float, float]]
    reason: str
    steps: int
    ret: float
    obstacles0: List[Tuple[float, float, float, bool]]
    goal: Tuple[float, float]


def make_env(cfg: StateCfg, speed_min: float, speed_max: float, patterns: Tuple[str, ...], stop_prob: float) -> NavigationEnv:
    return NavigationEnv(
        **cfg.to_env_kwargs(),
        dynamic_speed_min=float(speed_min),
        dynamic_speed_max=float(speed_max),
        dynamic_patterns=tuple(patterns),
        dynamic_stop_prob=float(stop_prob),
    )


def rollout_one(policy: Any, cfg: StateCfg, seed: int, max_steps: int, speed_min: float, speed_max: float, patterns: Tuple[str, ...], stop_prob: float) -> Rollout:
    set_seed(seed)
    env = make_env(cfg, speed_min, speed_max, patterns, stop_prob)
    s = env.reset()
    q = None
    if getattr(policy, "history_len", 1) > 1:
        q = deque([s.copy() for _ in range(policy.history_len)], maxlen=policy.history_len)

    traj = [(float(env.robot.x), float(env.robot.y))]
    obstacles0 = []
    for obs in getattr(env, "obstacles", []):
        if isinstance(obs, dict):
            obstacles0.append((float(obs.get("x", 0.0)), float(obs.get("y", 0.0)), float(obs.get("radius", 0.2)), bool(obs.get("is_dynamic", False))))
        else:
            obstacles0.append((float(getattr(obs, "x")), float(getattr(obs, "y")), float(getattr(obs, "radius")), bool(getattr(obs, "is_dynamic", False))))

    done = False
    info = {"reason": None}
    ret = 0.0
    step = 0
    while (not done) and step < int(max_steps):
        sin = np.stack(list(q), axis=0) if q is not None else s
        a = policy.act(sin)
        s, r, done, info = env.step(a)
        ret += float(r)
        if q is not None:
            q.append(s.copy())
        traj.append((float(env.robot.x), float(env.robot.y)))
        step += 1

    reason = str(info.get("reason", "max_steps"))
    if (not done) and step >= int(max_steps):
        reason = "max_steps"

    out = Rollout(
        traj=traj,
        reason=reason,
        steps=step,
        ret=float(ret),
        obstacles0=obstacles0,
        goal=(float(env.goal_x), float(env.goal_y)),
    )
    env.close()
    return out


def save_compare_png(ddpg: Rollout, att: Rollout, out_path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6.8, 6.2), dpi=130)
    w, h = float(EnvConfig.MAP_WIDTH), float(EnvConfig.MAP_HEIGHT)
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], linewidth=1.0, color="black")

    gx, gy = ddpg.goal
    ax.add_patch(plt.Circle((gx, gy), radius=float(EnvConfig.GOAL_RADIUS), fill=False, linewidth=2, edgecolor="green"))

    for x, y, r, is_dyn in ddpg.obstacles0:
        c = "#ff7f7f" if is_dyn else "0.7"
        ec = "#d62728" if is_dyn else "0.55"
        alpha = 0.30 if is_dyn else 0.50
        ax.add_patch(plt.Circle((x, y), radius=r, fill=True, alpha=alpha, facecolor=c, edgecolor=ec, linewidth=1))

    def draw(traj, color, label):
        if len(traj) < 2:
            return
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        ax.plot(xs, ys, color=color, linewidth=2.0, label=label)
        ax.scatter([xs[0]], [ys[0]], marker="o", s=24, color=color)
        ax.scatter([xs[-1]], [ys[-1]], marker="x", s=36, color=color)

    draw(ddpg.traj, "#1f77b4", f"DDPG (steps={ddpg.steps}, {ddpg.reason})")
    draw(att.traj, "#ff7f0e", f"LSTM-DDPG-Att (steps={att.steps}, {att.reason})")

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Training Scene Trajectory Compare: DDPG vs LSTM-DDPG-Attention")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ddpg_model", type=str, required=True)
    p.add_argument("--att_model", type=str, required=True)
    p.add_argument("--model_dir", type=str, default="")
    p.add_argument("--out", type=str, default="ddpg_vs_att_train_traj.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=int(getattr(EnvConfig, "MAX_STEPS", 500)))
    p.add_argument("--dynamic_speed_min", type=float, default=0.3)
    p.add_argument("--dynamic_speed_max", type=float, default=0.7)
    p.add_argument("--dynamic_patterns", type=str, default="bounce,random_walk")
    p.add_argument("--dynamic_stop_prob", type=float, default=0.05)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    ddpg_path = resolve_model_path(args.ddpg_model, args.model_dir)
    att_path = resolve_model_path(args.att_model, args.model_dir)

    patterns = tuple([x.strip() for x in str(args.dynamic_patterns).split(",") if x.strip()])

    ddpg_ckpt = safe_torch_load(ddpg_path, map_location="cpu")
    att_ckpt = safe_torch_load(att_path, map_location="cpu")
    if not isinstance(att_ckpt, dict) or ("actor_local" not in att_ckpt):
        raise RuntimeError("att_model 不是预期 checkpoint(dict)，缺少 actor_local")

    ddpg_state_dim = infer_ddpg_state_dim_from_ckpt(ddpg_ckpt)

    att_meta = att_ckpt.get("state_meta", {}) if isinstance(att_ckpt, dict) else {}
    att_cfg = state_cfg_from_meta(att_meta if isinstance(att_meta, dict) else {})

    # 若 meta 不可信，则回退到 state_dim 推断
    env_tmp = make_env(att_cfg, args.dynamic_speed_min, args.dynamic_speed_max, patterns, args.dynamic_stop_prob)
    if int(env_tmp.state_dim) != int(ddpg_state_dim):
        att_cfg = infer_state_cfg_from_state_dim(ddpg_state_dim)
    env_tmp.close()

    ddpg_policy = DDPGPolicy(ddpg_path, state_dim=ddpg_state_dim)
    att_policy = AttPolicy(att_path, ckpt=att_ckpt, state_dim=ddpg_state_dim)

    ddpg_roll = rollout_one(ddpg_policy, att_cfg, args.seed, args.max_steps, args.dynamic_speed_min, args.dynamic_speed_max, patterns, args.dynamic_stop_prob)
    att_roll = rollout_one(att_policy, att_cfg, args.seed, args.max_steps, args.dynamic_speed_min, args.dynamic_speed_max, patterns, args.dynamic_stop_prob)

    save_compare_png(ddpg_roll, att_roll, args.out)

    print(f"Saved: {args.out}")
    print(f"DDPG: steps={ddpg_roll.steps}, reason={ddpg_roll.reason}, return={ddpg_roll.ret:.2f}")
    print(f"ATT : steps={att_roll.steps}, reason={att_roll.reason}, return={att_roll.ret:.2f}")


if __name__ == "__main__":
    main()
