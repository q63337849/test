#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_lstm_ddpg_att.py

在两个场景评估 LSTM-DDPG(+Attention) 模型：
1) 训练场景（使用 checkpoint 记录的环境/状态配置，或命令行覆盖）
2) 在训练场景基础上 +2 静态障碍 +2 动态障碍（可通过 --add_static/--add_dynamic 修改）

输出：成功率/碰撞率/超时率/平均步数/实时决策时间(均值与P95)
 python train_lstm_ddpg_att.py  --num_episodes 3000  --history_len 10  --n_sectors 16 --sector_method min  --update_every 4 --update_times 2  --embed_dim 96 --lstm_hidden_dim 96  --batch_size 128 --buffer_size 600000 --learn_start 3000  --use_spatial_att --use_temporal_att  --sector_model_dim 48 --spatial_att_heads 4  --temporal_att_dim 64 --temporal_att_heads 4  --att_dropout 0.1  --dynamic_speed_min 0.3 --dynamic_speed_max 0.7  --dynamic_patterns bounce,random_walk  --dynamic_stop_prob 0.05  --save_interval 500

用法示例：
  python test_lstm_ddpg_att.py --model models/lstm_ddpg_att_best.pth --episodes 200
  python test_lstm_ddpg_att.py --model models/lstm_ddpg_att_final.pth --episodes 500 --seed 123

说明：
- 默认会优先从 checkpoint['state_meta'] 读取训练时的 enhanced state 与动态障碍配置；
  你也可以用命令行参数覆盖。
- 评估时默认 add_noise=False（确定性策略）。
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

import torch

# ===== PyTorch>=2.6 checkpoint load compatibility =====
# PyTorch 2.6 起 torch.load 默认 weights_only=True，可能拒绝反序列化某些对象（例如 builtins.slice）。
# 你的 checkpoint 来自本项目训练流程，可视为可信来源；这里做兼容加载。
from contextlib import nullcontext

def _safe_torch_load(path: str, map_location=None):
    try:
        from torch.serialization import safe_globals
        ctx = safe_globals([slice])
    except Exception:
        ctx = nullcontext()

    with ctx:
        try:
            return torch.load(path, map_location=map_location, weights_only=True)
        except TypeError:
            return torch.load(path, map_location=map_location)
        except Exception:
            try:
                return torch.load(path, map_location=map_location, weights_only=False)
            except TypeError:
                return torch.load(path, map_location=map_location)


from config import EnvConfig
from environment import NavigationEnv


def _parse_bool_optional(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return None


def _split_patterns(s: str) -> Tuple[str, ...]:
    if not s:
        return tuple()
    return tuple([p.strip() for p in str(s).split(",") if p.strip()])


@dataclass
class ScenarioResult:
    episodes: int
    success: int
    collision: int
    timeout: int
    avg_steps: float
    decision_time_ms_mean: float
    decision_time_ms_p95: float
    decision_time_ms_p50: float
    decision_time_ms_p90: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episodes": self.episodes,
            "success": self.success,
            "collision": self.collision,
            "timeout": self.timeout,
            "success_rate": self.success / self.episodes if self.episodes else 0.0,
            "collision_rate": self.collision / self.episodes if self.episodes else 0.0,
            "timeout_rate": self.timeout / self.episodes if self.episodes else 0.0,
            "avg_steps": float(self.avg_steps),
            "decision_time_ms_mean": float(self.decision_time_ms_mean),
            "decision_time_ms_p50": float(self.decision_time_ms_p50),
            "decision_time_ms_p90": float(self.decision_time_ms_p90),
            "decision_time_ms_p95": float(self.decision_time_ms_p95),
        }


def _percentile_ms(times_s: List[float], q: float) -> float:
    if not times_s:
        return 0.0
    arr = np.asarray(times_s, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, q))


def _build_env_from_meta(meta: Dict[str, Any], overrides: argparse.Namespace) -> Tuple[NavigationEnv, Dict[str, Any]]:
    # 从 checkpoint state_meta 读取；若命令行给了覆盖则用覆盖
    legacy_state = bool(meta.get("legacy_state", False))
    if overrides.legacy_state:
        legacy_state = True

    n_sectors = int(meta.get("n_sectors", 16))
    if overrides.n_sectors is not None:
        n_sectors = int(overrides.n_sectors)

    sector_method = str(meta.get("sector_method", "min"))
    if overrides.sector_method is not None:
        sector_method = str(overrides.sector_method)

    disable_lidar_diff = bool(meta.get("disable_lidar_diff", False))
    if overrides.disable_lidar_diff:
        disable_lidar_diff = True

    disable_delta_yaw = bool(meta.get("disable_delta_yaw", False))
    if overrides.disable_delta_yaw:
        disable_delta_yaw = True

    dynamic_speed_min = float(meta.get("dynamic_speed_min", 0.30))
    if overrides.dynamic_speed_min is not None:
        dynamic_speed_min = float(overrides.dynamic_speed_min)

    dynamic_speed_max = float(meta.get("dynamic_speed_max", 0.70))
    if overrides.dynamic_speed_max is not None:
        dynamic_speed_max = float(overrides.dynamic_speed_max)

    patterns_str = meta.get("dynamic_patterns", "bounce,random_walk")
    dynamic_patterns = _split_patterns(patterns_str)
    if overrides.dynamic_patterns is not None:
        dynamic_patterns = _split_patterns(overrides.dynamic_patterns)

    dynamic_stop_prob = float(meta.get("dynamic_stop_prob", 0.05))
    if overrides.dynamic_stop_prob is not None:
        dynamic_stop_prob = float(overrides.dynamic_stop_prob)

    enhanced_cfg = {
        "n_sectors": n_sectors,
        "sector_method": sector_method,
        "use_lidar_diff": (not disable_lidar_diff),
        "use_delta_yaw": (not disable_delta_yaw),
    }

    env = NavigationEnv(
        use_enhanced_state=(not legacy_state),
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=dynamic_speed_min,
        dynamic_speed_max=dynamic_speed_max,
        dynamic_patterns=dynamic_patterns,
        dynamic_stop_prob=dynamic_stop_prob,
        disable_lidar_diff=disable_lidar_diff,
        disable_delta_yaw=disable_delta_yaw,
    )

    resolved = {
        "legacy_state": legacy_state,
        "n_sectors": n_sectors,
        "sector_method": sector_method,
        "disable_lidar_diff": disable_lidar_diff,
        "disable_delta_yaw": disable_delta_yaw,
        "dynamic_speed_min": dynamic_speed_min,
        "dynamic_speed_max": dynamic_speed_max,
        "dynamic_patterns": ",".join(dynamic_patterns),
        "dynamic_stop_prob": dynamic_stop_prob,
    }
    return env, resolved


def _construct_agent_from_checkpoint(ckpt: Dict[str, Any], env_state_dim: int, env_action_dim: int):
    # 延迟导入，避免脚本在缺少文件时直接崩
    from lstm_ddpg_att import LSTMDdpgAgent  # 你的工程目录下应存在

    net_cfg = ckpt.get("net_cfg", {}) or {}
    state_meta = ckpt.get("state_meta", {}) or {}
    actor_sd = ckpt.get("actor_local", {}) or {}

    # 优先使用 checkpoint 顶层字段，其次 net_cfg，再其次 state_meta
    history_len = int(ckpt.get("history_len", net_cfg.get("history_len", state_meta.get("history_len", 5))))
    embed_dim = int(net_cfg.get("embed_dim", state_meta.get("embed_dim", 64)))
    lstm_hidden_dim = int(net_cfg.get("lstm_hidden_dim", state_meta.get("lstm_hidden_dim", 64)))
    hidden_dim = int(net_cfg.get("mlp_hidden_dim", state_meta.get("hidden_dim", 256)))
    max_lin_vel = float(net_cfg.get("max_lin_vel", EnvConfig.MAX_LINEAR_VEL))
    max_ang_vel = float(net_cfg.get("max_ang_vel", EnvConfig.MAX_ANGULAR_VEL))

    # Attention flags: if missing in metadata, infer from state_dict keys to avoid architecture mismatch
    if ("use_spatial_att" in net_cfg) or ("use_spatial_att" in state_meta):
        use_spatial_att = bool(net_cfg.get("use_spatial_att", state_meta.get("use_spatial_att", True)))
    else:
        use_spatial_att = any(("spatial_att" in k) or ("spatial_proj" in k) for k in actor_sd.keys())

    if ("use_temporal_att" in net_cfg) or ("use_temporal_att" in state_meta):
        use_temporal_att = bool(net_cfg.get("use_temporal_att", state_meta.get("use_temporal_att", True)))
    else:
        use_temporal_att = any("temporal_att" in k for k in actor_sd.keys())

    # dims: prefer metadata; if missing, infer from tensor shapes where possible
    sector_model_dim = int(net_cfg.get("sector_model_dim", state_meta.get("sector_model_dim", 32)))
    if ("sector_model_dim" not in net_cfg) and ("sector_model_dim" not in state_meta):
        w = actor_sd.get("encoder.spatial_proj.weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            sector_model_dim = int(w.shape[1])

    temporal_att_dim = int(net_cfg.get("temporal_att_dim", state_meta.get("temporal_att_dim", 64)))
    if ("temporal_att_dim" not in net_cfg) and ("temporal_att_dim" not in state_meta):
        w = actor_sd.get("temporal_att.k_proj.weight", None)
        if isinstance(w, torch.Tensor) and w.ndim == 2:
            temporal_att_dim = int(w.shape[0])

    # 兼容“多头/Dropout”版本（如果 checkpoint 里有就带上）
    spatial_att_heads = net_cfg.get("spatial_att_heads", state_meta.get("spatial_att_heads", None))
    temporal_att_heads = net_cfg.get("temporal_att_heads", state_meta.get("temporal_att_heads", None))
    att_dropout = net_cfg.get("att_dropout", state_meta.get("att_dropout", None))

    # 构造 kwargs，并按实际 __init__ 签名过滤，避免“unexpected keyword”
    kwargs = {
        "state_dim": int(env_state_dim),
        "action_dim": int(env_action_dim),
        "history_len": int(history_len),
        "embed_dim": int(embed_dim),
        "lstm_hidden_dim": int(lstm_hidden_dim),
        "hidden_dim": int(hidden_dim),
        "max_lin_vel": float(max_lin_vel),
        "max_ang_vel": float(max_ang_vel),
        "use_spatial_att": bool(use_spatial_att),
        "use_temporal_att": bool(use_temporal_att),
        "sector_model_dim": int(sector_model_dim),
        "temporal_att_dim": int(temporal_att_dim),
        # 推理不依赖的训练参数给默认值即可
        "batch_size": 64,
        "buffer_size": 10000,
        "update_every": 4,
        "update_times": 1,
    }

    if spatial_att_heads is not None:
        kwargs["spatial_att_heads"] = int(spatial_att_heads)
    if temporal_att_heads is not None:
        kwargs["temporal_att_heads"] = int(temporal_att_heads)
    if att_dropout is not None:
        kwargs["att_dropout"] = float(att_dropout)

    sig = inspect.signature(LSTMDdpgAgent.__init__)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}

    agent = LSTMDdpgAgent(**filtered)
    return agent


def _eval_env(agent, env: NavigationEnv, episodes: int, seed: int, add_noise: bool = False) -> ScenarioResult:
    # 决策时间（每步）
    decision_times_s: List[float] = []

    success = 0
    collision = 0
    timeout = 0
    steps_list: List[int] = []

    history_len = int(getattr(agent, "history_len", 5))

    # 预热（尤其是 CUDA，避免把首次 kernel init 计入统计）
    try:
        s0 = env.reset()
        q0 = np.asarray([s0.copy() for _ in range(history_len)], dtype=np.float32)
        for _ in range(10):
            _ = agent.act(q0, step=0, add_noise=False)
    except Exception:
        pass

    global_step = 0

    for ep in range(episodes):
        # 每个 episode 设不同 seed，保证可复现但不“死重复”
        np.random.seed(seed + ep)

        state = env.reset()
        state_queue = deque([state.copy() for _ in range(history_len)], maxlen=history_len)

        done = False
        info: Dict[str, Any] = {}

        for step_i in range(EnvConfig.MAX_STEPS):
            global_step += 1
            state_seq = np.asarray(state_queue, dtype=np.float32)

            # CUDA 计时需要同步
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass

            t0 = time.perf_counter()
            action = agent.act(state_seq, step=global_step, add_noise=add_noise)
            # act 内部通常会把 tensor .cpu()，但这里仍然同步一次保证计时准确
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            t1 = time.perf_counter()

            decision_times_s.append(t1 - t0)

            action_flat = np.asarray(action, dtype=np.float32).reshape(-1)
            next_state, reward, done, info = env.step(action_flat)

            state_queue.append(next_state.copy())
            state = next_state

            if done:
                steps_list.append(step_i + 1)
                break

        if not done:
            # 理论上不会发生（MAX_STEPS 会终止），但兜底
            steps_list.append(EnvConfig.MAX_STEPS)
            info = {"reason": "max_steps"}

        reason = str(info.get("reason", ""))
        if reason == "goal_reached":
            success += 1
        elif reason in ("collision_obstacle", "collision_wall"):
            collision += 1
        else:
            timeout += 1

    avg_steps = float(np.mean(steps_list)) if steps_list else 0.0

    # 决策时间统计
    mean_ms = float(np.mean(np.asarray(decision_times_s, dtype=np.float64) * 1000.0)) if decision_times_s else 0.0
    p50 = _percentile_ms(decision_times_s, 50)
    p90 = _percentile_ms(decision_times_s, 90)
    p95 = _percentile_ms(decision_times_s, 95)

    return ScenarioResult(
        episodes=episodes,
        success=success,
        collision=collision,
        timeout=timeout,
        avg_steps=avg_steps,
        decision_time_ms_mean=mean_ms,
        decision_time_ms_p95=p95,
        decision_time_ms_p50=p50,
        decision_time_ms_p90=p90,
    )


def _print_result(title: str, r: ScenarioResult) -> None:
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    print(f"Episodes: {r.episodes}")
    print(f"Success rate : {r.success / r.episodes * 100:6.2f}%  ({r.success}/{r.episodes})")
    print(f"Collision rate: {r.collision / r.episodes * 100:6.2f}%  ({r.collision}/{r.episodes})")
    print(f"Timeout rate  : {r.timeout / r.episodes * 100:6.2f}%  ({r.timeout}/{r.episodes})")
    print(f"Avg steps     : {r.avg_steps:7.2f}")
    print(
        "Decision time(ms/step): "
        f"mean={r.decision_time_ms_mean:.3f}, p50={r.decision_time_ms_p50:.3f}, "
        f"p90={r.decision_time_ms_p90:.3f}, p95={r.decision_time_ms_p95:.3f}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate LSTM-DDPG(+Attention) in two scenarios")

    p.add_argument("--model", type=str, required=True, help="Path to .pth checkpoint")
    p.add_argument("--episodes", type=int, default=200, help="Episodes per scenario")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=None, help="Optional json output path")

    # Scenario2: 额外障碍物数量
    p.add_argument("--add_static", type=int, default=2)
    p.add_argument("--add_dynamic", type=int, default=2)

    # 允许覆盖环境/状态配置（默认从 checkpoint state_meta 读取）
    p.add_argument("--legacy_state", action="store_true")
    p.add_argument("--n_sectors", type=int, choices=[8, 16], default=None)
    p.add_argument("--sector_method", type=str, choices=["min", "mean"], default=None)
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    p.add_argument("--dynamic_speed_min", type=float, default=None)
    p.add_argument("--dynamic_speed_max", type=float, default=None)
    p.add_argument("--dynamic_patterns", type=str, default=None)
    p.add_argument("--dynamic_stop_prob", type=float, default=None)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    ckpt = _safe_torch_load(args.model, map_location="cpu")
    state_meta = ckpt.get("state_meta", {}) or {}

    # 场景1：训练场景
    env1, resolved = _build_env_from_meta(state_meta, args)

    # 构造 agent 并 load
    agent = _construct_agent_from_checkpoint(ckpt, env_state_dim=env1.state_dim, env_action_dim=env1.action_dim)
    # evaluation does not need optimizers; strict=False for backward compatibility
    agent.load(args.model, strict=False, load_optimizers=False)

    # 场景2：在场景1基础上增加障碍物数量
    old_static = int(getattr(EnvConfig, "NUM_STATIC_OBSTACLES", 0))
    old_dynamic = int(getattr(EnvConfig, "NUM_DYNAMIC_OBSTACLES", 0))

    try:
        EnvConfig.NUM_STATIC_OBSTACLES = old_static + int(args.add_static)
        EnvConfig.NUM_DYNAMIC_OBSTACLES = old_dynamic + int(args.add_dynamic)
        env2, _resolved2 = _build_env_from_meta(state_meta, args)
    finally:
        # 还原，避免影响其他脚本
        EnvConfig.NUM_STATIC_OBSTACLES = old_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = old_dynamic

    print("=" * 70)
    print("LSTM-DDPG(+Attention) Two-Scenario Evaluation")
    print("=" * 70)
    print("Model:", args.model)
    print("Episodes per scenario:", args.episodes)
    print("Seed:", args.seed)
    print("\nResolved env config (scenario1):")
    for k in sorted(resolved.keys()):
        print(f"  {k}: {resolved[k]}")
    print(f"  static_obstacles: {old_static}")
    print(f"  dynamic_obstacles: {old_dynamic}")
    print("\nScenario2 obstacle override:")
    print(f"  static_obstacles: {old_static} + {args.add_static} = {old_static + args.add_static}")
    print(f"  dynamic_obstacles: {old_dynamic} + {args.add_dynamic} = {old_dynamic + args.add_dynamic}")

    torch.set_grad_enabled(False)

    r1 = _eval_env(agent, env1, episodes=args.episodes, seed=args.seed, add_noise=False)

    # 场景2要临时 patch obstacle counts（因为 env.reset() 会读 EnvConfig 常量）
    try:
        EnvConfig.NUM_STATIC_OBSTACLES = old_static + int(args.add_static)
        EnvConfig.NUM_DYNAMIC_OBSTACLES = old_dynamic + int(args.add_dynamic)
        r2 = _eval_env(agent, env2, episodes=args.episodes, seed=args.seed + 100000, add_noise=False)
    finally:
        EnvConfig.NUM_STATIC_OBSTACLES = old_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = old_dynamic

    _print_result("Scenario 1: 训练场景 (base)", r1)
    _print_result("Scenario 2: base +2静态 +2动态", r2)

    if args.out:
        out_obj = {
            "model": args.model,
            "episodes": args.episodes,
            "seed": args.seed,
            "scenario1": r1.to_dict(),
            "scenario2": r2.to_dict(),
            "resolved_env_s1": resolved,
            "scenario2_obstacles": {
                "static": old_static + int(args.add_static),
                "dynamic": old_dynamic + int(args.add_dynamic),
            },
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2)
        print("\nSaved json:", args.out)

    env1.close()
    env2.close()


if __name__ == "__main__":
    main()
