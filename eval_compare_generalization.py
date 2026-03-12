#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""\
对照评测脚本：DDPG / LSTM-DDPG / LSTM-DDPG-Attention / DDPG-Attention

目标：
- 在“复杂场景(分布外)”下评估成功率、碰撞率、超时率
- 评估实时性指标：策略前向推理时延(action latency)、环境交互循环时延(loop latency)、FPS
- 输出逐回合明细 + 分场景/分算法汇总 CSV，用于论文表格与泛化分析

使用方式（放在你的项目根目录，与 config.py / environment.py / ddpg.py 等同级）：

1) 直接用默认复杂场景集（推荐先跑通）：
   python eval_compare_generalization.py --episodes 200 \
     --ddpg_model models/ddpg_best.pth \
     --lstm_model models/lstm_ddpg_best.pth \
     --lstm_att_model models/lstm_ddpg_att_best.pth \
     --ddpg_att_model models/ddpg_att_best.pth

2) 指定设备（尽量与部署一致）：
   python eval_compare_generalization.py --device cpu ...

3) 读取自定义场景 JSON：
   python eval_compare_generalization.py --scenario_json scenarios_complex.json ...

注意：
- 四个模型的“状态构造配置”必须与训练一致（legacy_state / n_sectors / lidar_diff / delta_yaw 等）。
- LSTM 系列会自动从 checkpoint 读取 history_len 与网络配置；DDPG/DDPG-Att 若你改过 hidden_dim/heads/model_dim，
  需要通过命令行参数显式匹配（否则 load_state_dict 会报 shape mismatch）。
"""

from __future__ import annotations

import os
import sys
import json
import csv
import math
import time
import random
import argparse
from dataclasses import dataclass, field
from contextlib import contextmanager
from typing import Any, Dict, List, Tuple, Optional


# -----------------------------
# Utilities
# -----------------------------

def _now_str() -> str:
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def percentile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(values)
    if q <= 0:
        return float(xs[0])
    if q >= 100:
        return float(xs[-1])
    k = (len(xs) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs[int(k)])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)



@contextmanager
def temporary_patch(obj: Any, patch: Dict[str, Any]):
    """临时修改 config 类的属性并在退出时恢复。

    patch 的 value 支持：
      - 数值：直接 set
      - 字符串 "mul:<k>": 将原值乘以 k（常用于障碍物数量/步长等），并保持原类型（int 会 round）
      - 字符串 "add:<k>": 将原值加上 k，并保持原类型
    """
    def _apply(old_v: Any, new_v: Any) -> Any:
        # direct numeric
        if isinstance(new_v, (int, float, bool)):
            return new_v
        if isinstance(new_v, str):
            s = new_v.strip().lower()
            if s.startswith("mul:"):
                k = float(s.split(":", 1)[1])
                v = old_v * k
                if isinstance(old_v, int):
                    return int(round(v))
                return type(old_v)(v) if type(old_v) in (float,) else v
            if s.startswith("add:"):
                k = float(s.split(":", 1)[1])
                v = old_v + k
                if isinstance(old_v, int):
                    return int(round(v))
                return type(old_v)(v) if type(old_v) in (float,) else v
        return new_v

    old = {}
    for k, v in (patch or {}).items():
        if hasattr(obj, k):
            old_v = getattr(obj, k)
            old[k] = old_v
            try:
                setattr(obj, k, _apply(old_v, v))
            except Exception:
                setattr(obj, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(obj, k, v)


def infer_termination(info: Dict[str, Any], steps: int, max_steps: int) -> Tuple[bool, bool, bool, str]:
    reason = (info or {}).get("reason", "") or ""
    success = (reason == "goal_reached")
    collision = (reason in ("collision_obstacle", "collision_wall"))
    timeout = (reason == "max_steps")
    if (not success) and (not collision) and (not timeout):
        if steps >= max_steps:
            timeout = True
            reason = "max_steps"
    return success, collision, timeout, reason

def coarse_path_exists(env: Any,
                       EnvConfig: Any,
                       grid_res: float = 0.25,
                       inflate: float = 0.05,
                       static_only: bool = False) -> bool:
    """粗栅格可行性检测：检查在“静态圆形障碍+墙边界”下是否存在一条从起点到终点的连通路径。

    用途：当障碍过于密集导致所有算法 0% 成功率时，可开启该过滤，避免采样到“几何上不可达”的地图。
    说明：这是保守近似（不考虑动力学与动态障碍运动），只用于剔除明显不可行的地图样本。
    """
    try:
        import numpy as np
    except Exception:
        return True  # 无 numpy 时不做过滤

    w = float(getattr(EnvConfig, "MAP_WIDTH", 10.0))
    h = float(getattr(EnvConfig, "MAP_HEIGHT", 10.0))
    rr = float(getattr(EnvConfig, "ROBOT_RADIUS", 0.2))

    grid_res = max(0.05, float(grid_res))
    nx = int(math.ceil(w / grid_res)) + 1
    ny = int(math.ceil(h / grid_res)) + 1

    # node coordinates
    xs = np.linspace(0.0, w, nx, dtype=np.float32)
    ys = np.linspace(0.0, h, ny, dtype=np.float32)

    blocked = np.zeros((ny, nx), dtype=np.bool_)

    # wall clearance
    wall_clear = rr + float(inflate)
    blocked |= (xs[None, :] < wall_clear) | (xs[None, :] > (w - wall_clear)) | (ys[:, None] < wall_clear) | (ys[:, None] > (h - wall_clear))

    obs_list = getattr(env, "obstacles", []) or []
    for obs in obs_list:
        if static_only and bool(getattr(obs, "is_dynamic", False)):
            continue
        ox = float(getattr(obs, "x", 0.0))
        oy = float(getattr(obs, "y", 0.0))
        rad = float(getattr(obs, "radius", 0.0)) + rr + float(inflate)
        if rad <= 0:
            continue
        # vectorized circle mask
        dx = xs - ox
        dy = ys - oy
        # (ny, nx) via broadcasting
        blocked |= (dy[:, None] * dy[:, None] + dx[None, :] * dx[None, :]) <= (rad * rad)

    sx, sy = float(getattr(env.robot, "x", 0.0)), float(getattr(env.robot, "y", 0.0))
    gx, gy = float(getattr(env, "goal_x", 0.0)), float(getattr(env, "goal_y", 0.0))

    def _to_idx(x: float, y: float) -> tuple[int, int]:
        ix = int(round(x / grid_res))
        iy = int(round(y / grid_res))
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))
        return iy, ix

    s = _to_idx(sx, sy)
    g = _to_idx(gx, gy)

    if blocked[s] or blocked[g]:
        return False

    # BFS (8-neighbor)
    from collections import deque
    q = deque([s])
    visited = np.zeros_like(blocked)
    visited[s] = True
    neigh = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while q:
        y, x = q.popleft()
        if (y, x) == g:
            return True
        for dy, dx in neigh:
            yy, xx = y + dy, x + dx
            if 0 <= yy < ny and 0 <= xx < nx and (not visited[yy, xx]) and (not blocked[yy, xx]):
                visited[yy, xx] = True
                q.append((yy, xx))
    return False


# -----------------------------
# Scenario definition
# -----------------------------

@dataclass
class Scenario:
    name: str
    desc: str
    env_kwargs: Dict[str, Any]
    envconfig_patch: Dict[str, Any]
    wrappers: List[Dict[str, Any]] = field(default_factory=list)

def default_complex_scenarios() -> List[Scenario]:
    """默认“复杂/泛化”场景集。

    这些场景不会假设你训练时的静态/动态障碍数量；
    - 若 EnvConfig 中存在 NUM_STATIC_OBSTACLES / NUM_DYNAMIC_OBSTACLES / LIDAR_NOISE_STD 等字段，会被 patch；
    - 若不存在，会自动忽略 patch（但仍会改变 dynamic_speed/patterns）。
    """
    return [
        Scenario(
            name="in_domain_like",
            desc="与训练分布接近：中等速度动态障碍(bounce/random_walk)；不加噪声。",
            env_kwargs=dict(
                dynamic_speed_min=0.30,
                dynamic_speed_max=0.70,
                dynamic_patterns=("bounce", "random_walk"),
                dynamic_stop_prob=0.05,
            ),
            envconfig_patch={},
        ),
        Scenario(
            name="dense_static_dynamic",
            desc="密集障碍：静态/动态数量增加；速度保持中等。",
            env_kwargs=dict(
                dynamic_speed_min=0.30,
                dynamic_speed_max=0.70,
                dynamic_patterns=("bounce", "random_walk"),
                dynamic_stop_prob=0.05,
            ),
            envconfig_patch={
                # 若你的 config.py 中存在这些字段会生效
                "NUM_STATIC_OBSTACLES": "mul:1.4",
                "NUM_DYNAMIC_OBSTACLES": "mul:1.4",
            },
        ),
        Scenario(
            name="fast_dynamic_stopgo",
            desc="高速动态 + stop_and_go：提高速度上限并加入停走模式。",
            env_kwargs=dict(
                dynamic_speed_min=0.60,
                dynamic_speed_max=1.20,
                dynamic_patterns=("bounce", "random_walk", "stop_and_go"),
                dynamic_stop_prob=0.12,
            ),
            envconfig_patch={},
        ),
        Scenario(
            name="sensor_noise_dropout",
            desc="传感噪声：LiDAR 高斯噪声 + 丢点（Sim2Real 风格）。",
            env_kwargs=dict(
                dynamic_speed_min=0.30,
                dynamic_speed_max=0.70,
                dynamic_patterns=("bounce", "random_walk"),
                dynamic_stop_prob=0.05,
            ),
            envconfig_patch={
                "LIDAR_NOISE_STD": 0.03,
                "LIDAR_DROPOUT_PROB": 0.05,
            },
        ),
        Scenario(
            name="combo_dense_fast_noise",
            desc="组合泛化：密集 + 高速 + 噪声 + stop_and_go。",
            env_kwargs=dict(
                dynamic_speed_min=0.60,
                dynamic_speed_max=1.20,
                dynamic_patterns=("bounce", "random_walk", "stop_and_go"),
                dynamic_stop_prob=0.12,
            ),
            envconfig_patch={
                "NUM_STATIC_OBSTACLES": "mul:1.4",
                "NUM_DYNAMIC_OBSTACLES": "mul:1.4",
                "LIDAR_NOISE_STD": 0.03,
                "LIDAR_DROPOUT_PROB": 0.05,
            },
        ),
Scenario(
    name="dense_extreme_stress",
    desc="极限压力测试：密度更高（不建议纳入“泛化成功率”主表，可用“平均生存步数/碰撞步数”作为补充指标）。",
    env_kwargs=dict(
        dynamic_speed_min=0.30,
        dynamic_speed_max=0.70,
        dynamic_patterns=("bounce", "random_walk"),
        dynamic_stop_prob=0.05,
    ),
    envconfig_patch={
        "NUM_STATIC_OBSTACLES": "mul:1.8",
        "NUM_DYNAMIC_OBSTACLES": "mul:1.8",
    },
),
    ]


def load_scenarios_from_json(path: str) -> List[Scenario]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    scenarios: List[Scenario] = []
    for item in data.get("scenarios", []):
        scenarios.append(
            Scenario(
                name=str(item["name"]),
                desc=str(item.get("desc", "")),
                env_kwargs=dict(item.get("env_kwargs", {})),
                envconfig_patch=dict(item.get("envconfig_patch", {})),
                wrappers=list(item.get("wrappers", [])),
            )
        )
    if not scenarios:
        raise ValueError(f"No scenarios found in JSON: {path}")
    return scenarios



# -----------------------------
# Disturbances / wrappers (for generalization)
# -----------------------------

@dataclass
class StateSlices:
    """用于在 state 向量中定位 LiDAR/ΔLiDAR 等子向量的切片。

    EnhancedSim2RealStateV2 的默认顺序：
      [lidar(n), lidar_diff(n optional), target(3), vel(2), attitude(2), yaw_rate(1), delta_yaw(1 optional), prev_act(2)]
    """
    n_sectors: int
    lidar0: int
    lidar1: int
    diff0: Optional[int]
    diff1: Optional[int]

    @staticmethod
    def from_state_cfg(state_cfg: Dict[str, Any]) -> "StateSlices":
        n = int(state_cfg.get("n_sectors", 16))
        use_diff = (not bool(state_cfg.get("disable_lidar_diff", False)))
        lidar0, lidar1 = 0, n
        if use_diff:
            diff0, diff1 = n, 2 * n
        else:
            diff0, diff1 = None, None
        return StateSlices(n_sectors=n, lidar0=lidar0, lidar1=lidar1, diff0=diff0, diff1=diff1)

    def lidar_slice(self) -> slice:
        return slice(self.lidar0, self.lidar1)

    def diff_slice(self) -> Optional[slice]:
        if self.diff0 is None:
            return None
        return slice(self.diff0, self.diff1)

    def front_sector_indices(self, EnvConfig: Any, front_fov_deg: float = 60.0) -> List[int]:
        """返回“前方”扇区索引（基于 LiDAR FOV 的中心对称截取）。"""
        fov_deg = float(getattr(EnvConfig, "LIDAR_FOV", 240.0))
        fov_deg = max(10.0, fov_deg)
        front_fov_deg = float(front_fov_deg)
        front_fov_deg = max(1.0, min(front_fov_deg, fov_deg))
        k = int(round((front_fov_deg / fov_deg) * self.n_sectors))
        k = max(1, min(self.n_sectors, k))
        c = self.n_sectors // 2
        s = max(0, c - k // 2)
        e = min(self.n_sectors, s + k)
        return list(range(s, e))


class DisturbanceBase:
    """评测期扰动（不改训练），用于构造更有针对性的分布外场景。"""

    def reset(self, init_state, rng) -> None:
        pass

    def observe(self, state, step_i: int):
        return state

    def actuate(self, action, step_i: int):
        return action


class BurstSectorDropout(DisturbanceBase):
    """连续窗口(burst)地遮蔽部分扇区（模拟遮挡/短时失明/关键扇区失真）。"""

    def __init__(self,
                 slices: StateSlices,
                 EnvConfig: Any,
                 p_start: float = 0.03,
                 dur_min: int = 4,
                 dur_max: int = 8,
                 front_fov_deg: float = 60.0,
                 sectors: Optional[List[int]] = None,
                 fill_value: float = 1.0,
                 apply_to: str = "lidar+diff",
                 also_zero_diff: bool = True):
        self.slices = slices
        self.EnvConfig = EnvConfig
        self.p_start = float(p_start)
        self.dur_min = int(dur_min)
        self.dur_max = int(dur_max)
        self.front_fov_deg = float(front_fov_deg)
        self.sectors = sectors[:] if sectors else None
        self.fill_value = float(fill_value)
        self.apply_to = (apply_to or "lidar+diff").lower()
        self.also_zero_diff = bool(also_zero_diff)

        self._remain = 0
        self._idx = []

    def reset(self, init_state, rng) -> None:
        self._remain = 0
        self._idx = self.sectors if (self.sectors is not None) else self.slices.front_sector_indices(self.EnvConfig, self.front_fov_deg)

    def observe(self, state, step_i: int):
        import numpy as np
        s = np.asarray(state, dtype=np.float32).copy()

        if self._remain <= 0:
            if (self.p_start > 0) and (self._rng.rand() < self.p_start):
                self._remain = self._rng.randint(self.dur_min, self.dur_max + 1)
        if self._remain > 0:
            # lidar sectors
            if "lidar" in self.apply_to:
                ls = self.slices.lidar_slice()
                for i in self._idx:
                    s[ls.start + i] = self.fill_value
            # diff sectors
            ds = self.slices.diff_slice()
            if (ds is not None) and (("diff" in self.apply_to) or self.also_zero_diff):
                for i in self._idx:
                    s[ds.start + i] = 0.0
            self._remain -= 1
        return s

    def reset_rng(self, rng):
        self._rng = rng



class BurstNearestOcclusion(DisturbanceBase):
    """连续窗口(burst)遮蔽“最近的K个扇区”（可选限制在前方FOV内）。
    目的：比固定前方遮挡更“针对风险”，迫使策略在关键扇区缺失时依赖历史与空间注意力重分配。
    说明：默认假设“距离越小越危险”。若你的 LiDAR 编码相反（越大越近），请把 argsort 改为取最大值。
    """

    def __init__(self,
                 slices: StateSlices,
                 EnvConfig: Any,
                 p_start: float = 0.03,
                 dur_min: int = 4,
                 dur_max: int = 8,
                 k_nearest: int = 3,
                 front_only: bool = True,
                 front_fov_deg: float = 90.0,
                 fill_value: float = 1.0,
                 apply_to: str = "lidar+diff",
                 also_zero_diff: bool = True):
        self.slices = slices
        self.EnvConfig = EnvConfig
        self.p_start = float(p_start)
        self.dur_min = int(dur_min)
        self.dur_max = int(dur_max)
        self.k_nearest = max(1, int(k_nearest))
        self.front_only = bool(front_only)
        self.front_fov_deg = float(front_fov_deg)
        self.fill_value = float(fill_value)
        self.apply_to = (apply_to or "lidar+diff").lower()
        self.also_zero_diff = bool(also_zero_diff)

        self._remain = 0
        self._pool_idx: List[int] = []

    def reset(self, init_state, rng) -> None:
        self._remain = 0
        if self.front_only:
            self._pool_idx = self.slices.front_sector_indices(self.EnvConfig, self.front_fov_deg)
        else:
            self._pool_idx = list(range(self.slices.n_sectors))

    def observe(self, state, step_i: int):
        import numpy as np
        s = np.asarray(state, dtype=np.float32).copy()

        if self._remain <= 0:
            if (self.p_start > 0) and (self._rng.rand() < self.p_start):
                self._remain = self._rng.randint(self.dur_min, self.dur_max + 1)

        if self._remain > 0:
            ls = self.slices.lidar_slice()
            lidar = s[ls]
            pool = np.array(self._pool_idx, dtype=np.int32)

            vals = lidar[pool]
            k = min(self.k_nearest, vals.shape[0])
            idx_local = np.argsort(vals)[:k]  # 小到大：最近/最危险
            idx = pool[idx_local]

            if "lidar" in self.apply_to:
                for i in idx:
                    s[ls.start + int(i)] = self.fill_value

            ds = self.slices.diff_slice()
            if (ds is not None) and (("diff" in self.apply_to) or self.also_zero_diff):
                for i in idx:
                    s[ds.start + int(i)] = 0.0

            self._remain -= 1

        return s

    def reset_rng(self, rng):
        self._rng = rng


class BurstChannelDropout(DisturbanceBase):
    """对指定通道（lidar 或 diff）做 burst 置零/置 max_range。"""

    def __init__(self,
                 slices: StateSlices,
                 channel: str = "diff",
                 p_start: float = 0.05,
                 dur_min: int = 5,
                 dur_max: int = 12,
                 fill: float = 0.0):
        self.slices = slices
        self.channel = (channel or "diff").lower()
        self.p_start = float(p_start)
        self.dur_min = int(dur_min)
        self.dur_max = int(dur_max)
        self.fill = float(fill)
        self._remain = 0

    def reset(self, init_state, rng) -> None:
        self._remain = 0

    def observe(self, state, step_i: int):
        import numpy as np
        s = np.asarray(state, dtype=np.float32).copy()

        if self._remain <= 0:
            if (self.p_start > 0) and (self._rng.rand() < self.p_start):
                self._remain = self._rng.randint(self.dur_min, self.dur_max + 1)

        if self._remain > 0:
            if self.channel == "lidar":
                sl = self.slices.lidar_slice()
                s[sl] = self.fill
            else:
                ds = self.slices.diff_slice()
                if ds is not None:
                    s[ds] = self.fill
            self._remain -= 1
        return s

    def reset_rng(self, rng):
        self._rng = rng


class ObsStutter(DisturbanceBase):
    """观测卡顿（连续若干步返回上一帧/上若干帧的观测），模拟传感延迟或通信抖动。"""

    def __init__(self, p_start: float = 0.02, dur_min: int = 3, dur_max: int = 6, mode: str = "hold_last"):
        self.p_start = float(p_start)
        self.dur_min = int(dur_min)
        self.dur_max = int(dur_max)
        self.mode = (mode or "hold_last").lower()
        self._remain = 0
        self._hold = None

    def reset(self, init_state, rng) -> None:
        self._remain = 0
        self._hold = None

    def observe(self, state, step_i: int):
        import numpy as np
        s = np.asarray(state, dtype=np.float32)
        if self._hold is None:
            self._hold = s.copy()

        if self._remain <= 0:
            if (self.p_start > 0) and (self._rng.rand() < self.p_start):
                self._remain = self._rng.randint(self.dur_min, self.dur_max + 1)
                # 锁存
                self._hold = s.copy()

        if self._remain > 0:
            self._remain -= 1
            return self._hold.copy()

        self._hold = s.copy()
        return s.copy()

    def reset_rng(self, rng):
        self._rng = rng


class ActionDelay(DisturbanceBase):
    """动作延迟：执行的是 d 步之前的动作。"""

    def __init__(self, delay_steps: int = 2, init: str = "zeros"):
        from collections import deque
        self.delay_steps = max(0, int(delay_steps))
        self.init = (init or "zeros").lower()
        self._q = deque()

    def reset(self, init_state, rng) -> None:
        self._q.clear()

    def actuate(self, action, step_i: int):
        import numpy as np
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.delay_steps <= 0:
            return a
        if not self._q:
            # init fill
            fill = np.zeros_like(a) if self.init == "zeros" else a.copy()
            for _ in range(self.delay_steps):
                self._q.append(fill.copy())
        self._q.append(a.copy())
        return self._q.popleft()


class ActionLowPass(DisturbanceBase):
    """执行端一阶惯性：a_exec = alpha*a_prev + (1-alpha)*a_cmd。"""

    def __init__(self, alpha: float = 0.8):
        self.alpha = float(alpha)
        self._prev = None

    def reset(self, init_state, rng) -> None:
        self._prev = None

    def actuate(self, action, step_i: int):
        import numpy as np
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if self._prev is None:
            self._prev = a.copy()
            return a
        alpha = float(np.clip(self.alpha, 0.0, 0.999))
        out = alpha * self._prev + (1.0 - alpha) * a
        self._prev = out.copy()
        return out


class LidarBiasDrift(DisturbanceBase):
    """LiDAR 扇区的时间相关偏置漂移（OU-like），模拟传感标定漂移/相关噪声。"""

    def __init__(self,
                 slices: StateSlices,
                 sigma: float = 0.02,
                 tau_steps: int = 30,
                 clip_min: float = 0.0,
                 clip_max: float = 1.0,
                 apply_to: str = "lidar"):
        self.slices = slices
        self.sigma = float(sigma)
        self.tau_steps = max(1, int(tau_steps))
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.apply_to = (apply_to or "lidar").lower()
        self._b = None
        self._rho = math.exp(-1.0 / float(self.tau_steps))

    def reset(self, init_state, rng) -> None:
        import numpy as np
        self._b = np.zeros(self.slices.n_sectors, dtype=np.float32)

    def observe(self, state, step_i: int):
        import numpy as np
        s = np.asarray(state, dtype=np.float32).copy()
        if self._b is None:
            self._b = np.zeros(self.slices.n_sectors, dtype=np.float32)

        # OU update
        eps = self._rng.normal(0.0, 1.0, size=self.slices.n_sectors).astype(np.float32)
        self._b = (self._rho * self._b + (1.0 - self._rho) * (self.sigma * eps)).astype(np.float32)

        if "lidar" in self.apply_to:
            sl = self.slices.lidar_slice()
            s[sl] = np.clip(s[sl] + self._b, self.clip_min, self.clip_max)
        if "diff" in self.apply_to:
            ds = self.slices.diff_slice()
            if ds is not None:
                s[ds] = np.clip(s[ds] + self._b, -1.0, 1.0)

        return s

    def reset_rng(self, rng):
        self._rng = rng


class DisturbanceChain(DisturbanceBase):
    def __init__(self, disturbances: List[DisturbanceBase], rng):
        self.ds = disturbances
        self.rng = rng
        for d in self.ds:
            if hasattr(d, "reset_rng"):
                d.reset_rng(rng)

    def reset(self, init_state, rng) -> None:
        for d in self.ds:
            d.reset(init_state, rng)

    def observe(self, state, step_i: int):
        s = state
        for d in self.ds:
            s = d.observe(s, step_i)
        return s

    def actuate(self, action, step_i: int):
        a = action
        for d in self.ds:
            a = d.actuate(a, step_i)
        return a


def build_disturbance_chain(mods: ProjectModules,
                            scenario: Scenario,
                            state_cfg: Dict[str, Any],
                            seed: int) -> DisturbanceChain:
    """从 scenario.wrappers 构建扰动链。使用独立 RNG，避免影响环境内部随机性。"""
    np = mods.np
    rng = np.random.RandomState(int(seed) ^ 0x5EED1234)
    slices = StateSlices.from_state_cfg(state_cfg)

    ds: List[DisturbanceBase] = []
    for spec in (scenario.wrappers or []):
        if not isinstance(spec, dict):
            continue
        typ = str(spec.get("type", "")).lower().strip()
        if typ in ("burst_sector_dropout", "burst_dropout_front", "burst_front_blackout"):
            ds.append(BurstSectorDropout(
                slices=slices,
                EnvConfig=mods.EnvConfig,
                p_start=float(spec.get("p_start", 0.03)),
                dur_min=int(spec.get("dur_min", 4)),
                dur_max=int(spec.get("dur_max", 8)),
                front_fov_deg=float(spec.get("front_fov_deg", 60.0)),
                sectors=spec.get("sectors", None),
                fill_value=float(spec.get("fill_value", 1.0)),
                apply_to=str(spec.get("apply_to", "lidar+diff")),
                also_zero_diff=bool(spec.get("also_zero_diff", True)),
            ))
        elif typ in ("burst_nearest_occlusion", "nearest_occlusion_burst", "burst_occlude_nearest"):
            ds.append(BurstNearestOcclusion(
                slices=slices,
                EnvConfig=mods.EnvConfig,
                p_start=float(spec.get("p_start", 0.03)),
                dur_min=int(spec.get("dur_min", 4)),
                dur_max=int(spec.get("dur_max", 8)),
                k_nearest=int(spec.get("k_nearest", 3)),
                front_only=bool(spec.get("front_only", True)),
                front_fov_deg=float(spec.get("front_fov_deg", 90.0)),
                fill_value=float(spec.get("fill_value", 1.0)),
                apply_to=str(spec.get("apply_to", "lidar+diff")),
                also_zero_diff=bool(spec.get("also_zero_diff", True)),
            ))
        elif typ in ("obs_stutter", "sensor_stutter"):
            ds.append(ObsStutter(
                p_start=float(spec.get("p_start", 0.02)),
                dur_min=int(spec.get("dur_min", 3)),
                dur_max=int(spec.get("dur_max", 6)),
                mode=str(spec.get("mode", "hold_last")),
            ))
        elif typ in ("action_delay", "actuator_delay"):
            ds.append(ActionDelay(
                delay_steps=int(spec.get("delay_steps", 2)),
                init=str(spec.get("init", "zeros")),
            ))
        elif typ in ("action_lowpass", "actuator_lowpass", "action_lag"):
            ds.append(ActionLowPass(alpha=float(spec.get("alpha", 0.8))))
        elif typ in ("burst_channel_dropout", "diff_burst_dropout"):
            ds.append(BurstChannelDropout(
                slices=slices,
                channel=str(spec.get("channel", "diff")),
                p_start=float(spec.get("p_start", 0.05)),
                dur_min=int(spec.get("dur_min", 5)),
                dur_max=int(spec.get("dur_max", 12)),
                fill=float(spec.get("fill", 0.0)),
            ))
        elif typ in ("lidar_bias_drift", "correlated_lidar_bias"):
            ds.append(LidarBiasDrift(
                slices=slices,
                sigma=float(spec.get("sigma", 0.02)),
                tau_steps=int(spec.get("tau_steps", 30)),
                clip_min=float(spec.get("clip_min", 0.0)),
                clip_max=float(spec.get("clip_max", 1.0)),
                apply_to=str(spec.get("apply_to", "lidar")),
            ))

    return DisturbanceChain(ds, rng=rng)


# -----------------------------
# Algorithm wrappers
# -----------------------------

class AlgoWrapperBase:
    name: str

    def reset(self, init_state: Any) -> None:
        raise NotImplementedError

    def act(self, state: Any, step_i: int) -> Any:
        raise NotImplementedError


class DDPGWrapper(AlgoWrapperBase):
    def __init__(self, agent):
        self.agent = agent
        self.name = "DDPG"

    def reset(self, init_state: Any) -> None:
        # stateless
        return

    def act(self, state: Any, step_i: int):
        # agent.act returns shape (1,2)
        a = self.agent.act(state, step=step_i, add_noise=False)
        return a.reshape(-1)


class DDPGAttWrapper(AlgoWrapperBase):
    def __init__(self, agent):
        self.agent = agent
        self.name = "DDPG-Attention"

    def reset(self, init_state: Any) -> None:
        return

    def act(self, state: Any, step_i: int):
        a = self.agent.act(state, step=step_i, add_noise=False)
        return a.reshape(-1)


class LSTMWrapBase(AlgoWrapperBase):
    def __init__(self, agent, history_len: int, display_name: str):
        from collections import deque
        self.agent = agent
        self.history_len = int(history_len)
        self._q = deque(maxlen=self.history_len)
        self.name = display_name

    def reset(self, init_state: Any) -> None:
        self._q.clear()
        for _ in range(self.history_len):
            self._q.append(init_state.copy())

    def act(self, state: Any, step_i: int):
        import numpy as np
        state_seq = np.stack(list(self._q), axis=0)
        a = self.agent.act(state_seq, step=step_i, add_noise=False)
        return a.reshape(-1)

    def on_step(self, next_state: Any) -> None:
        self._q.append(next_state.copy())


class LSTMDDPGWrapper(LSTMWrapBase):
    def __init__(self, agent, history_len: int):
        super().__init__(agent, history_len=history_len, display_name="LSTM-DDPG")


class LSTMDDPGAttWrapper(LSTMWrapBase):
    def __init__(self, agent, history_len: int):
        super().__init__(agent, history_len=history_len, display_name="LSTM-DDPG-Attention")


# -----------------------------
# Agent builders (lazy import)
# -----------------------------

def prepare_imports(device: str) -> None:
    """根据设备选择，在 import torch 之前尽量限制 CUDA 可见性。"""
    if device.lower() == "cpu":
        # 必须在 torch import 前设置
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


@dataclass
class ProjectModules:
    torch: Any
    np: Any
    config: Any
    EnvConfig: Any
    NavigationEnv: Any

    ddpg_mod: Any
    ddpg_att_mod: Any
    lstm_mod: Any
    lstm_att_mod: Any


def import_project_modules(lstm_att_impl: str) -> ProjectModules:
    import importlib

    # delayed imports
    torch = importlib.import_module("torch")
    np = importlib.import_module("numpy")

    config = importlib.import_module("config")
    EnvConfig = getattr(config, "EnvConfig")

    environment = importlib.import_module("environment")
    NavigationEnv = getattr(environment, "NavigationEnv")

    ddpg_mod = importlib.import_module("ddpg")
    ddpg_att_mod = importlib.import_module("ddpg_att")
    lstm_mod = importlib.import_module("lstm_ddpg")

    if lstm_att_impl.lower() == "att1":
        lstm_att_mod = importlib.import_module("lstm_ddpg_att1")
    else:
        lstm_att_mod = importlib.import_module("lstm_ddpg_att")

    return ProjectModules(
        torch=torch,
        np=np,
        config=config,
        EnvConfig=EnvConfig,
        NavigationEnv=NavigationEnv,
        ddpg_mod=ddpg_mod,
        ddpg_att_mod=ddpg_att_mod,
        lstm_mod=lstm_mod,
        lstm_att_mod=lstm_att_mod,
    )


def load_checkpoint_any(torch_mod, path: str, map_location: str):
    # PyTorch 2.6+ needs weights_only=False
    try:
        return torch_mod.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch_mod.load(path, map_location=map_location)


def build_ddpg(mods: ProjectModules, model_path: str, state_dim: int, action_dim: int,
               hidden_dim: Optional[int] = None):
    Agent = getattr(mods.ddpg_mod, "DDPGAgent")
    kwargs = dict(state_dim=state_dim, action_dim=action_dim)
    if hidden_dim is not None:
        kwargs["hidden_dim"] = int(hidden_dim)
    agent = Agent(**kwargs)

    ckpt = load_checkpoint_any(mods.torch, model_path, map_location="cpu")
    # full checkpoint or actor-only
    if isinstance(ckpt, dict) and "actor_local" in ckpt:
        agent.load(model_path, strict=True, load_optimizers=False)
    else:
        agent.load_actor(model_path)
    return agent


def build_ddpg_att(mods: ProjectModules, model_path: str, state_dim: int, action_dim: int,
                   hidden_dim: Optional[int] = None,
                   sector_embed_dim: int = 32,
                   spatial_att_heads: int = 4,
                   att_dropout: float = 0.0):
    Agent = getattr(mods.ddpg_att_mod, "DDPGAttentionAgent")
    kwargs = dict(
        state_dim=state_dim,
        action_dim=action_dim,
        sector_embed_dim=int(sector_embed_dim),
        spatial_att_heads=int(spatial_att_heads),
        att_dropout=float(att_dropout),
    )
    if hidden_dim is not None:
        kwargs["hidden_dim"] = int(hidden_dim)

    agent = Agent(**kwargs)

    ckpt = load_checkpoint_any(mods.torch, model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "actor_local" in ckpt:
        agent.load(model_path, strict=True, load_optimizers=False)
    else:
        agent.load_actor(model_path)
    return agent


def build_lstm_ddpg(mods: ProjectModules, model_path: str):
    Agent = getattr(mods.lstm_mod, "LSTMDdpgAgent")
    ckpt = load_checkpoint_any(mods.torch, model_path, map_location="cpu")
    net_cfg = ckpt.get("net_cfg", {}) if isinstance(ckpt, dict) else {}
    history_len = int(ckpt.get("history_len", net_cfg.get("history_len", 5)))

    agent = Agent(
        state_dim=int(net_cfg.get("state_dim", mods.EnvConfig.LIDAR_RAYS + 7)),
        action_dim=int(net_cfg.get("action_dim", 2)),
        hidden_dim=int(net_cfg.get("mlp_hidden_dim", getattr(mods.config, "DDPGConfig").HIDDEN_DIM)),
        history_len=history_len,
        embed_dim=int(net_cfg.get("embed_dim", getattr(mods.config, "DDPGConfig").LSTM_EMBED_DIM)),
        lstm_hidden_dim=int(net_cfg.get("lstm_hidden_dim", getattr(mods.config, "DDPGConfig").LSTM_HIDDEN_DIM)),
        max_lin_vel=float(net_cfg.get("max_lin_vel", mods.EnvConfig.MAX_LINEAR_VEL)),
        max_ang_vel=float(net_cfg.get("max_ang_vel", mods.EnvConfig.MAX_ANGULAR_VEL)),
    )
    agent.load(model_path, strict=True, load_optimizers=False)
    return agent, history_len


def build_lstm_ddpg_att(mods: ProjectModules, model_path: str):
    Agent = getattr(mods.lstm_att_mod, "LSTMDdpgAgent")
    ckpt = load_checkpoint_any(mods.torch, model_path, map_location="cpu")
    net_cfg = ckpt.get("net_cfg", {}) if isinstance(ckpt, dict) else {}
    history_len = int(ckpt.get("history_len", net_cfg.get("history_len", 5)))

    agent = Agent(
        state_dim=int(net_cfg.get("state_dim", mods.EnvConfig.LIDAR_RAYS + 7)),
        action_dim=int(net_cfg.get("action_dim", 2)),
        hidden_dim=int(net_cfg.get("mlp_hidden_dim", getattr(mods.config, "DDPGConfig").HIDDEN_DIM)),
        history_len=history_len,
        embed_dim=int(net_cfg.get("embed_dim", getattr(mods.config, "DDPGConfig").LSTM_EMBED_DIM)),
        lstm_hidden_dim=int(net_cfg.get("lstm_hidden_dim", getattr(mods.config, "DDPGConfig").LSTM_HIDDEN_DIM)),
        max_lin_vel=float(net_cfg.get("max_lin_vel", mods.EnvConfig.MAX_LINEAR_VEL)),
        max_ang_vel=float(net_cfg.get("max_ang_vel", mods.EnvConfig.MAX_ANGULAR_VEL)),
        use_spatial_att=bool(net_cfg.get("use_spatial_att", True)),
        use_temporal_att=bool(net_cfg.get("use_temporal_att", True)),
        sector_model_dim=int(net_cfg.get("sector_model_dim", 32)),
        temporal_att_dim=int(net_cfg.get("temporal_att_dim", 64)),
        spatial_att_heads=int(net_cfg.get("spatial_att_heads", 4)),
        temporal_att_heads=int(net_cfg.get("temporal_att_heads", 4)),
    )
    agent.load(model_path, strict=True, load_optimizers=False)
    return agent, history_len


# -----------------------------
# Evaluation
# -----------------------------

@dataclass
class EpisodeResult:
    algo: str
    scenario: str
    seed: int
    episode: int
    success: int
    collision: int
    timeout: int
    reason: str
    steps: int
    ep_return: float
    path_length: float
    ep_time_s: float
    action_ms_mean: float
    action_ms_p95: float
    env_step_ms_mean: float
    loop_ms_mean: float
    fps_mean: float
    map_feasible: int
    resample_tries: int


def run_one_episode(mods: ProjectModules,
                    env_kwargs_base: Dict[str, Any],
                    state_cfg: Dict[str, Any],
                    scenario: Scenario,
                    wrapper: AlgoWrapperBase,
                    seed: int,
                    episode_i: int,
                    args=None) -> EpisodeResult:
    np = mods.np

    # 每回合固定 seed，保证初始场景可复现
    set_global_seed(seed)

    # 构造 env（每回合 new 一个 env，避免状态残留；同时保证 reset 的随机性一致）
    env_kwargs = {}
    env_kwargs.update(env_kwargs_base)
    env_kwargs.update(state_cfg)
    env_kwargs.update(scenario.env_kwargs)

    with temporary_patch(mods.EnvConfig, scenario.envconfig_patch):
        env = mods.NavigationEnv(**env_kwargs)

        # reset (+ optional feasibility filtering)
        state = env.reset()
        resample_tries = 0
        map_feasible = True
        if (args is not None) and getattr(args, "require_feasible_map", False):
            map_feasible = coarse_path_exists(
                env=env,
                EnvConfig=mods.EnvConfig,
                grid_res=float(getattr(args, "feasible_grid_res", 0.25)),
                inflate=float(getattr(args, "feasible_inflate", 0.05)),
                static_only=bool(getattr(args, "feasible_static_only", False)),
            )
            max_tries = int(getattr(args, "feasible_max_tries", 25))
            while (not map_feasible) and resample_tries < max_tries:
                state = env.reset()
                resample_tries += 1
                map_feasible = coarse_path_exists(
                    env=env,
                    EnvConfig=mods.EnvConfig,
                    grid_res=float(getattr(args, "feasible_grid_res", 0.25)),
                    inflate=float(getattr(args, "feasible_inflate", 0.05)),
                    static_only=bool(getattr(args, "feasible_static_only", False)),
                )

        # disturbances (independent RNG; won't affect env randomness inside env)
        dist = build_disturbance_chain(mods=mods, scenario=scenario, state_cfg=state_cfg, seed=seed)
        dist.reset(state, dist.rng)
        obs = dist.observe(state, step_i=0)

        wrapper.reset(obs)
        state_for_agent = obs

        total_reward = 0.0
        steps = 0
        info = {}

        # path length
        last_x, last_y = float(env.robot.x), float(env.robot.y)
        path_len = 0.0

        action_ms: List[float] = []
        env_step_ms: List[float] = []
        loop_ms: List[float] = []

        t_ep0 = time.perf_counter()
        for t in range(int(mods.EnvConfig.MAX_STEPS)):
            t0 = time.perf_counter()
            action = wrapper.act(state_for_agent, step_i=t)
            t1 = time.perf_counter()

            # actuator disturbance (delay/low-pass etc.)
            action_exec = dist.actuate(action, step_i=t)
            t1b = time.perf_counter()

            next_state_raw, reward, done, info = env.step(action_exec)
            t2 = time.perf_counter()

            # metrics
            action_ms.append((t1 - t0) * 1000.0)
            env_step_ms.append((t2 - t1b) * 1000.0)
            loop_ms.append((t2 - t0) * 1000.0)

            total_reward += float(reward)
            steps += 1

            # path
            x, y = float(env.robot.x), float(env.robot.y)
            path_len += math.hypot(x - last_x, y - last_y)
            last_x, last_y = x, y

            next_obs = dist.observe(next_state_raw, step_i=t+1)
            state_for_agent = next_obs
            if isinstance(wrapper, LSTMWrapBase):
                wrapper.on_step(next_obs)

            if done:
                break

        ep_time = time.perf_counter() - t_ep0

        success, collision, timeout, reason = infer_termination(info, steps, int(mods.EnvConfig.MAX_STEPS))

        # stats
        a_mean = float(np.mean(action_ms)) if action_ms else float("nan")
        a_p95 = float(percentile(action_ms, 95)) if action_ms else float("nan")
        e_mean = float(np.mean(env_step_ms)) if env_step_ms else float("nan")
        l_mean = float(np.mean(loop_ms)) if loop_ms else float("nan")
        fps = 1000.0 / l_mean if l_mean and (not math.isnan(l_mean)) and l_mean > 1e-9 else float("nan")

        env.close()

        return EpisodeResult(
            algo=wrapper.name,
            scenario=scenario.name,
            seed=seed,
            episode=episode_i,
            success=int(success),
            collision=int(collision),
            timeout=int(timeout),
            reason=str(reason),
            steps=int(steps),
            ep_return=float(total_reward),
            path_length=float(path_len),
            ep_time_s=float(ep_time),
            action_ms_mean=a_mean,
            action_ms_p95=a_p95,
            env_step_ms_mean=e_mean,
            loop_ms_mean=l_mean,
            fps_mean=float(fps),
            map_feasible=int(bool(map_feasible)),
            resample_tries=int(resample_tries),
        )


def summarize(results: List[EpisodeResult]) -> List[Dict[str, Any]]:
    """按 (algo, scenario) 汇总。"""
    import numpy as np

    key2rows: Dict[Tuple[str, str], List[EpisodeResult]] = {}
    for r in results:
        key2rows.setdefault((r.algo, r.scenario), []).append(r)

    summary: List[Dict[str, Any]] = []
    for (algo, scenario), rows in sorted(key2rows.items()):
        n = len(rows)
        succ = sum(x.success for x in rows)
        coll = sum(x.collision for x in rows)
        tout = sum(x.timeout for x in rows)

        def _mean(field: str) -> float:
            xs = [getattr(x, field) for x in rows]
            return float(np.mean(xs)) if xs else float("nan")

        def _std(field: str) -> float:
            xs = [getattr(x, field) for x in rows]
            return float(np.std(xs)) if xs else float("nan")

        summary.append({
            "algo": algo,
            "scenario": scenario,
            "episodes": n,
            "success_rate_%": succ / n * 100.0 if n else 0.0,
            "collision_rate_%": coll / n * 100.0 if n else 0.0,
            "timeout_rate_%": tout / n * 100.0 if n else 0.0,
            "steps_mean": _mean("steps"),
            "steps_std": _std("steps"),
            "return_mean": _mean("ep_return"),
            "return_std": _std("ep_return"),
            "path_len_mean": _mean("path_length"),
            "path_len_std": _std("path_length"),
            "action_ms_mean": _mean("action_ms_mean"),
            "action_ms_p95_mean": _mean("action_ms_p95"),
            "loop_ms_mean": _mean("loop_ms_mean"),
            "fps_mean": _mean("fps_mean"),
            "feasible_rate_%": _mean("map_feasible") * 100.0,
            "resample_tries_mean": _mean("resample_tries"),
            "ep_time_s_mean": _mean("ep_time_s"),
        })

    return summary


def summarize_by_algo_overall(summary_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """由 (algo, scenario) 汇总进一步聚合到 algo 级别（跨所有场景平均）。"""
    import numpy as np

    algo2 = {}
    for r in summary_rows:
        algo2.setdefault(r["algo"], []).append(r)

    out = []
    for algo, rows in sorted(algo2.items()):
        episodes = int(sum(int(x["episodes"]) for x in rows))

        def wavg(key: str) -> float:
            num = 0.0
            den = 0.0
            for x in rows:
                w = float(x["episodes"])
                num += float(x[key]) * w
                den += w
            return float(num / den) if den > 0 else float("nan")

        out.append({
            "algo": algo,
            "episodes_total": episodes,
            "success_rate_%": wavg("success_rate_%"),
            "collision_rate_%": wavg("collision_rate_%"),
            "timeout_rate_%": wavg("timeout_rate_%"),
            "steps_mean": wavg("steps_mean"),
            "return_mean": wavg("return_mean"),
            "path_len_mean": wavg("path_len_mean"),
            "action_ms_mean": wavg("action_ms_mean"),
            "action_ms_p95_mean": wavg("action_ms_p95_mean"),
            "loop_ms_mean": wavg("loop_ms_mean"),
            "fps_mean": wavg("fps_mean"),
            "ep_time_s_mean": wavg("ep_time_s_mean"),
        })
    return out


def generalization_gaps(summary_rows: List[Dict[str, Any]],
                       in_domain_scenario: str = "in_domain_like") -> List[Dict[str, Any]]:
    """泛化差距：以 in_domain_scenario 为基线，计算其它场景 success_rate 的变化(ΔSR)。

    delta_success_rate_% < 0 表示相对基线成功率下降（泛化退化）。
    """
    base = {}
    for r in summary_rows:
        if r["scenario"] == in_domain_scenario:
            base[r["algo"]] = float(r["success_rate_%"])

    out = []
    for r in summary_rows:
        algo = r["algo"]
        sc = r["scenario"]
        sr = float(r["success_rate_%"])
        sr_base = base.get(algo, float("nan"))
        delta = sr - sr_base if (not (sr_base != sr_base)) else float("nan")  # NaN check
        out.append({
            "algo": algo,
            "scenario": sc,
            "success_rate_%": sr,
            "baseline_scenario": in_domain_scenario,
            "baseline_success_rate_%": sr_base,
            "delta_success_rate_%": delta,
        })
    return out


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_episode_csv(path: str, rows: List[EpisodeResult]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "algo", "scenario", "seed", "episode",
        "success", "collision", "timeout", "reason",
        "steps", "ep_return", "path_length", "ep_time_s",
        "action_ms_mean", "action_ms_p95",
        "env_step_ms_mean", "loop_ms_mean", "fps_mean",
        "map_feasible", "resample_tries",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in rows:
            w.writerow([
                r.algo, r.scenario, r.seed, r.episode,
                r.success, r.collision, r.timeout, r.reason,
                r.steps, f"{r.ep_return:.6f}", f"{r.path_length:.6f}", f"{r.ep_time_s:.6f}",
                f"{r.action_ms_mean:.6f}", f"{r.action_ms_p95:.6f}",
                f"{r.env_step_ms_mean:.6f}", f"{r.loop_ms_mean:.6f}", f"{r.fps_mean:.6f}",
            ])


def print_summary_table(summary_rows: List[Dict[str, Any]]) -> None:
    # 简单控制台输出（避免依赖 tabulate）
    cols = [
        "algo", "scenario", "episodes",
        "success_rate_%", "collision_rate_%", "timeout_rate_%",
        "steps_mean", "action_ms_mean", "loop_ms_mean", "fps_mean",
    ]
    # widths
    widths = {c: max(len(c), 10) for c in cols}
    for r in summary_rows:
        for c in cols:
            v = r.get(c, "")
            s = f"{v:.2f}" if isinstance(v, float) else str(v)
            widths[c] = max(widths[c], len(s))

    def _fmt_row(r: Dict[str, Any]) -> str:
        parts = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, float):
                s = f"{v:.2f}"
            else:
                s = str(v)
            parts.append(s.ljust(widths[c]))
        return " | ".join(parts)

    print("\n" + _fmt_row({c: c for c in cols}))
    print("-" * (sum(widths.values()) + 3 * (len(cols) - 1)))
    for r in summary_rows:
        print(_fmt_row(r))
    print()


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    # models
    p.add_argument("--ddpg_model", type=str, required=True, help="DDPG checkpoint path")
    p.add_argument("--lstm_model", type=str, required=True, help="LSTM-DDPG checkpoint path")
    p.add_argument("--lstm_att_model", type=str, required=True, help="LSTM-DDPG-Att checkpoint path")
    p.add_argument("--ddpg_att_model", type=str, required=True, help="DDPG-Att checkpoint path")

    # evaluation
    p.add_argument("--episodes", type=int, default=200, help="episodes per scenario per algo")
    p.add_argument("--base_seed", type=int, default=12345, help="base seed")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="policy device")

    # state config (must match training)
    p.add_argument("--legacy_state", action="store_true", help="use legacy state (no enhanced state)")
    p.add_argument("--n_sectors", type=int, default=16)
    p.add_argument("--sector_method", type=str, default="min")
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    # ddpg/ddpg_att arch overrides (only needed if你改过这些超参)
    p.add_argument("--ddpg_hidden_dim", type=int, default=None)
    p.add_argument("--ddpg_att_hidden_dim", type=int, default=None)
    p.add_argument("--ddpg_att_embed_dim", type=int, default=32)
    p.add_argument("--ddpg_att_heads", type=int, default=4)
    p.add_argument("--ddpg_att_dropout", type=float, default=0.0)

    # lstm-att implementation selection
    p.add_argument("--lstm_att_impl", type=str, default="att", choices=["att", "att1"],
                   help="import lstm_ddpg_att (att) or lstm_ddpg_att1 (att1)")

    # scenarios
    p.add_argument("--scenario_json", type=str, default=None, help="custom scenarios json")

    # optional: filter out geometrically-infeasible maps (very dense obstacles)
    p.add_argument("--require_feasible_map", action="store_true",
                   help="resample env.reset() until coarse static feasibility holds (avoid impossible dense maps)")
    p.add_argument("--feasible_grid_res", type=float, default=0.25, help="coarse grid resolution for feasibility check")
    p.add_argument("--feasible_inflate", type=float, default=0.05, help="extra inflate margin in feasibility check")
    p.add_argument("--feasible_static_only", action="store_true",
                   help="only use static obstacles in feasibility check (default uses both static+dynamic at t=0)")
    p.add_argument("--feasible_max_tries", type=int, default=25, help="max resample tries per episode")

    # output
    p.add_argument("--out_dir", type=str, default=None, help="output directory")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # safety: check paths
    for k in ("ddpg_model", "lstm_model", "lstm_att_model", "ddpg_att_model"):
        path = getattr(args, k)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{k} not found: {path}")

    prepare_imports(args.device)

    mods = import_project_modules(args.lstm_att_impl)

    # runtime device sanity
    if args.device == "cuda":
        if not mods.torch.cuda.is_available():
            print("[WARN] --device cuda requested but CUDA not available, fallback to CPU.")
    else:
        # evaluation only, avoid unnecessary nondeterminism
        try:
            mods.torch.set_num_threads(max(1, os.cpu_count() // 2))
        except Exception:
            pass

    # scenarios
    if args.scenario_json:
        scenarios = load_scenarios_from_json(args.scenario_json)
    else:
        scenarios = default_complex_scenarios()

    # base env kwargs (common)
    dyn_patterns_default = ("bounce", "random_walk")
    env_kwargs_base = dict(
        render_mode=None,
        dynamic_speed_min=0.30,
        dynamic_speed_max=0.70,
        dynamic_patterns=dyn_patterns_default,
        dynamic_stop_prob=0.05,
    )

    # state cfg shared for all algos (must match checkpoint)
    enhanced_cfg = {
        "n_sectors": int(args.n_sectors),
        "sector_method": str(args.sector_method),
        "use_lidar_diff": (not args.disable_lidar_diff),
        "use_delta_yaw": (not args.disable_delta_yaw),
    }
    state_cfg = dict(
        use_enhanced_state=(not args.legacy_state),
        enhanced_state_config=enhanced_cfg,
        disable_lidar_diff=bool(args.disable_lidar_diff),
        disable_delta_yaw=bool(args.disable_delta_yaw),
    )

    # a probe env to get dims
    probe_env = mods.NavigationEnv(**{**env_kwargs_base, **state_cfg})
    state_dim = int(probe_env.state_dim)
    action_dim = int(probe_env.action_dim)
    probe_env.close()

    # build agents
    ddpg_agent = build_ddpg(mods, args.ddpg_model, state_dim, action_dim, hidden_dim=args.ddpg_hidden_dim)
    ddpg_att_agent = build_ddpg_att(
        mods,
        args.ddpg_att_model,
        state_dim,
        action_dim,
        hidden_dim=args.ddpg_att_hidden_dim,
        sector_embed_dim=args.ddpg_att_embed_dim,
        spatial_att_heads=args.ddpg_att_heads,
        att_dropout=args.ddpg_att_dropout,
    )

    lstm_agent, lstm_h = build_lstm_ddpg(mods, args.lstm_model)
    lstm_att_agent, lstm_att_h = build_lstm_ddpg_att(mods, args.lstm_att_model)

    wrappers: List[AlgoWrapperBase] = [
        DDPGWrapper(ddpg_agent),
        LSTMDDPGWrapper(lstm_agent, history_len=lstm_h),
        LSTMDDPGAttWrapper(lstm_att_agent, history_len=lstm_att_h),
        DDPGAttWrapper(ddpg_att_agent),
    ]

    # output
    out_dir = args.out_dir or os.path.join("results", f"eval_generalization_{_now_str()}")
    os.makedirs(out_dir, exist_ok=True)

    # save meta
    meta = {
        "created": _now_str(),
        "episodes_per_scenario": int(args.episodes),
        "base_seed": int(args.base_seed),
        "device": str(args.device),
        "state_cfg": state_cfg,
        "models": {
            "ddpg": args.ddpg_model,
            "lstm": args.lstm_model,
            "lstm_att": args.lstm_att_model,
            "ddpg_att": args.ddpg_att_model,
        },
        "scenarios": [
            {
                "name": s.name,
                "desc": s.desc,
                "env_kwargs": s.env_kwargs,
                "envconfig_patch": s.envconfig_patch,
            }
            for s in scenarios
        ],
        "notes": "If any EnvConfig patch key does not exist in your config.py, it will be ignored.",
    }
    with open(os.path.join(out_dir, "eval_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # run
    all_results: List[EpisodeResult] = []

    print("=" * 72)
    print("Generalization Evaluation (DDPG family)")
    print("=" * 72)
    print(f"state_dim={state_dim}, action_dim={action_dim}")
    print(f"episodes/scenario/algo={args.episodes}")
    print(f"out_dir={out_dir}")
    print("Scenarios:")
    for s in scenarios:
        print(f"  - {s.name}: {s.desc}")
    print("Algorithms:")
    for w in wrappers:
        print(f"  - {w.name}")
    print("=" * 72 + "\n")

    total_jobs = len(scenarios) * len(wrappers) * int(args.episodes)
    job_i = 0
    t_all0 = time.perf_counter()

    for s in scenarios:
        for w in wrappers:
            for ep in range(int(args.episodes)):
                job_i += 1
                seed = int(args.base_seed) + ep  # per-episode seed shared across algos

                r = run_one_episode(
                    mods=mods,
                    env_kwargs_base=env_kwargs_base,
                    state_cfg=state_cfg,
                    scenario=s,
                    wrapper=w,
                    seed=seed,
                    episode_i=ep + 1,
                    args=args,
                )
                all_results.append(r)

                if (job_i % 20) == 0 or job_i == total_jobs:
                    elapsed = time.perf_counter() - t_all0
                    print(
                        f"[{job_i}/{total_jobs}] {w.name} | {s.name} | ep {ep + 1} "
                        f"-> succ={r.success} coll={r.collision} tout={r.timeout} "
                        f"steps={r.steps} a_ms={r.action_ms_mean:.3f} loop_ms={r.loop_ms_mean:.3f}"
                    )

    t_all = time.perf_counter() - t_all0
    print(f"\nDone. Total wall time: {t_all:.2f}s\n")

    # write episode csv
    ep_csv = os.path.join(out_dir, "episodes.csv")
    write_episode_csv(ep_csv, all_results)

    # summary
    summary_rows = summarize(all_results)
    sum_csv = os.path.join(out_dir, "summary_by_algo_scenario.csv")
    write_csv(
        sum_csv,
        summary_rows,
        fieldnames=list(summary_rows[0].keys()) if summary_rows else [],
    )


    # overall by algo
    overall_rows = summarize_by_algo_overall(summary_rows)
    overall_csv = os.path.join(out_dir, "summary_by_algo_overall.csv")
    if overall_rows:
        write_csv(overall_csv, overall_rows, fieldnames=list(overall_rows[0].keys()))

    # generalization gaps (ΔSR vs baseline scenario)
    gaps_rows = generalization_gaps(summary_rows, in_domain_scenario="in_domain_like")
    gaps_csv = os.path.join(out_dir, "generalization_gaps.csv")
    if gaps_rows:
        write_csv(gaps_csv, gaps_rows, fieldnames=list(gaps_rows[0].keys()))

    print_summary_table(summary_rows)

    print("Saved:")
    print(f"  - {ep_csv}")
    print(f"  - {sum_csv}")
    print(f"  - {overall_csv}")
    print(f"  - {gaps_csv}")
    print(f"  - {os.path.join(out_dir, 'eval_meta.json')}\n")


if __name__ == "__main__":
    # 确保优先从当前目录 import
    sys.path.insert(0, os.getcwd())
    main()
