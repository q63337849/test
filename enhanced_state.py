#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ddpg_navigation.enhanced_state

EnhancedSim2RealStateV2

面向 2D LiDAR + 里程计的“Sim2Real 友好”状态构建器：
- 不包含绝对位置 (x,y)
- 提供目标相对信息 (d, cos(psi), sin(psi))
- LiDAR 扇区压缩 (min/mean)
- (可选) ΔLiDAR：帧间差分，附带“无回波”异常抑制
- (可选) Δyaw：归一化后的航向变化提示，辅助区分“自转导致的 LiDAR 变化”

输出维度（默认对齐你给的 V2 结构）：
  state = [lidar_sectors(n), lidar_diff(n), target(3), vel(2), attitude(2), yaw_rate(1), delta_yaw(1), prev_act(2)]
其中 attitude(2) 对于本 2D 移动机器人环境固定为 0（保留维度便于与 UAV/Sim2Real 版本兼容）。

如果你想更凸显“LSTM 依赖记忆”的优势，可将 use_lidar_diff=False，让 LSTM 从多帧 LiDAR 自行推断速度。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class EnhancedStateConfig:
    lidar_rays: int = 32
    n_sectors: int = 16
    max_range: float = 3.5
    min_range: float = 0.12
    dt: float = 0.1
    yaw_rate_max: float = 2.0
    # max_speed：机器人自身最大速度（用于归一化 vx/vy 与 prev_action）
    max_speed: float = 0.22

    # dynamic_obs_max_speed：动态障碍的最大速度（用于估计“相对速度上限”）
    # - 若 relative_speed_max 未提供，则默认采用 max_speed + dynamic_obs_max_speed
    dynamic_obs_max_speed: float = 0.15

    # relative_speed_max：最坏情况下的相对速度上限（建议：v_robot + v_obs）
    # - 设为 None 则按 max_speed + dynamic_obs_max_speed 自动计算
    relative_speed_max: float | None = None

    # diff_scale 是否使用相对速度上限来估计“一步最大位移”
    # - True：更符合动态障碍环境；False：退化为仅用 robot max_speed
    use_relative_speed_for_diff_scale: bool = True
    map_width: float = 10.0
    map_height: float = 10.0

    sector_method: str = "min"  # 'min' or 'mean'
    use_lidar_diff: bool = True
    use_delta_yaw: bool = True

    # 无回波判定阈值（归一化后接近 1.0 视为无回波）
    no_echo_threshold: float = 0.95

    # ΔLiDAR 缩放：将“一步最大位移”映射到约 0.5 的差分幅值（再裁剪到 [-1,1]）
    diff_target_amp: float = 0.8


class EnhancedSim2RealStateV2:
    """增强状态设计（修正版）。"""

    def __init__(self, config: Dict):
        cfg = EnhancedStateConfig(**config)
        self.cfg = cfg

        if cfg.n_sectors <= 0:
            raise ValueError("n_sectors must be positive")
        if cfg.lidar_rays % cfg.n_sectors != 0:
            raise ValueError("lidar_rays must be divisible by n_sectors")

        self.n_sectors = int(cfg.n_sectors)
        self.lidar_rays = int(cfg.lidar_rays)
        self.rays_per_sector = self.lidar_rays // self.n_sectors

        # diff_scale：基于“归一化后的一步最大位移”计算
        # 关键点：用于 ΔLiDAR 的最大变化应由“相对速度上限”决定，而非仅由机器人自身速度决定
        # - 最坏情况（相向运动）：v_rel ~= v_robot + v_obs
        if bool(cfg.use_relative_speed_for_diff_scale):
            rel_v = cfg.relative_speed_max
            if rel_v is None:
                rel_v = float(cfg.max_speed) + float(cfg.dynamic_obs_max_speed)
            rel_v = float(rel_v)
        else:
            rel_v = float(cfg.max_speed)

        # 归一化的一步最大位移： (v_rel * dt) / max_range
        max_step_disp_norm = (rel_v * float(cfg.dt)) / max(float(cfg.max_range), 1e-6)
        target_amp = max(float(cfg.diff_target_amp), 1e-6)
        self.diff_scale = max_step_disp_norm / target_amp
        # 保护：过小会导致 scaled_diff 爆炸；过大又会抹平信号
        self.diff_scale = float(np.clip(self.diff_scale, 1e-4, 0.2))

        # 历史缓存
        self.prev_lidar_sectors = None
        self.prev_valid_mask = None
        self.prev_yaw = None

        # 维度：保持与提案一致（attitude 2 维保留但对机器人恒为 0）
        self.state_dim = (
            self.n_sectors +  # LiDAR
            (self.n_sectors if cfg.use_lidar_diff else 0) +
            3 +               # 目标 (d, cos, sin)
            2 +               # 速度 (vx, vy)
            2 +               # 姿态 (roll, pitch) - 本环境恒 0
            1 +               # yaw_rate
            (1 if cfg.use_delta_yaw else 0) +
            2                # prev_action
        )

    def reset(self) -> None:
        self.prev_lidar_sectors = None
        self.prev_valid_mask = None
        self.prev_yaw = None

    # ----------------------------- public API -----------------------------
    def build_state(
        self,
        robot,
        goal_xy: Tuple[float, float],
        raw_lidar_ranges: np.ndarray,
        prev_action: np.ndarray,
    ) -> np.ndarray:
        """从环境对象构建状态。

        参数约定：
        - robot: environment.Robot 实例，至少包含 x,y,theta, vx,vy, angular_vel
        - goal_xy: (goal_x, goal_y)
        - raw_lidar_ranges: shape (lidar_rays,) 的距离数组（单位：米）
        - prev_action: shape(2,) [linear_vel_cmd, angular_vel_cmd]
        """
        lidar_clean = self._clean_lidar(raw_lidar_ranges)
        lidar_sectors, valid_mask = self._sector_compress_with_validity(lidar_clean)

        parts = [lidar_sectors]

        if self.cfg.use_lidar_diff:
            lidar_diff = self._compute_lidar_diff_safe(lidar_sectors, valid_mask)
            parts.append(lidar_diff)

        target_info = self._get_target_polar(robot, goal_xy)
        velocity = self._get_body_velocity(robot)
        attitude = np.zeros(2, dtype=np.float32)  # 2D robot: roll/pitch not available
        yaw_rate = np.array([
            np.clip(float(getattr(robot, "angular_vel", 0.0)) / max(self.cfg.yaw_rate_max, 1e-6), -1.0, 1.0)
        ], dtype=np.float32)

        parts.extend([target_info, velocity, attitude, yaw_rate])

        if self.cfg.use_delta_yaw:
            delta_yaw = self._compute_delta_yaw_normalized(float(getattr(robot, "theta", 0.0)))
            parts.append(delta_yaw)

        prev_act = self._normalize_prev_action(prev_action)
        parts.append(prev_act)

        state = np.concatenate(parts).astype(np.float32)
        return state

    # ----------------------------- components -----------------------------
    def _clean_lidar(self, raw_lidar: np.ndarray) -> np.ndarray:
        """LiDAR 数据清洗并归一化到 [0,1]。"""
        lidar = np.asarray(raw_lidar, dtype=np.float32).reshape(-1)
        if lidar.shape[0] != self.lidar_rays:
            raise ValueError(f"lidar shape mismatch: got {lidar.shape}, expected ({self.lidar_rays},)")

        invalid_mask = (
            np.isinf(lidar) |
            np.isnan(lidar) |
            (lidar <= 0) |
            (lidar > float(self.cfg.max_range))
        )
        lidar = np.where(invalid_mask, float(self.cfg.max_range), lidar)
        lidar = np.clip(lidar, float(self.cfg.min_range), float(self.cfg.max_range))
        lidar_norm = lidar / max(float(self.cfg.max_range), 1e-6)
        return lidar_norm.astype(np.float32)

    def _sector_compress_with_validity(self, lidar_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """扇区压缩 + 有效性掩码。

        valid_mask[i]=True 表示该扇区至少命中了“非无回波”的距离。
        """
        sectors = np.empty(self.n_sectors, dtype=np.float32)
        valid = np.empty(self.n_sectors, dtype=bool)
        method = (self.cfg.sector_method or "min").lower()

        for i in range(self.n_sectors):
            start = i * self.rays_per_sector
            end = start + self.rays_per_sector
            seg = lidar_norm[start:end]
            if method == "mean":
                v = float(np.mean(seg))
            else:
                v = float(np.min(seg))
            sectors[i] = v
            valid[i] = v < float(self.cfg.no_echo_threshold)

        return sectors, valid

    def _compute_lidar_diff_safe(self, current: np.ndarray, current_valid: np.ndarray) -> np.ndarray:
        """计算 ΔLiDAR（带异常抑制）。"""
        if self.prev_lidar_sectors is None:
            diff = np.zeros(self.n_sectors, dtype=np.float32)
            prev_valid = np.ones(self.n_sectors, dtype=bool)
        else:
            raw_diff = current - self.prev_lidar_sectors
            scaled = raw_diff / max(self.diff_scale, 1e-6)
            prev_valid = self.prev_valid_mask if self.prev_valid_mask is not None else np.ones(self.n_sectors, dtype=bool)
            both_valid = current_valid & prev_valid
            diff = np.where(both_valid, scaled, 0.0)
            diff = np.clip(diff, -1.0, 1.0).astype(np.float32)

        self.prev_lidar_sectors = current.copy()
        self.prev_valid_mask = current_valid.copy()
        return diff

    def _compute_delta_yaw_normalized(self, current_yaw: float) -> np.ndarray:
        """Δyaw 归一化（与 yaw_rate_max * dt 绑定）。"""
        if self.prev_yaw is None:
            delta = 0.0
        else:
            delta = current_yaw - self.prev_yaw
            delta = math.atan2(math.sin(delta), math.cos(delta))
            max_delta = max(float(self.cfg.yaw_rate_max) * float(self.cfg.dt), 1e-6)
            delta = float(np.clip(delta / max_delta, -1.0, 1.0))
        self.prev_yaw = float(current_yaw)
        return np.array([delta], dtype=np.float32)

    def _get_target_polar(self, robot, goal_xy: Tuple[float, float]) -> np.ndarray:
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        rx, ry = float(getattr(robot, "x", 0.0)), float(getattr(robot, "y", 0.0))
        yaw = float(getattr(robot, "theta", 0.0))

        dx = gx - rx
        dy = gy - ry
        d = math.sqrt(dx * dx + dy * dy)

        # 使用地图对角线做距离归一化，更稳健
        diag = math.sqrt(float(self.cfg.map_width) ** 2 + float(self.cfg.map_height) ** 2)
        d_norm = float(np.clip(d / max(diag, 1e-6), 0.0, 1.0))

        theta = math.atan2(dy, dx)
        psi = theta - yaw
        psi = math.atan2(math.sin(psi), math.cos(psi))
        return np.array([d_norm, math.cos(psi), math.sin(psi)], dtype=np.float32)

    def _get_body_velocity(self, robot) -> np.ndarray:
        """机体系速度（2D 机器人：vy 近似为 0；仍保留 2 维以兼容 UAV 版本接口）。"""
        # 本环境 robot.vx/vy 已由 linear_vel*cos/sin 得到（世界系）
        vx_w = float(getattr(robot, "vx", 0.0))
        vy_w = float(getattr(robot, "vy", 0.0))
        yaw = float(getattr(robot, "theta", 0.0))

        vx_b = vx_w * math.cos(yaw) + vy_w * math.sin(yaw)
        vy_b = -vx_w * math.sin(yaw) + vy_w * math.cos(yaw)

        max_v = max(float(self.cfg.max_speed), 1e-6)
        vx_n = float(np.clip(vx_b / max_v, -1.0, 1.0))
        vy_n = float(np.clip(vy_b / max_v, -1.0, 1.0))
        return np.array([vx_n, vy_n], dtype=np.float32)

    def _normalize_prev_action(self, prev_action: np.ndarray) -> np.ndarray:
        a = np.asarray(prev_action, dtype=np.float32).reshape(-1)
        if a.shape[0] != 2:
            a = np.zeros(2, dtype=np.float32)

        # 线速度命令：本环境不允许负速度（保持 baseline 行为），归一化到 [0,1]
        # 角速度命令：归一化到 [-1,1]
        v = float(a[0]) / max(float(self.cfg.max_speed), 1e-6)
        w = float(a[1]) / max(float(self.cfg.yaw_rate_max), 1e-6)
        return np.array([np.clip(v, 0.0, 1.0), np.clip(w, -1.0, 1.0)], dtype=np.float32)
