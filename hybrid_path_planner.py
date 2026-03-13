#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MIDBO + LSTM-DDPG-Attention 混合航迹规划器。

设计要点（依据《混合航迹规划系统设计要求》）：
1) 全局层：使用 MIDBO 生成参考航迹，并密化到最大间距 <= 0.3m。
2) 局部层：基于前视距离从全局航迹中选局部目标点，结合安全筛选与跳点兜底。
3) 触发器：LiDAR 前向最小距离 + 短程路径安全性双判据。
4) 模式切换：GLOBAL/LOCAL 迟滞门控，避免抖动。
5) 再规划：偏离过大/局部反复触发/推进效率低时调用 MIDBO 重规划。

该实现与项目现有模块对接：
- 全局规划：midbo_path_planner.py
- 局部策略：lstm_ddpg_att.py（可选注入，不注入时退化为几何避障动作）
- 环境：environment.py
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Protocol

import numpy as np

from config import EnvConfig
from environment import NavigationEnv, Robot
from midbo_path_planner import TrajectoryEnvironment, plan_path_with_midbo


class LocalPolicy(Protocol):
    """局部策略接口（兼容 LSTM-DDPG-Attention agent）。"""

    history_len: int

    def act(self, state_seq: np.ndarray, add_noise: bool = False) -> np.ndarray:
        ...


@dataclass
class HybridConfig:
    """混合规划配置。"""

    # 全局 MIDBO
    midbo_population: int = 30
    midbo_iterations: int = 120
    waypoint_count: int = 10
    fast_mode: bool = True
    fast_population_cap: int = 12
    fast_iteration_cap: int = 40
    replan_population_scale: float = 0.6
    replan_iteration_scale: float = 0.5
    max_replans_per_episode: int = 2
    global_corridor_width: float = 10.0
    global_min_progress_step: float = 0.20
    global_smooth_iterations: int = 4
    global_smooth_alpha: float = 0.5

    # 航迹密化
    densify_spacing: float = 0.30

    # 局部目标点选择
    lookahead_distance: float = 1.2
    lookahead_max: float = 2.5
    lookahead_expand_steps: tuple[float, ...] = (1.0, 1.3, 1.6)
    max_ahead_idx: int = 250
    max_skip_points: int = 50

    # 触发器
    lidar_trigger_distance: float = 1.0
    lidar_goal_sector_half_angle_deg: float = 60.0
    short_horizon_sec: float = 1.0
    short_path_sample_step: float = 0.25

    # 安全半径
    robot_radius: float = EnvConfig.ROBOT_RADIUS
    safety_margin: float = 0.15

    # 迟滞门控
    exit_clear_distance: float = 1.5
    exit_safe_steps: int = 10

    # 再规划
    deviation_threshold: float = 0.8
    deviation_steps_trigger: int = 12
    max_local_stuck_steps: int = 25
    replan_cooldown_steps: int = 80

    # 执行
    max_episode_steps: int = 2000


@dataclass
class PlannerStats:
    total_steps: int = 0
    global_steps: int = 0
    local_steps: int = 0
    trigger_count: int = 0
    replan_count: int = 0
    global_plan_time_sec: float = 0.0
    replan_time_sec: float = 0.0


@dataclass
class EpisodeTrace:
    """单回合关键调试信息。"""

    min_lidar_front: float = float("inf")
    max_deviation: float = 0.0
    local_trigger_steps: list[int] | None = None

    def __post_init__(self) -> None:
        if self.local_trigger_steps is None:
            self.local_trigger_steps = []


class HybridTrajectoryPlanner:
    """MIDBO 全局 + LSTM-DDPG-Attention 局部避障混合规划器。"""

    def __init__(
        self,
        env: NavigationEnv,
        local_policy: Optional[LocalPolicy] = None,
        config: Optional[HybridConfig] = None,
        random_state: Optional[int] = None,
    ) -> None:
        self.env = env
        self.local_policy = local_policy
        self.cfg = config or HybridConfig()
        self.rng = np.random.default_rng(random_state)

        self.global_path: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.current_goal_idx: int = 0
        self.last_goal_idx: int = 0

        self.mode: str = "GLOBAL"
        self._exit_safe_counter = 0
        self._deviation_counter = 0
        self._stuck_counter = 0
        self._last_replan_step = -10**9

        self._state_queue: Optional[Deque[np.ndarray]] = None
        self.stats = PlannerStats()
        self.trace = EpisodeTrace()

    # ------------------------ 全局层 ------------------------

    def _circle_to_aabb(self, x: float, y: float, radius: float) -> np.ndarray:
        return np.array([x - radius, y - radius, 0.0, 2 * radius, 2 * radius, 5.0], dtype=np.float64)

    def _build_global_env(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> TrajectoryEnvironment:
        static_boxes = []

        line_vec = goal_xy - start_xy
        line_len2 = float(np.dot(line_vec, line_vec))

        def _dist_to_segment(px: float, py: float) -> float:
            point = np.array([px, py], dtype=np.float64)
            if line_len2 < 1e-9:
                return float(np.linalg.norm(point - start_xy))
            t = float(np.dot(point - start_xy, line_vec) / line_len2)
            t = float(np.clip(t, 0.0, 1.0))
            proj = start_xy + t * line_vec
            return float(np.linalg.norm(point - proj))

        for obs in self.env.obstacles:
            if not obs.is_dynamic:
                d = _dist_to_segment(obs.x, obs.y)
                if d <= self.cfg.global_corridor_width:
                    static_boxes.append(self._circle_to_aabb(obs.x, obs.y, obs.radius))

        if not static_boxes:
            for obs in self.env.obstacles:
                if not obs.is_dynamic:
                    static_boxes.append(self._circle_to_aabb(obs.x, obs.y, obs.radius))

        obstacles = np.asarray(static_boxes, dtype=np.float64) if static_boxes else np.zeros((0, 6), dtype=np.float64)

        return TrajectoryEnvironment(
            start_pos=(float(start_xy[0]), float(start_xy[1]), 1.0),
            goal_pos=(float(goal_xy[0]), float(goal_xy[1]), 1.0),
            map_range=(float(EnvConfig.MAP_WIDTH), float(EnvConfig.MAP_HEIGHT), 3.0),
            obstacles=obstacles,
            waypoint_count=self.cfg.waypoint_count,
            sample_count=max(80, self.cfg.waypoint_count * 12),
            min_height=0.5,
            max_height=2.5,
            weights=(3.0, 5.0, 0.0, 1.0),
            interpolation="pchip",
        )

    def _static_obstacle_circles(self) -> list[tuple[float, float, float]]:
        circles: list[tuple[float, float, float]] = []
        for obs in self.env.obstacles:
            if not obs.is_dynamic:
                circles.append((float(obs.x), float(obs.y), float(obs.radius)))
        return circles

    def _remove_backtracking(
        self,
        path_xy: np.ndarray,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
    ) -> np.ndarray:
        """移除明显回退段，抑制全局航迹中的大幅折返。"""
        if len(path_xy) <= 2:
            return path_xy

        axis = goal_xy - start_xy
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-9:
            return path_xy
        axis = axis / axis_norm

        kept = [path_xy[0]]
        last_proj = float(np.dot(path_xy[0] - start_xy, axis))
        min_step = float(self.cfg.global_min_progress_step)

        for p in path_xy[1:]:
            proj = float(np.dot(p - start_xy, axis))
            if proj + 1e-6 < last_proj:
                continue
            if proj - last_proj < min_step and len(kept) > 1:
                continue
            kept.append(p)
            last_proj = proj

        if np.linalg.norm(kept[-1] - goal_xy) > 1e-6:
            kept.append(goal_xy)
        return np.asarray(kept, dtype=np.float32)

    def _smooth_path(
        self,
        path_xy: np.ndarray,
        static_obstacles: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """轻量平滑：拉普拉斯迭代，并对障碍碰撞点回退。"""
        if len(path_xy) <= 2:
            return path_xy

        out = path_xy.astype(np.float32).copy()
        alpha = float(np.clip(self.cfg.global_smooth_alpha, 0.0, 1.0))
        safe_margin = float(self.cfg.robot_radius + self.cfg.safety_margin)

        for _ in range(max(0, int(self.cfg.global_smooth_iterations))):
            prev = out.copy()
            for i in range(1, len(out) - 1):
                lap = 0.5 * (prev[i - 1] + prev[i + 1])
                candidate = (1.0 - alpha) * prev[i] + alpha * lap

                safe = True
                for ox, oy, rr in static_obstacles:
                    if math.hypot(float(candidate[0]) - ox, float(candidate[1]) - oy) <= rr + safe_margin:
                        safe = False
                        break
                out[i] = candidate if safe else prev[i]
        return out

    def _postprocess_global_path(
        self,
        path_xy: np.ndarray,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
    ) -> np.ndarray:
        filtered = self._remove_backtracking(path_xy, start_xy, goal_xy)
        smoothed = self._smooth_path(filtered, self._static_obstacle_circles())
        return smoothed

    @staticmethod
    def _densify_path(path_xy: np.ndarray, max_spacing: float) -> np.ndarray:
        if len(path_xy) <= 1:
            return path_xy
        dense = [path_xy[0]]
        for i in range(len(path_xy) - 1):
            p0 = path_xy[i]
            p1 = path_xy[i + 1]
            seg = p1 - p0
            dist = float(np.linalg.norm(seg))
            n = max(1, int(math.ceil(dist / max_spacing)))
            for k in range(1, n + 1):
                dense.append(p0 + seg * (k / n))
        return np.asarray(dense, dtype=np.float32)

    def _resolve_midbo_budget(self, for_replan: bool) -> tuple[int, int]:
        pop = int(self.cfg.midbo_population)
        iters = int(self.cfg.midbo_iterations)

        if self.cfg.fast_mode:
            pop = min(pop, int(self.cfg.fast_population_cap))
            iters = min(iters, int(self.cfg.fast_iteration_cap))

        if for_replan:
            pop = max(6, int(round(pop * self.cfg.replan_population_scale)))
            iters = max(20, int(round(iters * self.cfg.replan_iteration_scale)))

        return pop, iters

    def plan_global_path(self, for_replan: bool = False) -> np.ndarray:
        t0 = time.perf_counter()
        start_xy = np.array([self.env.robot.x, self.env.robot.y], dtype=np.float64)
        goal_xy = np.array([self.env.goal_x, self.env.goal_y], dtype=np.float64)
        population, iterations = self._resolve_midbo_budget(for_replan)

        g_env = self._build_global_env(start_xy, goal_xy)
        _, best_pos, _ = plan_path_with_midbo(
            g_env,
            population=population,
            iterations=iterations,
            random_state=int(self.rng.integers(1, 10**9)),
        )
        coarse = g_env.sample_path(best_pos)[:, :2]
        clean = self._postprocess_global_path(coarse, start_xy, goal_xy)
        self.global_path = self._densify_path(clean, self.cfg.densify_spacing)

        dists = np.linalg.norm(self.global_path - start_xy[None, :], axis=1)
        self.current_goal_idx = int(np.argmin(dists))
        self.last_goal_idx = self.current_goal_idx
        self.stats.global_plan_time_sec += time.perf_counter() - t0
        return self.global_path

    def _get_current_lidar_ranges(self, state: np.ndarray) -> np.ndarray:
        """获取当前LiDAR距离（米）。

        优先使用环境 LiDAR 的最近一次真实扫描，避免 Enhanced state 下索引错位。
        """
        last = getattr(self.env.lidar, "last_readings", None)
        if last is not None:
            arr = np.asarray(last, dtype=np.float32)
            if arr.size == EnvConfig.LIDAR_RAYS:
                return np.clip(arr, EnvConfig.LIDAR_MIN_RANGE, EnvConfig.LIDAR_MAX_RANGE)

        # 兜底：若无法读取 last_readings，再尝试从状态前32维恢复
        raw = np.asarray(state[:EnvConfig.LIDAR_RAYS], dtype=np.float32)
        if np.max(raw) <= 1.5 and np.min(raw) >= -0.5:
            raw = raw * float(EnvConfig.LIDAR_MAX_RANGE)
        return np.clip(raw, EnvConfig.LIDAR_MIN_RANGE, EnvConfig.LIDAR_MAX_RANGE)

    # ------------------------ 感知/安全 ------------------------

    def get_planner_obstacles(self) -> list[tuple[float, float, float]]:
        """返回“先验静态 + 已感知动态（LiDAR量程内）”障碍物。"""
        obs = []
        rx, ry = self.env.robot.x, self.env.robot.y
        for o in self.env.obstacles:
            if (not o.is_dynamic) or (math.hypot(o.x - rx, o.y - ry) <= EnvConfig.LIDAR_MAX_RANGE + o.radius):
                obs.append((float(o.x), float(o.y), float(o.radius)))
        return obs

    def _point_is_safe(self, p: np.ndarray, obstacles: list[tuple[float, float, float]]) -> bool:
        inflate = self.cfg.robot_radius + self.cfg.safety_margin
        for ox, oy, rr in obstacles:
            if math.hypot(float(p[0]) - ox, float(p[1]) - oy) <= rr + inflate:
                return False
        return True

    def _segment_is_safe(self, p0: np.ndarray, p1: np.ndarray, obstacles: list[tuple[float, float, float]]) -> bool:
        length = float(np.linalg.norm(p1 - p0))
        n = max(1, int(math.ceil(length / self.cfg.short_path_sample_step)))
        for i in range(n + 1):
            t = i / n
            p = p0 + (p1 - p0) * t
            if not self._point_is_safe(p, obstacles):
                return False
        return True

    def _forward_sector_min_range(self, lidar_ranges: np.ndarray, heading_to_goal: float) -> float:
        half = math.radians(self.cfg.lidar_goal_sector_half_angle_deg)
        ray_angles = np.linspace(-math.pi, math.pi, len(lidar_ranges), endpoint=False)
        rel = np.array([Robot.normalize_angle(a - heading_to_goal) for a in ray_angles], dtype=np.float32)
        mask = np.abs(rel) <= half
        if not np.any(mask):
            return float(np.min(lidar_ranges))
        return float(np.min(lidar_ranges[mask]))

    # ------------------------ 局部目标点 ------------------------

    def select_local_goal(self, lookahead: float, obstacles: list[tuple[float, float, float]]) -> tuple[np.ndarray, int]:
        pos = np.array([self.env.robot.x, self.env.robot.y], dtype=np.float32)
        if len(self.global_path) == 0:
            return pos.copy(), 0

        near_idx = int(np.argmin(np.linalg.norm(self.global_path - pos[None, :], axis=1)))
        idx0 = max(near_idx, self.last_goal_idx)
        end = min(len(self.global_path) - 1, idx0 + self.cfg.max_ahead_idx)

        for scale in self.cfg.lookahead_expand_steps:
            r = min(self.cfg.lookahead_max, lookahead * scale)
            candidate_idx = []
            for i in range(idx0, end + 1):
                if np.linalg.norm(self.global_path[i] - pos) <= r:
                    candidate_idx.append(i)

            for i in reversed(candidate_idx):
                if self._point_is_safe(self.global_path[i], obstacles):
                    self.last_goal_idx = i
                    return self.global_path[i].copy(), i

        jump = min(len(self.global_path) - 1, idx0 + 1)
        best = jump
        for i in range(jump, min(len(self.global_path), jump + self.cfg.max_skip_points)):
            best = i
            if self._point_is_safe(self.global_path[i], obstacles):
                break
        self.last_goal_idx = best
        return self.global_path[best].copy(), best

    # ------------------------ 模式切换 ------------------------

    def _should_trigger_local(
        self,
        local_goal: np.ndarray,
        obstacles: list[tuple[float, float, float]],
        lidar: np.ndarray,
        heading: float,
    ) -> bool:
        min_front = self._forward_sector_min_range(lidar, heading)

        pos = np.array([self.env.robot.x, self.env.robot.y], dtype=np.float32)
        short_path_unsafe = not self._segment_is_safe(pos, local_goal, obstacles)
        return (min_front < self.cfg.lidar_trigger_distance) or short_path_unsafe

    def _maybe_switch_mode(
        self,
        local_goal: np.ndarray,
        obstacles: list[tuple[float, float, float]],
        lidar: np.ndarray,
        heading: float,
    ) -> None:
        trigger = self._should_trigger_local(local_goal, obstacles, lidar, heading)

        if self.mode == "GLOBAL":
            if trigger:
                self.mode = "LOCAL"
                self.stats.trigger_count += 1
                self._exit_safe_counter = 0
            return

        # LOCAL -> GLOBAL (迟滞)
        min_front = self._forward_sector_min_range(lidar, heading)
        path_safe = self._segment_is_safe(np.array([self.env.robot.x, self.env.robot.y], dtype=np.float32), local_goal, obstacles)

        if min_front >= self.cfg.exit_clear_distance and path_safe:
            self._exit_safe_counter += 1
        else:
            self._exit_safe_counter = 0

        if self._exit_safe_counter >= self.cfg.exit_safe_steps:
            self.mode = "GLOBAL"
            self._exit_safe_counter = 0

    # ------------------------ 控制器 ------------------------

    def _global_tracker_action(self, local_goal: np.ndarray) -> np.ndarray:
        rx, ry, rt = self.env.robot.x, self.env.robot.y, self.env.robot.theta
        goal_ang = math.atan2(float(local_goal[1] - ry), float(local_goal[0] - rx))
        err = Robot.normalize_angle(goal_ang - rt)
        dist = math.hypot(float(local_goal[0] - rx), float(local_goal[1] - ry))

        linear = float(np.clip(0.7 * dist, 0.08, EnvConfig.MAX_LINEAR_VEL))
        angular = float(np.clip(1.2 * err, -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL))
        if abs(err) > 1.2:
            linear *= 0.6
        return np.array([linear, angular], dtype=np.float32)

    def _fallback_local_action(self, local_goal: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        """未提供学习策略时的几何局部动作（简化兜底）。"""
        action = self._global_tracker_action(local_goal)
        if float(np.min(lidar)) < 0.8:
            action[0] = min(action[0], 0.2)
            left = float(np.mean(lidar[: len(lidar) // 2]))
            right = float(np.mean(lidar[len(lidar) // 2 :]))
            action[1] = np.clip(action[1] + (right - left) * 0.8, -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL)
        return action

    def _local_policy_action(self, state: np.ndarray, local_goal: np.ndarray, lidar: np.ndarray) -> np.ndarray:
        if self.local_policy is None:
            return self._fallback_local_action(local_goal, lidar)

        if self._state_queue is None:
            h = max(1, int(getattr(self.local_policy, "history_len", 5)))
            self._state_queue = deque([state.copy() for _ in range(h)], maxlen=h)
        else:
            self._state_queue.append(state.copy())

        seq = np.asarray(self._state_queue, dtype=np.float32)
        action = self.local_policy.act(seq, add_noise=False)
        return np.asarray(action).reshape(-1)[:2].astype(np.float32)

    # ------------------------ 再规划 ------------------------

    def _distance_to_path(self) -> float:
        if len(self.global_path) == 0:
            return 0.0
        pos = np.array([self.env.robot.x, self.env.robot.y], dtype=np.float32)
        return float(np.min(np.linalg.norm(self.global_path - pos[None, :], axis=1)))

    def _should_replan(self, local_goal_idx: int, step: int) -> bool:
        deviation = self._distance_to_path()
        if deviation > self.cfg.deviation_threshold:
            self._deviation_counter += 1
        else:
            self._deviation_counter = max(0, self._deviation_counter - 1)

        progressed = local_goal_idx > self.current_goal_idx
        self.current_goal_idx = max(self.current_goal_idx, local_goal_idx)
        if not progressed and self.mode == "LOCAL":
            self._stuck_counter += 1
        else:
            self._stuck_counter = max(0, self._stuck_counter - 1)

        if step - self._last_replan_step < self.cfg.replan_cooldown_steps:
            return False
        if self._deviation_counter >= self.cfg.deviation_steps_trigger:
            return True
        if self._stuck_counter >= self.cfg.max_local_stuck_steps:
            return True
        return False

    # ------------------------ 主流程 ------------------------

    def run_episode(self, reset: bool = True) -> dict:
        self.trace = EpisodeTrace()
        self.stats = PlannerStats()
        self._last_replan_step = -10**9
        self._deviation_counter = 0
        self._stuck_counter = 0
        self.mode = "GLOBAL"
        state = self.env.reset() if reset else self.env._get_state()
        if len(self.global_path) == 0:
            self.plan_global_path(for_replan=False)

        done = False
        total_reward = 0.0
        trajectory = [(self.env.robot.x, self.env.robot.y)]
        reasons = ""

        # 记录动态障碍物最近40步轨迹
        dyn_hist: dict[int, deque[tuple[float, float]]] = {}
        for idx, obs in enumerate(self.env.obstacles):
            if obs.is_dynamic:
                dyn_hist[idx] = deque(maxlen=40)
                dyn_hist[idx].append((float(obs.x), float(obs.y)))

        for t in range(self.cfg.max_episode_steps):
            obstacles = self.get_planner_obstacles()
            local_goal, goal_idx = self.select_local_goal(self.cfg.lookahead_distance, obstacles)
            heading = self.env._get_heading_to_goal()
            lidar = self._get_current_lidar_ranges(state)

            prev_mode = self.mode
            self._maybe_switch_mode(local_goal, obstacles, lidar, heading)
            if prev_mode == "GLOBAL" and self.mode == "LOCAL":
                self.trace.local_trigger_steps.append(t)

            if self.mode == "LOCAL":
                action = self._local_policy_action(state, local_goal, lidar)
                self.stats.local_steps += 1
            else:
                action = self._global_tracker_action(local_goal)
                self.stats.global_steps += 1

            next_state, reward, done, info = self.env.step(action)
            state = next_state
            total_reward += reward
            trajectory.append((self.env.robot.x, self.env.robot.y))

            for idx, obs in enumerate(self.env.obstacles):
                if obs.is_dynamic and idx in dyn_hist:
                    dyn_hist[idx].append((float(obs.x), float(obs.y)))

            lidar = self._get_current_lidar_ranges(state)
            heading = self.env._get_heading_to_goal()
            min_front = self._forward_sector_min_range(lidar, heading)
            self.trace.min_lidar_front = min(self.trace.min_lidar_front, float(min_front))

            deviation = self._distance_to_path()
            self.trace.max_deviation = max(self.trace.max_deviation, deviation)

            self.stats.total_steps += 1
            if self._should_replan(goal_idx, t) and self.stats.replan_count < self.cfg.max_replans_per_episode:
                t_replan = time.perf_counter()
                self.plan_global_path(for_replan=True)
                self.stats.replan_count += 1
                self.stats.replan_time_sec += time.perf_counter() - t_replan
                self._last_replan_step = t

            if done:
                reasons = str(info.get("reason", ""))
                break

        if not done and not reasons:
            reasons = "timeout_max_steps"

        return {
            "success": bool(self.env.episode_success),
            "failure": bool(self.env.episode_failure),
            "mode": self.mode,
            "reason": reasons,
            "reward": float(total_reward),
            "steps": len(trajectory) - 1,
            "trajectory": np.asarray(trajectory, dtype=np.float32),
            "stats": self.stats,
            "trace": {
                "min_lidar_front": float(self.trace.min_lidar_front),
                "max_deviation": float(self.trace.max_deviation),
                "local_trigger_steps": list(self.trace.local_trigger_steps),
            },
            "dynamic_obstacle_traces": {
                idx: np.asarray(list(path), dtype=np.float32) for idx, path in dyn_hist.items()
            },
        }


__all__ = [
    "HybridConfig",
    "HybridTrajectoryPlanner",
    "PlannerStats",
    "EpisodeTrace",
    "LocalPolicy",
]
