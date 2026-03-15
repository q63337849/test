#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
仿真环境 - 优化版
1. 向量化 LiDAR 扫描（加速约2-3倍）
2. 可选 LiDAR 噪声（Sim2Real）
3. 实时可视化渲染
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
import time

from config import EnvConfig, RewardConfig
from enhanced_state import EnhancedSim2RealStateV2


class Robot:
    """机器人类"""

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.radius = EnvConfig.ROBOT_RADIUS

    def reset(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.vx = 0.0
        self.vy = 0.0
        self.linear_vel = 0.0
        self.angular_vel = 0.0

    def step(self, linear_vel, angular_vel, dt):
        self.linear_vel = np.clip(linear_vel, 0, EnvConfig.MAX_LINEAR_VEL)
        self.angular_vel = np.clip(angular_vel, -EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL)

        self.theta += self.angular_vel * dt
        self.theta = self.normalize_angle(self.theta)

        self.vx = self.linear_vel * math.cos(self.theta)
        self.vy = self.linear_vel * math.sin(self.theta)
        self.x += self.vx * dt
        self.y += self.vy * dt

    @staticmethod
    def normalize_angle(angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


class LiDAR:
    """向量化 LiDAR（优化版）"""

    def __init__(self):
        self.num_rays = EnvConfig.LIDAR_RAYS
        self.max_range = EnvConfig.LIDAR_MAX_RANGE
        self.min_range = EnvConfig.LIDAR_MIN_RANGE
        self.fov = math.radians(EnvConfig.LIDAR_FOV)

        # 预计算射线角度偏移
        self.ray_offsets = np.linspace(0, self.fov, self.num_rays, endpoint=False) - self.fov / 2

        # Sim2Real 噪声参数
        self.noise_std = getattr(EnvConfig, 'LIDAR_NOISE_STD', 0.0)
        self.dropout_prob = getattr(EnvConfig, 'LIDAR_DROPOUT_PROB', 0.0)

    def scan(self, robot, obstacles, walls):
        """向量化 LiDAR 扫描"""
        ranges = np.full(self.num_rays, self.max_range, dtype=np.float64)

        # 预计算射线方向（向量化）
        ray_angles = robot.theta + self.ray_offsets
        dx = np.cos(ray_angles)
        dy = np.sin(ray_angles)

        rx, ry = robot.x, robot.y

        # 检测障碍物（部分向量化）
        for obs in obstacles:
            dists = self._ray_circle_batch(rx, ry, dx, dy, obs['x'], obs['y'], obs['radius'])
            valid = dists > 0
            ranges = np.where(valid & (dists < ranges), dists, ranges)

        # 检测墙壁
        for wall in walls:
            dists = self._ray_line_batch(rx, ry, dx, dy,
                                         wall['x1'], wall['y1'], wall['x2'], wall['y2'])
            valid = dists > 0
            ranges = np.where(valid & (dists < ranges), dists, ranges)

        ranges = np.clip(ranges, self.min_range, self.max_range)

        # 添加噪声（Sim2Real）
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std * self.max_range, self.num_rays)
            ranges = ranges + noise
            ranges = np.clip(ranges, self.min_range, self.max_range)

        # 随机丢点
        if self.dropout_prob > 0:
            dropout_mask = np.random.random(self.num_rays) < self.dropout_prob
            ranges[dropout_mask] = self.max_range

        # 保存最后的读数供render使用
        self.last_readings = ranges

        return ranges.astype(np.float32)

    def _ray_circle_batch(self, rx, ry, dx, dy, cx, cy, radius):
        """批量计算射线与圆的交点距离"""
        fx = rx - cx
        fy = ry - cy

        a = dx * dx + dy * dy  # 向量化
        b = 2 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - radius * radius

        discriminant = b * b - 4 * a * c

        result = np.full_like(a, -1.0)
        valid = discriminant >= 0

        if np.any(valid):
            sqrt_disc = np.sqrt(np.maximum(discriminant, 0))
            t1 = (-b - sqrt_disc) / (2 * a)
            t2 = (-b + sqrt_disc) / (2 * a)

            # 优先取较小的正值
            use_t1 = valid & (t1 >= self.min_range)
            use_t2 = valid & ~use_t1 & (t2 >= self.min_range)

            result = np.where(use_t1, t1, result)
            result = np.where(use_t2, t2, result)

        return result

    def _ray_line_batch(self, rx, ry, dx, dy, x1, y1, x2, y2):
        """批量计算射线与线段的交点距离"""
        sx = x2 - x1
        sy = y2 - y1

        denom = dx * sy - dy * sx

        result = np.full_like(dx, -1.0)
        valid = np.abs(denom) > 1e-10

        if np.any(valid):
            t = np.where(valid, ((x1 - rx) * sy - (y1 - ry) * sx) / np.where(valid, denom, 1), -1)
            u = np.where(valid, ((x1 - rx) * dy - (y1 - ry) * dx) / np.where(valid, denom, 1), -1)

            hit = valid & (t >= self.min_range) & (u >= 0) & (u <= 1)
            result = np.where(hit, t, result)

        return result


class Obstacle:
    """障碍物类"""

    def __init__(
            self,
            x,
            y,
            radius,
            is_dynamic=False,
            is_known=False,
            vx=0.0,
            vy=0.0,
            pattern="bounce",
            speed_min=0.0,
            speed_max=0.0,
            turn_sigma=0.35,
            stop_prob=0.0,
    ):
        self.x = x
        self.y = y
        self.radius = radius
        self.is_dynamic = is_dynamic
        self.is_known = bool(is_known)
        self.vx = vx
        self.vy = vy
        self.pattern = pattern
        self.speed_min = float(speed_min)
        self.speed_max = float(speed_max)
        self.turn_sigma = float(turn_sigma)
        self.stop_prob = float(stop_prob)

    def step(self, dt, map_width, map_height):
        if not self.is_dynamic:
            return

        if self.pattern == "stop_and_go":
            if np.random.rand() < self.stop_prob:
                self.vx = 0.0
                self.vy = 0.0
            elif self.vx == 0.0 and self.vy == 0.0:
                speed = np.random.uniform(self.speed_min, self.speed_max)
                ang = np.random.uniform(0.0, 2.0 * math.pi)
                self.vx = float(speed * math.cos(ang))
                self.vy = float(speed * math.sin(ang))
        elif self.pattern == "random_walk":
            speed = math.hypot(self.vx, self.vy)
            if speed < 1e-9:
                speed = np.random.uniform(self.speed_min, self.speed_max)
                ang = np.random.uniform(0.0, 2.0 * math.pi)
            else:
                ang = math.atan2(self.vy, self.vx)
                ang += np.random.normal(0.0, self.turn_sigma) * dt
                speed *= float(np.clip(1.0 + np.random.normal(0.0, 0.15) * dt, 0.7, 1.3))
                speed = float(np.clip(speed, self.speed_min, self.speed_max))
            self.vx = float(speed * math.cos(ang))
            self.vy = float(speed * math.sin(ang))

        self.x += self.vx * dt
        self.y += self.vy * dt

        margin = self.radius + 0.5
        if self.x < margin or self.x > map_width - margin:
            self.vx = -self.vx
            self.x = np.clip(self.x, margin, map_width - margin)
        if self.y < margin or self.y > map_height - margin:
            self.vy = -self.vy
            self.y = np.clip(self.y, margin, map_height - margin)

    def to_dict(self):
        return {
            'x': self.x,
            'y': self.y,
            'radius': self.radius,
            'is_dynamic': self.is_dynamic,
            'is_known': self.is_known,
            'vx': self.vx,
            'vy': self.vy
        }


class NavigationEnv:
    """导航环境（优化版）"""

    def __init__(
            self,
            render_mode=None,
            use_enhanced_state: bool = False,
            enhanced_state_config: dict | None = None,
            dynamic_profile: str | None = None,
            dynamic_speed_min: float | None = None,
            dynamic_speed_max: float | None = None,
            dynamic_patterns: tuple | None = None,
            dynamic_stop_prob: float = 0.0,
            disable_lidar_diff: bool = False,
            disable_delta_yaw: bool = False,
    ):
        self.render_mode = render_mode

        self.robot = Robot()
        self.lidar = LiDAR()
        self.obstacles = []
        self.walls = []

        self.goal_x = 0.0
        self.goal_y = 0.0

        self.use_enhanced_state = bool(use_enhanced_state)
        self._enhanced_state = None

        # 动态障碍物参数
        profile = str(dynamic_profile) if dynamic_profile is not None else str(
            getattr(EnvConfig, 'DYNAMIC_PROFILE_DEFAULT', 'baseline'))

        if dynamic_speed_min is None:
            if profile.lower() in ('lstm', 'fast', 'enhanced'):
                dynamic_speed_min = float(getattr(EnvConfig, 'DYNAMIC_OBS_VEL_MIN_FAST', EnvConfig.DYNAMIC_OBS_VEL_MIN))
            else:
                dynamic_speed_min = float(getattr(EnvConfig, 'DYNAMIC_OBS_VEL_MIN', 0.05))
        if dynamic_speed_max is None:
            if profile.lower() in ('lstm', 'fast', 'enhanced'):
                dynamic_speed_max = float(getattr(EnvConfig, 'DYNAMIC_OBS_VEL_MAX_FAST', EnvConfig.DYNAMIC_OBS_VEL_MAX))
            else:
                dynamic_speed_max = float(getattr(EnvConfig, 'DYNAMIC_OBS_VEL_MAX', 0.15))

        self.dynamic_speed_min = float(dynamic_speed_min)
        self.dynamic_speed_max = float(dynamic_speed_max)
        if self.dynamic_speed_max < self.dynamic_speed_min:
            self.dynamic_speed_min, self.dynamic_speed_max = self.dynamic_speed_max, self.dynamic_speed_min

        if dynamic_patterns is None:
            if profile.lower() in ('lstm', 'fast', 'enhanced'):
                dynamic_patterns = tuple(getattr(EnvConfig, 'DYNAMIC_OBS_PATTERNS_FAST', ("bounce",)))
            else:
                dynamic_patterns = ("bounce",)
        self.dynamic_patterns = tuple(dynamic_patterns)

        if dynamic_stop_prob == 0.0 and profile.lower() in ('lstm', 'fast', 'enhanced'):
            dynamic_stop_prob = float(getattr(EnvConfig, 'DYNAMIC_OBS_STOP_PROB_FAST', 0.0))
        self.dynamic_stop_prob = float(dynamic_stop_prob)

        self._disable_lidar_diff = bool(disable_lidar_diff)
        self._disable_delta_yaw = bool(disable_delta_yaw)

        self.step_count = 0
        self.previous_distance = 0.0
        self.previous_heading = 0.0
        self.last_action = "FORWARD"

        self.episode_success = False
        self.episode_failure = False

        self._create_walls()

        self.prev_action = np.zeros(2, dtype=np.float32)

        # 状态空间
        if self.use_enhanced_state:
            cfg = {
                'lidar_rays': EnvConfig.LIDAR_RAYS,
                'n_sectors': getattr(EnvConfig, 'ENHANCED_N_SECTORS', 16),
                'max_range': EnvConfig.LIDAR_MAX_RANGE,
                'min_range': EnvConfig.LIDAR_MIN_RANGE,
                'dt': EnvConfig.DT,
                'yaw_rate_max': EnvConfig.MAX_ANGULAR_VEL,
                'max_speed': float(EnvConfig.MAX_LINEAR_VEL),
                'dynamic_obs_max_speed': float(self.dynamic_speed_max),
                'relative_speed_max': float(EnvConfig.MAX_LINEAR_VEL + self.dynamic_speed_max),
                'use_lidar_diff': bool(
                    getattr(EnvConfig, 'ENHANCED_USE_LIDAR_DIFF_DEFAULT', True) and (not self._disable_lidar_diff)),
                'use_delta_yaw': bool(
                    getattr(EnvConfig, 'ENHANCED_USE_DELTA_YAW_DEFAULT', True) and (not self._disable_delta_yaw)),
                'map_width': EnvConfig.MAP_WIDTH,
                'map_height': EnvConfig.MAP_HEIGHT,
            }
            if isinstance(enhanced_state_config, dict):
                cfg.update(enhanced_state_config)
            self._enhanced_state = EnhancedSim2RealStateV2(cfg)
            self.state_dim = int(self._enhanced_state.state_dim)
        else:
            self.state_dim = EnvConfig.LIDAR_RAYS + 2 + 2 + 1 + 2
        self.action_dim = 2

        # 渲染相关
        self.fig = None
        self.ax = None
        if self.render_mode == 'human':
            plt.ion()  # 交互模式
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_xlim(0, EnvConfig.MAP_WIDTH)
            self.ax.set_ylim(0, EnvConfig.MAP_HEIGHT)
            self.ax.set_aspect('equal')
            self.ax.grid(True, alpha=0.3)
            self.ax.set_xlabel('X (m)')
            self.ax.set_ylabel('Y (m)')
            self.ax.set_title('UAV Navigation Environment')

    def _create_walls(self):
        w, h = EnvConfig.MAP_WIDTH, EnvConfig.MAP_HEIGHT
        self.walls = [
            {'x1': 0, 'y1': 0, 'x2': w, 'y2': 0},
            {'x1': w, 'y1': 0, 'x2': w, 'y2': h},
            {'x1': w, 'y1': h, 'x2': 0, 'y2': h},
            {'x1': 0, 'y1': h, 'x2': 0, 'y2': 0},
        ]

    def _create_obstacles(self):
        self.obstacles = []
        known_static = int(np.clip(getattr(EnvConfig, 'KNOWN_STATIC_OBSTACLES', EnvConfig.NUM_STATIC_OBSTACLES), 0,
                                   EnvConfig.NUM_STATIC_OBSTACLES))

        for i in range(EnvConfig.NUM_STATIC_OBSTACLES):
            while True:
                x = np.random.uniform(1.5, EnvConfig.MAP_WIDTH - 1.5)
                y = np.random.uniform(1.5, EnvConfig.MAP_HEIGHT - 1.5)
                radius = np.random.uniform(EnvConfig.OBSTACLE_RADIUS_MIN, EnvConfig.OBSTACLE_RADIUS_MAX)

                if self._is_valid_obstacle_position(x, y, radius):
                    self.obstacles.append(
                        Obstacle(x, y, radius, is_dynamic=False, is_known=(i < known_static))
                    )
                    break

        for _ in range(EnvConfig.NUM_DYNAMIC_OBSTACLES):
            while True:
                x = np.random.uniform(2, EnvConfig.MAP_WIDTH - 2)
                y = np.random.uniform(2, EnvConfig.MAP_HEIGHT - 2)
                radius = np.random.uniform(EnvConfig.OBSTACLE_RADIUS_MIN, EnvConfig.OBSTACLE_RADIUS_MAX)

                if self._is_valid_obstacle_position(x, y, radius):
                    speed = np.random.uniform(self.dynamic_speed_min, self.dynamic_speed_max)
                    angle = np.random.uniform(0.0, 2.0 * math.pi)
                    vx = float(speed * math.cos(angle))
                    vy = float(speed * math.sin(angle))

                    pattern = "bounce"
                    if len(self.dynamic_patterns) > 0:
                        pattern = str(np.random.choice(self.dynamic_patterns))

                    self.obstacles.append(
                        Obstacle(
                            x, y, radius,
                            is_dynamic=True,
                            is_known=False,
                            vx=vx, vy=vy,
                            pattern=pattern,
                            speed_min=self.dynamic_speed_min,
                            speed_max=self.dynamic_speed_max,
                            stop_prob=self.dynamic_stop_prob,
                        )
                    )
                    break

    def _is_valid_obstacle_position(self, x, y, radius):
        dist_to_robot = math.hypot(x - self.robot.x, y - self.robot.y)
        if dist_to_robot < radius + self.robot.radius + 0.5:
            return False

        dist_to_goal = math.hypot(x - self.goal_x, y - self.goal_y)
        if dist_to_goal < radius + EnvConfig.GOAL_RADIUS + 0.3:
            return False

        for obs in self.obstacles:
            dist = math.hypot(x - obs.x, y - obs.y)
            if dist < radius + obs.radius + 0.2:
                return False

        return True

    def reset(self):
        self.step_count = 0
        self.episode_success = False
        self.episode_failure = False

        start_x = np.random.uniform(1, 2)
        start_y = np.random.uniform(1, EnvConfig.MAP_HEIGHT - 1)
        start_theta = np.random.uniform(-math.pi, math.pi)
        self.robot.reset(start_x, start_y, start_theta)

        self.goal_x = np.random.uniform(EnvConfig.MAP_WIDTH - 2, EnvConfig.MAP_WIDTH - 1)
        self.goal_y = np.random.uniform(1, EnvConfig.MAP_HEIGHT - 1)

        self._create_obstacles()

        self.prev_action = np.zeros(2, dtype=np.float32)
        if self._enhanced_state is not None:
            self._enhanced_state.reset()

        self.previous_distance = self._get_distance_to_goal()
        self.previous_heading = self._get_heading_to_goal()

        # 渲染初始状态（如果启用）
        if self.render_mode == 'human':
            self.render()

        return self._get_state()

    def step(self, action):
        self.step_count += 1

        linear_vel = float(action[0])
        angular_vel = float(action[1])

        self.prev_action = np.array([linear_vel, angular_vel], dtype=np.float32)

        if linear_vel >= 0 and abs(angular_vel) < 0.25:
            self.last_action = "FORWARD"
        elif angular_vel > 0:
            self.last_action = "TURN_LEFT"
        elif angular_vel < 0:
            self.last_action = "TURN_RIGHT"
        else:
            self.last_action = "STOP"

        self.robot.step(linear_vel, angular_vel, EnvConfig.DT)

        for obs in self.obstacles:
            obs.step(EnvConfig.DT, EnvConfig.MAP_WIDTH, EnvConfig.MAP_HEIGHT)

        state = self._get_state()
        done, info = self._check_done()
        reward = self._compute_reward(done, info)

        # 渲染环境（如果启用）
        if self.render_mode == 'human':
            self.render()

        return state, reward, done, info

    def _get_state(self):
        self._update_obstacle_knowledge()
        obs_dicts = [obs.to_dict() for obs in self.obstacles]
        lidar_ranges = self.lidar.scan(self.robot, obs_dicts, self.walls)

        if self._enhanced_state is not None:
            return self._enhanced_state.build_state(
                robot=self.robot,
                goal_xy=(self.goal_x, self.goal_y),
                raw_lidar_ranges=lidar_ranges,
                prev_action=self.prev_action,
            )

        lidar_normalized = lidar_ranges / EnvConfig.LIDAR_MAX_RANGE

        distance_to_goal = self._get_distance_to_goal()
        heading_to_goal = self._get_heading_to_goal()
        goal_info = np.array([
            distance_to_goal / EnvConfig.MAP_WIDTH,
            heading_to_goal / math.pi
        ])

        robot_pos = np.array([
            self.robot.x / EnvConfig.MAP_WIDTH,
            self.robot.y / EnvConfig.MAP_HEIGHT
        ])

        robot_theta = np.array([self.robot.theta / math.pi])

        robot_vel = np.array([
            self.robot.linear_vel / EnvConfig.MAX_LINEAR_VEL,
            self.robot.angular_vel / EnvConfig.MAX_ANGULAR_VEL
        ])

        state = np.concatenate([
            lidar_normalized,
            goal_info,
            robot_pos,
            robot_theta,
            robot_vel
        ])

        return state.astype(np.float32)

    def _update_obstacle_knowledge(self):
        """动态障碍物：进入LiDAR量程后标记为已知。"""
        rx, ry = self.robot.x, self.robot.y
        detect_r = float(EnvConfig.LIDAR_MAX_RANGE)
        for obs in self.obstacles:
            if obs.is_dynamic and (not obs.is_known):
                if math.hypot(obs.x - rx, obs.y - ry) <= detect_r + obs.radius:
                    obs.is_known = True

    def _get_distance_to_goal(self):
        return math.hypot(self.goal_x - self.robot.x, self.goal_y - self.robot.y)

    def _get_heading_to_goal(self):
        goal_angle = math.atan2(self.goal_y - self.robot.y, self.goal_x - self.robot.x)
        heading = goal_angle - self.robot.theta
        return Robot.normalize_angle(heading)

    def _check_done(self):
        info = {'reason': None}

        if self._get_distance_to_goal() < EnvConfig.GOAL_RADIUS:
            self.episode_success = True
            info['reason'] = 'goal_reached'
            return True, info

        for obs in self.obstacles:
            dist = math.hypot(self.robot.x - obs.x, self.robot.y - obs.y)
            if dist < self.robot.radius + obs.radius:
                self.episode_failure = True
                info['reason'] = 'collision_obstacle'
                return True, info

        if (self.robot.x < self.robot.radius or
                self.robot.x > EnvConfig.MAP_WIDTH - self.robot.radius or
                self.robot.y < self.robot.radius or
                self.robot.y > EnvConfig.MAP_HEIGHT - self.robot.radius):
            self.episode_failure = True
            info['reason'] = 'collision_wall'
            return True, info

        if self.step_count >= EnvConfig.MAX_STEPS:
            self.episode_failure = True
            info['reason'] = 'max_steps'
            return True, info

        return False, info

    def _compute_reward(self, done, info):
        current_distance = self._get_distance_to_goal()
        current_heading = self._get_heading_to_goal()

        # ⚡ 关键修改：使用连续距离变化，而非二元判断
        # distance_diff = current - previous（负值表示接近）
        distance_diff = current_distance - self.previous_distance

        # 连续距离奖励：接近得正奖励，远离得负奖励
        # 总距离奖励 = 初始距离 * DISTANCE_SCALE（约 8m * 10 = 80）
        DISTANCE_SCALE = 10.0
        dtg_reward = -distance_diff * DISTANCE_SCALE

        step_reward = RewardConfig.STEP_PENALTY

        # 朝向奖励（保持简单）
        htg_reward = 0
        if RewardConfig.HTG_REWARD != 0 or RewardConfig.HTG_PENALTY != 0:
            heading_diff = abs(current_heading) - abs(self.previous_heading)
            if heading_diff < 0:
                htg_reward = RewardConfig.HTG_REWARD
            else:
                htg_reward = RewardConfig.HTG_PENALTY

        non_terminal_reward = step_reward + dtg_reward + htg_reward

        if done:
            if info['reason'] == 'goal_reached':
                terminal_reward = RewardConfig.GOAL_REWARD
            elif info['reason'] in ['collision_obstacle', 'collision_wall']:
                terminal_reward = RewardConfig.COLLISION_PENALTY
            else:
                terminal_reward = RewardConfig.TIMEOUT_PENALTY  # 超时也要惩罚！
            reward = non_terminal_reward + terminal_reward
        else:
            reward = non_terminal_reward

        self.previous_distance = current_distance
        self.previous_heading = current_heading

        return reward

    def get_episode_status(self):
        return self.episode_success, self.episode_failure

    def render(self):
        """渲染环境"""
        if self.render_mode != 'human' or self.fig is None:
            return

        # 清除之前的绘图
        self.ax.clear()

        # 设置坐标轴
        self.ax.set_xlim(0, EnvConfig.MAP_WIDTH)
        self.ax.set_ylim(0, EnvConfig.MAP_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')

        # 标题显示步数和状态
        status = "SUCCESS" if self.episode_success else ("FAILURE" if self.episode_failure else "RUNNING")
        self.ax.set_title(f'Step: {self.step_count} | Status: {status}')

        # 绘制墙壁
        for wall in self.walls:
            self.ax.plot([wall['x1'], wall['x2']], [wall['y1'], wall['y2']],
                         'k-', linewidth=3, label='Wall' if wall == self.walls[0] else '')

        # 绘制目标点
        goal_circle = Circle((self.goal_x, self.goal_y), EnvConfig.GOAL_RADIUS,
                             color='green', alpha=0.3, label='Goal')
        self.ax.add_patch(goal_circle)
        self.ax.plot(self.goal_x, self.goal_y, 'g*', markersize=20)

        # 绘制障碍物
        for i, obs in enumerate(self.obstacles):
            if not obs.is_dynamic:  # 静态障碍物（蓝）
                if obs.is_known:
                    obs_circle = Circle((obs.x, obs.y), obs.radius,
                                        facecolor='blue', edgecolor='blue', alpha=0.45,
                                        label='Known Static' if i == 0 else '')
                else:
                    obs_circle = Circle((obs.x, obs.y), obs.radius,
                                        facecolor='none', edgecolor='blue', linewidth=1.8, alpha=0.9,
                                        label='Unknown Static' if i == 0 else '')
            else:  # 动态障碍物（红）
                if obs.is_known:
                    obs_circle = Circle((obs.x, obs.y), obs.radius,
                                        facecolor='red', edgecolor='red', alpha=0.45,
                                        label='Known Dynamic' if i == 1 else '')
                else:
                    obs_circle = Circle((obs.x, obs.y), obs.radius,
                                        facecolor='none', edgecolor='red', linewidth=1.8, alpha=0.9,
                                        label='Unknown Dynamic' if i == 1 else '')
            self.ax.add_patch(obs_circle)

            # 绘制动态障碍物的速度向量
            if obs.is_dynamic:
                arrow_scale = 0.5
                self.ax.arrow(obs.x, obs.y,
                              obs.vx * arrow_scale, obs.vy * arrow_scale,
                              head_width=0.15, head_length=0.1, fc='darkred', ec='darkred')

        # 绘制机器人
        robot_circle = Circle((self.robot.x, self.robot.y), self.robot.radius,
                              color='blue', alpha=0.7, label='UAV')
        self.ax.add_patch(robot_circle)

        # 绘制机器人朝向
        arrow_length = 0.3
        dx = arrow_length * np.cos(self.robot.theta)
        dy = arrow_length * np.sin(self.robot.theta)
        self.ax.arrow(self.robot.x, self.robot.y, dx, dy,
                      head_width=0.15, head_length=0.1, fc='darkblue', ec='darkblue')

        # 绘制LiDAR扫描线（只显示检测到障碍物的）
        if hasattr(self.lidar, 'last_readings'):
            for i, reading in enumerate(self.lidar.last_readings):
                if reading < EnvConfig.LIDAR_MAX_RANGE - 0.01:  # 检测到障碍物
                    angle = self.robot.theta + (i / EnvConfig.LIDAR_RAYS) * 2 * np.pi
                    end_x = self.robot.x + reading * np.cos(angle)
                    end_y = self.robot.y + reading * np.sin(angle)
                    self.ax.plot([self.robot.x, end_x], [self.robot.y, end_y],
                                 'y-', alpha=0.3, linewidth=0.5)

        # 添加图例
        self.ax.legend(loc='upper right', fontsize='small')

        # 添加信息文本
        info_text = f"Robot: ({self.robot.x:.2f}, {self.robot.y:.2f})\n"
        info_text += f"Goal: ({self.goal_x:.2f}, {self.goal_y:.2f})\n"
        info_text += f"Distance: {np.sqrt((self.robot.x - self.goal_x) ** 2 + (self.robot.y - self.goal_y) ** 2):.2f}m"
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                     verticalalignment='top', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 刷新显示
        plt.pause(0.001)

    def close(self):
        """关闭渲染窗口"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == '__main__':
    env = NavigationEnv()
    state = env.reset()
    print(f"State shape: {state.shape}")

    for i in range(10):
        action = [0.1, 0.5]
        next_state, reward, done, info = env.step(action)
        print(f"Step {i + 1}: reward={reward:.2f}, done={done}, info={info}")
        if done:
            break
