#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化模块
使用pygame实现实时渲染
"""

import math
import numpy as np

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("Warning: pygame not installed. Visualization will not be available.")
    print("Install with: pip install pygame")

from config import EnvConfig, VisConfig


class Visualizer:
    """环境可视化器"""
    
    def __init__(self, env):
        if not PYGAME_AVAILABLE:
            raise ImportError("pygame is required for visualization")
        
        self.env = env
        self.width = VisConfig.WINDOW_WIDTH
        self.height = VisConfig.WINDOW_HEIGHT
        
        # 计算比例尺
        self.scale_x = self.width / EnvConfig.MAP_WIDTH
        self.scale_y = self.height / EnvConfig.MAP_HEIGHT
        
        # 初始化pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("DDPG Navigation - 32-Line LiDAR")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        
        # 颜色定义
        self.colors = {
            'background': (240, 240, 240),
            'wall': (50, 50, 50),
            'robot': (0, 100, 200),
            'robot_direction': (200, 0, 0),
            'goal': (0, 200, 0),
            'static_obstacle': (100, 100, 100),
            'dynamic_obstacle': (200, 100, 0),
            'lidar': (255, 0, 0),
            'lidar_hit': (255, 100, 100),
            'trajectory': (100, 100, 255),
            'text': (0, 0, 0),
        }
        
        # 轨迹缓存
        self.trajectory_cache = []
        
    def world_to_screen(self, x, y):
        """世界坐标转屏幕坐标"""
        screen_x = int(x * self.scale_x)
        screen_y = int(self.height - y * self.scale_y)  # y轴翻转
        return screen_x, screen_y
    
    def render(self, trajectory=None):
        """渲染一帧"""
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return False
        
        # 清屏
        self.screen.fill(self.colors['background'])
        
        # 绘制网格
        self._draw_grid()
        
        # 绘制墙壁
        self._draw_walls()
        
        # 绘制障碍物
        self._draw_obstacles()
        
        # 绘制目标
        self._draw_goal()
        
        # 绘制轨迹
        if trajectory:
            self._draw_trajectory(trajectory)
        
        # 绘制激光雷达
        if VisConfig.SHOW_LIDAR:
            self._draw_lidar()
        
        # 绘制机器人
        self._draw_robot()
        
        # 绘制信息面板
        self._draw_info_panel()
        
        # 更新显示
        pygame.display.flip()
        self.clock.tick(VisConfig.FPS)
        
        return True
    
    def _draw_grid(self):
        """绘制背景网格"""
        grid_color = (220, 220, 220)
        for x in range(0, int(EnvConfig.MAP_WIDTH) + 1):
            start = self.world_to_screen(x, 0)
            end = self.world_to_screen(x, EnvConfig.MAP_HEIGHT)
            pygame.draw.line(self.screen, grid_color, start, end, 1)
        for y in range(0, int(EnvConfig.MAP_HEIGHT) + 1):
            start = self.world_to_screen(0, y)
            end = self.world_to_screen(EnvConfig.MAP_WIDTH, y)
            pygame.draw.line(self.screen, grid_color, start, end, 1)
    
    def _draw_walls(self):
        """绘制墙壁"""
        for wall in self.env.walls:
            start = self.world_to_screen(wall['x1'], wall['y1'])
            end = self.world_to_screen(wall['x2'], wall['y2'])
            pygame.draw.line(self.screen, self.colors['wall'], start, end, 3)
    
    def _draw_obstacles(self):
        """绘制障碍物"""
        for obs in self.env.obstacles:
            center = self.world_to_screen(obs.x, obs.y)
            radius = int(obs.radius * self.scale_x)
            
            if obs.is_dynamic:
                color = self.colors['dynamic_obstacle']
                # 绘制速度方向箭头
                if abs(obs.vx) > 0.01 or abs(obs.vy) > 0.01:
                    vel_scale = 2.0  # 速度箭头缩放
                    end_x = obs.x + obs.vx * vel_scale
                    end_y = obs.y + obs.vy * vel_scale
                    end = self.world_to_screen(end_x, end_y)
                    pygame.draw.line(self.screen, (255, 150, 0), center, end, 2)
            else:
                color = self.colors['static_obstacle']
            
            pygame.draw.circle(self.screen, color, center, radius)
            pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 2)
    
    def _draw_goal(self):
        """绘制目标"""
        center = self.world_to_screen(self.env.goal_x, self.env.goal_y)
        radius = int(EnvConfig.GOAL_RADIUS * self.scale_x)
        
        # 目标区域（半透明）
        goal_surface = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(goal_surface, (*self.colors['goal'], 100), (radius, radius), radius)
        self.screen.blit(goal_surface, (center[0] - radius, center[1] - radius))
        
        # 目标边界
        pygame.draw.circle(self.screen, self.colors['goal'], center, radius, 2)
        
        # 目标标记（十字）
        cross_size = 10
        pygame.draw.line(self.screen, self.colors['goal'], 
                        (center[0] - cross_size, center[1]), 
                        (center[0] + cross_size, center[1]), 3)
        pygame.draw.line(self.screen, self.colors['goal'], 
                        (center[0], center[1] - cross_size), 
                        (center[0], center[1] + cross_size), 3)
    
    def _draw_robot(self):
        """绘制机器人"""
        robot = self.env.robot
        center = self.world_to_screen(robot.x, robot.y)
        radius = int(robot.radius * self.scale_x)
        
        # 机器人身体
        pygame.draw.circle(self.screen, self.colors['robot'], center, radius)
        pygame.draw.circle(self.screen, (0, 0, 0), center, radius, 2)
        
        # 方向指示线
        dir_length = radius * 1.5
        dir_x = robot.x + math.cos(robot.theta) * robot.radius * 1.5
        dir_y = robot.y + math.sin(robot.theta) * robot.radius * 1.5
        dir_end = self.world_to_screen(dir_x, dir_y)
        pygame.draw.line(self.screen, self.colors['robot_direction'], center, dir_end, 3)
    
    def _draw_lidar(self):
        """绘制激光雷达射线"""
        robot = self.env.robot
        obs_dicts = [obs.to_dict() for obs in self.env.obstacles]
        ranges = self.env.lidar.scan(robot, obs_dicts, self.env.walls)
        
        robot_pos = self.world_to_screen(robot.x, robot.y)
        
        for i, (angle, dist) in enumerate(zip(self.env.lidar.ray_angles, ranges)):
            ray_angle = robot.theta + angle - self.env.lidar.fov / 2
            
            # 射线终点
            end_x = robot.x + dist * math.cos(ray_angle)
            end_y = robot.y + dist * math.sin(ray_angle)
            end_pos = self.world_to_screen(end_x, end_y)
            
            # 根据距离选择颜色
            if dist < EnvConfig.LIDAR_MAX_RANGE * 0.3:
                color = (255, 0, 0)  # 近距离红色
            elif dist < EnvConfig.LIDAR_MAX_RANGE * 0.6:
                color = (255, 165, 0)  # 中距离橙色
            else:
                color = (0, 255, 0)  # 远距离绿色
            
            pygame.draw.line(self.screen, color, robot_pos, end_pos, 1)
            
            # 绘制击中点
            if dist < EnvConfig.LIDAR_MAX_RANGE - 0.1:
                pygame.draw.circle(self.screen, self.colors['lidar_hit'], end_pos, 3)
    
    def _draw_trajectory(self, trajectory):
        """绘制轨迹"""
        if len(trajectory) < 2:
            return
        
        # 只保留最近的点
        points = trajectory[-VisConfig.TRAJECTORY_LENGTH:]
        
        for i in range(len(points) - 1):
            start = self.world_to_screen(points[i][0], points[i][1])
            end = self.world_to_screen(points[i+1][0], points[i+1][1])
            
            # 渐变颜色（越旧越淡）
            alpha = int(255 * (i + 1) / len(points))
            color = (100, 100, 255, alpha)
            
            pygame.draw.line(self.screen, self.colors['trajectory'], start, end, 2)
    
    def _draw_info_panel(self):
        """绘制信息面板"""
        robot = self.env.robot
        distance = self.env._get_distance_to_goal()
        heading = math.degrees(self.env._get_heading_to_goal())
        
        info_texts = [
            f"Step: {self.env.step_count}",
            f"Robot: ({robot.x:.2f}, {robot.y:.2f})",
            f"Goal: ({self.env.goal_x:.2f}, {self.env.goal_y:.2f})",
            f"Distance: {distance:.2f} m",
            f"Heading: {heading:.1f}°",
            f"Linear Vel: {robot.linear_vel:.2f} m/s",
            f"Angular Vel: {math.degrees(robot.angular_vel):.1f}°/s",
            f"Static Obs: {EnvConfig.NUM_STATIC_OBSTACLES}",
            f"Dynamic Obs: {EnvConfig.NUM_DYNAMIC_OBSTACLES}",
            f"LiDAR Rays: {EnvConfig.LIDAR_RAYS}",
        ]
        
        # 绘制半透明背景
        panel_width = 180
        panel_height = len(info_texts) * 20 + 10
        panel_surface = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surface, (255, 255, 255, 200), (0, 0, panel_width, panel_height))
        self.screen.blit(panel_surface, (5, 5))
        
        # 绘制文本
        for i, text in enumerate(info_texts):
            text_surface = self.font.render(text, True, self.colors['text'])
            self.screen.blit(text_surface, (10, 10 + i * 20))
    
    def close(self):
        """关闭可视化器"""
        pygame.quit()


# 测试代码
if __name__ == '__main__':
    from environment import NavigationEnv
    import time
    
    if not PYGAME_AVAILABLE:
        print("Cannot run visualization test: pygame not installed")
        exit(1)
    
    env = NavigationEnv()
    visualizer = Visualizer(env)
    
    state = env.reset()
    trajectory = [(env.robot.x, env.robot.y)]
    
    running = True
    step = 0
    
    print("Press Q to quit, R to reset")
    
    while running:
        # 处理输入
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False
        if keys[pygame.K_r]:
            state = env.reset()
            trajectory = [(env.robot.x, env.robot.y)]
            step = 0
        
        # 随机动作（用于测试）
        action = [
            np.random.uniform(0, EnvConfig.MAX_LINEAR_VEL),
            np.random.uniform(-EnvConfig.MAX_ANGULAR_VEL, EnvConfig.MAX_ANGULAR_VEL) * 0.3
        ]
        
        state, reward, done, info = env.step(action)
        trajectory.append((env.robot.x, env.robot.y))
        
        running = visualizer.render(trajectory)
        
        step += 1
        if done:
            print(f"Episode ended: {info['reason']}, steps: {step}, reward: {reward:.2f}")
            time.sleep(1)
            state = env.reset()
            trajectory = [(env.robot.x, env.robot.y)]
            step = 0
    
    visualizer.close()
