#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_with_gif.py - 带GIF可视化的测试脚本

功能：
1. 测试训练好的模型
2. 实时可视化导航过程
3. 将测试过程保存为GIF动画

用法：
  python test_with_gif.py --model models/lstm_ddpg_att_best.pth --episodes 2 --save_gif
  python test_with_gif.py --model models/lstm_ddpg_att_best.pth --episodes 10 --gif_name test_result.gif
"""

import argparse
import os
import time
from collections import deque
from typing import List, Tuple, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

import torch

from config import EnvConfig, MODEL_DIR, RESULT_DIR
from environment import NavigationEnv


# ==================== PyTorch 2.6+ 兼容加载 ====================
def safe_torch_load(path: str, map_location=None):
    """兼容 PyTorch 2.6+ 的模型加载"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ==================== 可视化类 ====================
class NavigationVisualizer:
    """导航可视化器，支持实时显示和GIF保存"""
    
    def __init__(self, env: NavigationEnv, figsize=(8, 8)):
        self.env = env
        self.figsize = figsize
        self.frames = []  # 存储帧用于GIF
        
        # 设置matplotlib
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.setup_plot()
        
    def setup_plot(self):
        """设置绘图"""
        self.ax.set_xlim(0, EnvConfig.MAP_WIDTH)
        self.ax.set_ylim(0, EnvConfig.MAP_HEIGHT)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        
    def render(self, trajectory: List[Tuple[float, float]], 
               step: int = 0, episode: int = 0, 
               info: Dict[str, Any] = None,
               save_frame: bool = True) -> np.ndarray:
        """渲染当前帧
        
        Args:
            trajectory: 机器人轨迹点列表
            step: 当前步数
            episode: 当前episode
            info: 额外信息
            save_frame: 是否保存帧用于GIF
            
        Returns:
            frame: 当前帧的图像数组
        """
        self.ax.clear()
        self.setup_plot()
        
        # 标题
        title = f"Episode {episode+1}, Step {step}"
        if info and 'reason' in info:
            title += f" - {info['reason']}"
        self.ax.set_title(title)
        
        # 绘制目标
        goal_circle = patches.Circle(
            (self.env.goal_x, self.env.goal_y), 
            EnvConfig.GOAL_RADIUS,
            color='green', alpha=0.5, label='Goal'
        )
        self.ax.add_patch(goal_circle)
        self.ax.plot(self.env.goal_x, self.env.goal_y, 'g*', markersize=15)
        
        # 绘制障碍物
        for obs in self.env.obstacles:
            color = 'red' if obs.is_dynamic else 'gray'
            alpha = 0.7 if obs.is_dynamic else 0.5
            circle = patches.Circle(
                (obs.x, obs.y), obs.radius,
                color=color, alpha=alpha
            )
            self.ax.add_patch(circle)
            
            # 动态障碍物速度箭头
            if obs.is_dynamic and (abs(obs.vx) > 0.01 or abs(obs.vy) > 0.01):
                self.ax.arrow(obs.x, obs.y, obs.vx*0.5, obs.vy*0.5,
                             head_width=0.1, head_length=0.05, fc='red', ec='red')
        
        # 绘制轨迹
        if len(trajectory) > 1:
            traj = np.array(trajectory)
            self.ax.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1.5, alpha=0.7, label='Trajectory')
        
        # 绘制机器人
        robot = self.env.robot
        robot_circle = patches.Circle(
            (robot.x, robot.y), robot.radius,
            color='blue', alpha=0.8
        )
        self.ax.add_patch(robot_circle)
        
        # 机器人朝向箭头
        arrow_len = 0.3
        dx = arrow_len * np.cos(robot.theta)
        dy = arrow_len * np.sin(robot.theta)
        self.ax.arrow(robot.x, robot.y, dx, dy,
                     head_width=0.1, head_length=0.05, fc='blue', ec='blue')
        
        # 绘制LiDAR射线（可选，显示前8条）
        if hasattr(self.env, 'lidar'):
            lidar = self.env.lidar
            obs_dicts = [obs.to_dict() for obs in self.env.obstacles]
            ranges = lidar.scan(robot, obs_dicts, self.env.walls)
            
            # 只绘制部分射线避免太密集
            num_show = min(16, len(ranges))
            indices = np.linspace(0, len(ranges)-1, num_show, dtype=int)
            
            for i in indices:
                angle = robot.theta + lidar.ray_offsets[i]
                r = min(ranges[i], lidar.max_range)
                end_x = robot.x + r * np.cos(angle)
                end_y = robot.y + r * np.sin(angle)
                
                # 根据距离着色
                if r < 0.5:
                    color = 'red'
                    alpha = 0.5
                elif r < 1.0:
                    color = 'orange'
                    alpha = 0.3
                else:
                    color = 'yellow'
                    alpha = 0.2
                    
                self.ax.plot([robot.x, end_x], [robot.y, end_y], 
                            color=color, alpha=alpha, linewidth=0.5)
        
        # 图例
        self.ax.legend(loc='upper right')
        
        # 更新显示
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # 保存帧
        if save_frame:
            # 将图转换为numpy数组
            self.fig.canvas.draw()
            frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            frame = frame.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            self.frames.append(frame)
            
        return frame if save_frame else None
    
    def save_gif(self, filename: str, fps: int = 10):
        """保存为GIF
        
        Args:
            filename: 输出文件名
            fps: 帧率
        """
        if not self.frames:
            print("No frames to save!")
            return
            
        print(f"Saving GIF with {len(self.frames)} frames to {filename}...")
        
        # 使用imageio保存GIF
        try:
            import imageio
            imageio.mimsave(filename, self.frames, fps=fps, loop=0)
            print(f"GIF saved successfully: {filename}")
        except ImportError:
            # 备选：使用matplotlib
            print("imageio not found, using matplotlib...")
            fig, ax = plt.subplots(figsize=self.figsize)
            
            def update(frame_idx):
                ax.clear()
                ax.imshow(self.frames[frame_idx])
                ax.axis('off')
                return []
            
            anim = FuncAnimation(fig, update, frames=len(self.frames), interval=1000/fps)
            anim.save(filename, writer=PillowWriter(fps=fps))
            plt.close(fig)
            print(f"GIF saved successfully: {filename}")
    
    def clear_frames(self):
        """清空帧缓存"""
        self.frames = []
        
    def close(self):
        """关闭可视化"""
        plt.ioff()
        plt.close(self.fig)


# ==================== 测试函数 ====================
def test_with_visualization(args):
    """带可视化的测试"""
    
    print("=" * 60)
    print("Navigation Test with Visualization")
    print("=" * 60)
    
    # 加载模型
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")
    
    ckpt = safe_torch_load(args.model, map_location="cpu")
    state_meta = ckpt.get("state_meta", {}) or {}
    
    # 从checkpoint获取配置
    n_sectors = state_meta.get("n_sectors", 16)
    use_lidar_diff = not state_meta.get("disable_lidar_diff", False)
    use_delta_yaw = not state_meta.get("disable_delta_yaw", False)
    history_len = state_meta.get("history_len", 5)
    
    # 环境配置
    enhanced_cfg = {
        "n_sectors": n_sectors,
        "sector_method": state_meta.get("sector_method", "min"),
        "use_lidar_diff": use_lidar_diff,
        "use_delta_yaw": use_delta_yaw,
    }
    
    env = NavigationEnv(
        use_enhanced_state=True,
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=state_meta.get("dynamic_speed_min", 0.1),
        dynamic_speed_max=state_meta.get("dynamic_speed_max", 0.35),
    )
    
    print(f"Environment: state_dim={env.state_dim}, action_dim={env.action_dim}")
    print(f"History length: {history_len}")
    
    # 创建Agent
    # 检查是否是attention版本
    use_spatial_att = state_meta.get("use_spatial_att", False)
    use_temporal_att = state_meta.get("use_temporal_att", False)
    
    if use_spatial_att or use_temporal_att or "lstm_ddpg_att" in args.model:
        from lstm_ddpg_att import LSTMDdpgAgent
        agent = LSTMDdpgAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            history_len=history_len,
            use_spatial_att=use_spatial_att,
            use_temporal_att=use_temporal_att,
            n_sectors=n_sectors,
            sector_input_dim=2 if use_lidar_diff else 1,
            sector_model_dim=state_meta.get("sector_model_dim", 16),
            spatial_att_heads=state_meta.get("spatial_att_heads", 2),
            temporal_att_dim=state_meta.get("temporal_att_dim", 64),
            temporal_att_heads=state_meta.get("temporal_att_heads", 4),
        )
        print("Using LSTM-DDPG with Attention")
    else:
        from lstm_ddpg import LSTMDdpgAgent
        agent = LSTMDdpgAgent(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            history_len=history_len,
        )
        print("Using LSTM-DDPG")
    
    agent.load(args.model, strict=False, load_optimizers=False)
    print(f"Model loaded from {args.model}")
    
    # 创建可视化器
    visualizer = NavigationVisualizer(env)
    
    # 统计
    success_count = 0
    collision_count = 0
    timeout_count = 0
    total_steps = []
    
    print(f"\nRunning {args.episodes} episodes...")
    print("-" * 60)
    
    try:
        for episode in range(args.episodes):
            np.random.seed(args.seed + episode)
            
            state = env.reset()
            state_queue = deque([state.copy() for _ in range(history_len)], maxlen=history_len)
            
            trajectory = [(env.robot.x, env.robot.y)]
            episode_reward = 0
            
            for step in range(EnvConfig.MAX_STEPS):
                # 选择动作
                state_seq = np.asarray(state_queue, dtype=np.float32)
                action = agent.act(state_seq, step=step, add_noise=False)
                action_flat = action.reshape(-1)
                
                # 执行动作
                next_state, reward, done, info = env.step(action_flat)
                episode_reward += reward
                
                # 更新状态队列
                state_queue.append(next_state.copy())
                state = next_state
                
                # 记录轨迹
                trajectory.append((env.robot.x, env.robot.y))
                
                # 可视化
                visualizer.render(
                    trajectory, 
                    step=step+1, 
                    episode=episode,
                    info=info if done else None,
                    save_frame=args.save_gif
                )
                
                if args.delay > 0:
                    time.sleep(args.delay)
                
                if done:
                    break
            
            # 统计结果
            total_steps.append(step + 1)
            
            if info.get('reason') == 'goal_reached':
                success_count += 1
                result = "SUCCESS"
            elif info.get('reason') in ['collision_obstacle', 'collision_wall']:
                collision_count += 1
                result = "COLLISION"
            else:
                timeout_count += 1
                result = "TIMEOUT"
            
            print(f"Episode {episode+1:3d} | Steps: {step+1:3d} | "
                  f"Reward: {episode_reward:7.1f} | Result: {result}")
            
            # 短暂暂停显示最终状态
            if args.pause_end > 0:
                time.sleep(args.pause_end)
    
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    n_episodes = len(total_steps)
    print(f"Total episodes: {n_episodes}")
    print(f"Success rate: {success_count/n_episodes*100:.1f}% ({success_count}/{n_episodes})")
    print(f"Collision rate: {collision_count/n_episodes*100:.1f}% ({collision_count}/{n_episodes})")
    print(f"Timeout rate: {timeout_count/n_episodes*100:.1f}% ({timeout_count}/{n_episodes})")
    print(f"Average steps: {np.mean(total_steps):.1f}")
    
    # 保存GIF
    if args.save_gif and visualizer.frames:
        gif_path = args.gif_name
        if not gif_path:
            gif_path = os.path.join(RESULT_DIR, f"navigation_test.gif")
        visualizer.save_gif(gif_path, fps=args.gif_fps)
    
    visualizer.close()
    env.close()


def parse_args():
    p = argparse.ArgumentParser(description='Navigation Test with GIF Visualization')
    
    p.add_argument('--model', type=str, 
                   default=os.path.join(MODEL_DIR, 'lstm_ddpg_att_best.pth'),
                   help='Path to model checkpoint')
    p.add_argument('--episodes', type=int, default=5,
                   help='Number of test episodes')
    p.add_argument('--seed', type=int, default=42,
                   help='Random seed')
    
    # 可视化选项
    p.add_argument('--save_gif', action='store_true',
                   help='Save test as GIF animation')
    p.add_argument('--gif_name', type=str, default=None,
                   help='Output GIF filename')
    p.add_argument('--gif_fps', type=int, default=10,
                   help='GIF frame rate')
    p.add_argument('--delay', type=float, default=0.02,
                   help='Delay between frames (seconds)')
    p.add_argument('--pause_end', type=float, default=0.5,
                   help='Pause at end of each episode (seconds)')
    
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test_with_visualization(args)
