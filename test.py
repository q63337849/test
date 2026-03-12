#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试脚本 - 支持 DDPG 和 LSTM-DDPG
"""

import os
import argparse
import numpy as np
import time
from collections import deque

from config import EnvConfig, MODEL_DIR
from environment import NavigationEnv


def test(args):
    """测试函数"""
    
    print("=" * 60)
    print(f"{'LSTM-DDPG' if args.lstm else 'DDPG'} Navigation Testing")
    print("=" * 60)
    
    # 环境配置
    enhanced_cfg = {
        "n_sectors": args.n_sectors,
        "sector_method": args.sector_method,
        "use_lidar_diff": (not args.disable_lidar_diff),
        "use_delta_yaw": (not args.disable_delta_yaw),
    }
    
    env = NavigationEnv(
        use_enhanced_state=(not args.legacy_state),
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=args.dynamic_speed_min,
        dynamic_speed_max=args.dynamic_speed_max,
    )
    
    print(f"Environment: state_dim={env.state_dim}, action_dim={env.action_dim}")
    
    # 创建智能体
    if args.lstm:
        from lstm_ddpg import LSTMDdpgAgent
        agent = LSTMDdpgAgent(
            state_dim=env.state_dim, 
            action_dim=env.action_dim,
            history_len=args.history_len,
        )
    else:
        from ddpg import DDPGAgent
        agent = DDPGAgent(state_dim=env.state_dim, action_dim=env.action_dim)
    
    # 加载模型
    if not os.path.exists(args.model):
        print(f"Error: Model not found at {args.model}")
        return
    
    agent.load(args.model)
    print(f"Model loaded from {args.model}")
    
    # 创建可视化器
    visualizer = None
    if args.render:
        try:
            from visualizer import Visualizer
            visualizer = Visualizer(env)
        except ImportError:
            print("Warning: Visualizer not available")
    
    # 统计变量
    total_rewards = []
    total_steps = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    print(f"\nRunning {args.num_episodes} test episodes...")
    print("-" * 60)
    
    for episode in range(args.num_episodes):
        state = env.reset()
        episode_reward = 0
        trajectory = [(env.robot.x, env.robot.y)]
        
        # LSTM 需要状态队列
        if args.lstm:
            state_queue = deque([state.copy() for _ in range(args.history_len)], maxlen=args.history_len)
        
        for step in range(EnvConfig.MAX_STEPS):
            # 选择动作（不加噪声）
            if args.lstm:
                state_seq = np.asarray(state_queue, dtype=np.float32)
                action = agent.act(state_seq, add_noise=False)
            else:
                action = agent.act(state, add_noise=False)
            
            action_flat = action.flatten()
            
            # 执行动作
            next_state, reward, done, info = env.step(action_flat)
            episode_reward += reward
            
            # 更新状态
            if args.lstm:
                state_queue.append(next_state.copy())
            state = next_state
            
            # 记录轨迹
            trajectory.append((env.robot.x, env.robot.y))
            
            # 可视化
            if visualizer:
                visualizer.render(trajectory)
                time.sleep(0.02)
            
            if done:
                break
        
        # 统计
        total_rewards.append(episode_reward)
        total_steps.append(step + 1)
        
        success, failure = env.get_episode_status()
        if success:
            success_count += 1
            result = "SUCCESS"
        elif info['reason'] in ['collision_obstacle', 'collision_wall']:
            collision_count += 1
            result = "COLLISION"
        else:
            timeout_count += 1
            result = "TIMEOUT"
        
        print(f"Episode {episode + 1:3d} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Steps: {step + 1:4d} | "
              f"Result: {result}")
    
    # 打印统计结果
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Average reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"Average steps: {np.mean(total_steps):.2f}")
    print(f"Success rate: {success_count / args.num_episodes * 100:.1f}%")
    print(f"Collision rate: {collision_count / args.num_episodes * 100:.1f}%")
    print(f"Timeout rate: {timeout_count / args.num_episodes * 100:.1f}%")
    
    if visualizer:
        visualizer.close()
    
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Navigation Testing')
    parser.add_argument('--model', type=str, 
                        default=os.path.join(MODEL_DIR, 'ddpg_best.pth'),
                        help='Path to model checkpoint')
    parser.add_argument('--num_episodes', type=int, default=100,
                        help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during testing')
    
    # 算法选择
    parser.add_argument('--lstm', action='store_true',
                        help='Use LSTM-DDPG instead of DDPG')
    parser.add_argument('--history_len', type=int, default=5,
                        help='History length for LSTM')
    
    # 环境配置
    parser.add_argument("--legacy_state", action="store_true")
    parser.add_argument("--n_sectors", type=int, default=16)
    parser.add_argument("--sector_method", type=str, default="min")
    parser.add_argument("--disable_lidar_diff", action="store_true")
    parser.add_argument("--disable_delta_yaw", action="store_true")
    parser.add_argument("--dynamic_speed_min", type=float, default=0.30)
    parser.add_argument("--dynamic_speed_max", type=float, default=0.70)
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    test(args)
