#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练脚本
基于原项目 start_td3_training.py 的训练流程
python train.py --num_episodes 1000
"""

import os
import time
import argparse
import numpy as np
from datetime import datetime

from config import DDPGConfig, EnvConfig, MODEL_DIR, RESULT_DIR, LOG_DIR
from environment import NavigationEnv
from ddpg import DDPGAgent
from utils import Logger, plot_training_curves


def train(args):
    """主训练函数"""
    
    # ==================== 初始化 ====================
    print("=" * 60)
    print("DDPG Navigation Training")
    print("=" * 60)
    
    # 创建环境
    env = NavigationEnv()
    print(f"Environment created")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Map size: {EnvConfig.MAP_WIDTH} x {EnvConfig.MAP_HEIGHT}")
    print(f"  LiDAR rays: {EnvConfig.LIDAR_RAYS}")
    
    # 创建智能体
    agent = DDPGAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim
    )
    print(f"Agent created")
    print(f"  Hidden dim: {agent.hidden_dim}")
    print(f"  Actor LR: {agent.actor_lr}")
    print(f"  Critic LR: {agent.critic_lr}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Tau: {agent.tau}")
    
    # 恢复训练
    start_episode = 0
    if args.resume:
        if os.path.exists(args.resume):
            agent.load(args.resume)
            # 从文件名提取episode数
            try:
                start_episode = int(args.resume.split('_ep')[-1].split('.')[0])
            except:
                pass
            print(f"Resumed from {args.resume}, starting at episode {start_episode}")
    
    # 创建日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(os.path.join(LOG_DIR, f'training_{timestamp}.csv'))
    
    # ==================== 训练循环 ====================
    total_steps = 0
    best_reward = -float('inf')
    
    # 统计变量
    episode_rewards = []
    episode_steps = []
    success_count = 0
    collision_count = 0
    timeout_count = 0
    
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")
    
    for episode in range(start_episode, args.num_episodes):
        episode_start_time = time.time()
        
        # 重置环境和噪声
        state = env.reset()
        agent.reset_noise()
        
        episode_reward = 0
        
        for step in range(EnvConfig.MAX_STEPS):
            total_steps += 1
            
            # 选择动作
            action = agent.act(state, step=total_steps, add_noise=True)
            action_flat = action.flatten()
            
            # 执行动作
            next_state, reward, done, info = env.step(action_flat)
            
            # 存储经验
            agent.step(state, action, reward, next_state, done)
            
            # 学习
            if total_steps > DDPGConfig.LEARN_START:
                agent.learn()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 回合结束统计
        episode_time = time.time() - episode_start_time
        episode_rewards.append(episode_reward)
        episode_steps.append(step + 1)
        
        success, failure = env.get_episode_status()
        if success:
            success_count += 1
        elif info['reason'] in ['collision_obstacle', 'collision_wall']:
            collision_count += 1
        else:
            timeout_count += 1
        
        # 计算移动平均
        avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
        success_rate = success_count / (episode + 1) * 100
        
        # 记录日志
        logger.log({
            'episode': episode + 1,
            'reward': episode_reward,
            'steps': step + 1,
            'avg_reward': avg_reward,
            'success': int(success),
            'collision': int(info['reason'] in ['collision_obstacle', 'collision_wall']),
            'timeout': int(info['reason'] == 'max_steps'),
            'success_rate': success_rate,
            'total_steps': total_steps,
            'time': episode_time
        })
        
        # 打印进度
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:5d} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Avg: {avg_reward:8.2f} | "
                  f"Steps: {step + 1:4d} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Result: {info['reason']}")
        
        # 保存最佳模型
        if avg_reward > best_reward and episode > 100:
            best_reward = avg_reward
            agent.save(os.path.join(MODEL_DIR, 'ddpg_best.pth'))
            print(f"  -> New best model saved! Avg reward: {best_reward:.2f}")
        
        # 定期保存模型
        if (episode + 1) % args.save_interval == 0:
            agent.save(os.path.join(MODEL_DIR, f'ddpg_ep{episode + 1}.pth'))
    
    # ==================== 训练结束 ====================
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Total episodes: {args.num_episodes}")
    print(f"Total steps: {total_steps}")
    print(f"Success rate: {success_count / args.num_episodes * 100:.1f}%")
    print(f"Collision rate: {collision_count / args.num_episodes * 100:.1f}%")
    print(f"Timeout rate: {timeout_count / args.num_episodes * 100:.1f}%")
    print(f"Best avg reward: {best_reward:.2f}")
    
    # 保存最终模型
    agent.save(os.path.join(MODEL_DIR, 'ddpg_final.pth'))
    
    # 绘制训练曲线
    plot_training_curves(
        episode_rewards,
        episode_steps,
        save_path=os.path.join(RESULT_DIR, f'training_curves_{timestamp}.png')
    )
    
    logger.close()
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description='DDPG Navigation Training')
    parser.add_argument('--num_episodes', type=int, default=DDPGConfig.NUM_EPISODES,
                        help='Number of training episodes')
    parser.add_argument('--save_interval', type=int, default=DDPGConfig.SAVE_INTERVAL,
                        help='Model save interval')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
