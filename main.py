#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG Navigation 主入口
快速启动训练或测试
"""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description='DDPG Navigation for Mobile Robot',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('mode', choices=['train', 'test', 'train_lstm_ddpg', 'test_lstm_ddpg', 'demo'],
                        help='''运行模式:
  train - 训练模型
  test  - 测试已训练的模型
  train_lstm_ddpg - 训练 LSTM-DDPG（序列窗口）
  test_lstm_ddpg  - 测试 LSTM-DDPG
  demo  - 可视化演示（随机动作）''')
    
    parser.add_argument('--episodes', type=int, default=None,
                        help='训练/测试回合数')
    parser.add_argument('--model', type=str, default=None,
                        help='模型路径（用于test模式）')
    parser.add_argument('--render', action='store_true',
                        help='启用可视化')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        from train import train, parse_args as train_parse_args
        train_args = train_parse_args()
        if args.episodes:
            train_args.num_episodes = args.episodes
        if args.render:
            train_args.render = True
        if args.resume:
            train_args.resume = args.resume
        train(train_args)
        
    elif args.mode == 'test':
        from test import test, parse_args as test_parse_args
        test_args = test_parse_args()
        if args.episodes:
            test_args.num_episodes = args.episodes
        if args.model:
            test_args.model = args.model
        if args.render:
            test_args.render = True
        test(test_args)

    elif args.mode == 'train_lstm_ddpg':
        from train_lstm_ddpg import train as train_lstm_ddpg, parse_args as parse_args_lstm
        train_args = parse_args_lstm()
        if args.episodes:
            train_args.num_episodes = args.episodes
        if args.resume:
            train_args.resume = args.resume
        if args.render:
            # 预留参数，占位（当前训练脚本不强依赖 render）
            pass
        train_lstm_ddpg(train_args)

    elif args.mode == 'test_lstm_ddpg':
        from test_lstm_ddpg import test as test_lstm_ddpg, parse_args as parse_args_test_lstm
        test_args = parse_args_test_lstm()
        if args.episodes:
            test_args.num_episodes = args.episodes
        if args.model:
            test_args.model = args.model
        if args.render:
            test_args.render = True
        test_lstm_ddpg(test_args)
        
    elif args.mode == 'demo':
        # 运行可视化演示
        try:
            import pygame
        except ImportError:
            print("Error: pygame is required for demo mode")
            print("Install with: pip install pygame")
            sys.exit(1)
        
        from environment import NavigationEnv
        from visualizer import Visualizer
        from config import EnvConfig
        import numpy as np
        import time
        
        print("=" * 60)
        print("DDPG Navigation Demo")
        print("=" * 60)
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset environment")
        print("  SPACE - Pause/Resume")
        print("=" * 60)
        
        env = NavigationEnv()
        visualizer = Visualizer(env)
        
        state = env.reset()
        trajectory = [(env.robot.x, env.robot.y)]
        paused = False
        running = True
        episode = 0
        step = 0
        total_reward = 0
        
        while running:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_r:
                        episode += 1
                        step = 0
                        total_reward = 0
                        state = env.reset()
                        trajectory = [(env.robot.x, env.robot.y)]
                        print(f"Reset - Episode {episode}")
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("Paused" if paused else "Resumed")
            
            if not paused and running:
                # 随机动作
                action = [
                    np.random.uniform(0.1, EnvConfig.MAX_LINEAR_VEL),
                    np.random.uniform(-1.0, 1.0)
                ]
                
                state, reward, done, info = env.step(action)
                total_reward += reward
                trajectory.append((env.robot.x, env.robot.y))
                step += 1
                
                if done:
                    result = "SUCCESS" if info['reason'] == 'goal_reached' else info['reason'].upper()
                    print(f"Episode {episode}: {result} | Steps: {step} | Reward: {total_reward:.1f}")
                    time.sleep(0.5)
                    episode += 1
                    step = 0
                    total_reward = 0
                    state = env.reset()
                    trajectory = [(env.robot.x, env.robot.y)]
            
            running = visualizer.render(trajectory) and running
        
        visualizer.close()
        print("\nDemo ended")


if __name__ == '__main__':
    main()
