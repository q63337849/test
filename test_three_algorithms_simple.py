#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_three_algorithms_simple.py

简化版三算法对比测试，避免模型检查问题
"""

import argparse
import numpy as np
import torch
from collections import deque

from environment import NavigationEnv
from ddpg import DDPGAgent
from lstm_ddpg import LSTMDdpgAgent
from lstm_ddpg_att import LSTMDdpgAgent as LSTMDdpgAttAgent


def test_model(agent, env, episodes, history_len=1, name="Model"):
    """测试单个模型"""
    print(f"\n{'=' * 70}")
    print(f"测试 {name}")
    print('=' * 70)
    
    successes = 0
    collisions = 0
    timeouts = 0
    episode_rewards = []
    episode_steps = []
    
    for ep in range(episodes):
        state = env.reset()
        
        # 初始化历史队列
        if history_len > 1:
            state_queue = deque([state.copy() for _ in range(history_len)], maxlen=history_len)
        
        episode_reward = 0.0
        step = 0
        
        while step < 500:  # 最大步数
            # 准备状态
            if history_len > 1:
                state_seq = np.asarray(state_queue, dtype=np.float32)
                action = agent.act(state_seq, add_noise=False)
            else:
                action = agent.act(state, add_noise=False)
            
            # 执行动作
            next_state, reward, done, info = env.step(action.reshape(-1))
            
            episode_reward += reward
            state = next_state
            
            if history_len > 1:
                state_queue.append(next_state.copy())
            
            step += 1
            
            if done:
                break
        
        # 记录结果
        episode_rewards.append(episode_reward)
        episode_steps.append(step)
        
        success, _ = env.get_episode_status()
        if success:
            successes += 1
        elif info.get("reason") in ("collision_obstacle", "collision_wall"):
            collisions += 1
        else:
            timeouts += 1
        
        # 显示进度
        if (ep + 1) % 10 == 0:
            sr = successes / (ep + 1) * 100
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"  Ep {ep+1:3d}/{episodes} | SR: {sr:5.1f}% | Avg Reward: {avg_reward:7.1f}")
    
    # 最终统计
    success_rate = successes / episodes * 100
    collision_rate = collisions / episodes * 100
    timeout_rate = timeouts / episodes * 100
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    print(f"\n{name} 测试结果:")
    print(f"  成功率: {success_rate:.1f}% ({successes}/{episodes})")
    print(f"  碰撞率: {collision_rate:.1f}% ({collisions}/{episodes})")
    print(f"  超时率: {timeout_rate:.1f}% ({timeouts}/{episodes})")
    print(f"  平均奖励: {avg_reward:.2f}")
    print(f"  平均步数: {avg_steps:.1f}")
    
    return {
        'name': name,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'successes': successes,
        'total': episodes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ddpg_model', type=str, required=True)
    parser.add_argument('--lstm_model', type=str, required=True)
    parser.add_argument('--att_model', type=str, required=True)
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--dynamic_speed_min', type=float, default=0.50)
    parser.add_argument('--dynamic_speed_max', type=float, default=0.70)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("=" * 70)
    print("三算法对比测试 - 简化版")
    print("=" * 70)
    print(f"测试轮数: {args.episodes}")
    print(f"速度范围: {args.dynamic_speed_min:.2f}-{args.dynamic_speed_max:.2f} m/s")
    print(f"随机种子: {args.seed}")
    
    # 创建环境
    env_config = {
        'use_enhanced_state': True,
        'enhanced_state_config': {
            'n_sectors': 16,
            'sector_method': 'min',
            'use_lidar_diff': True,
            'use_delta_yaw': True,
        },
        'dynamic_speed_min': args.dynamic_speed_min,
        'dynamic_speed_max': args.dynamic_speed_max,
        'dynamic_patterns': ('bounce', 'random_walk'),
        'dynamic_stop_prob': 0.05,
    }
    
    env = NavigationEnv(**env_config)
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print(f"\n环境配置:")
    print(f"  State dim: {state_dim}")
    print(f"  Action dim: {action_dim}")
    
    results = []
    
    # ================================================================
    # 测试1: DDPG
    # ================================================================
    try:
        print("\n正在加载 DDPG 模型...")
        ddpg_agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim)
        ddpg_agent.load(args.ddpg_model, strict=False)
        
        result = test_model(ddpg_agent, env, args.episodes, history_len=1, name="DDPG")
        results.append(result)
        
    except Exception as e:
        print(f"❌ DDPG测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ================================================================
    # 测试2: LSTM-DDPG V6
    # ================================================================
    try:
        print("\n正在加载 LSTM-DDPG V6 模型...")
        lstm_agent = LSTMDdpgAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            history_len=5,
            embed_dim=64,
            lstm_hidden_dim=64,
        )
        lstm_agent.load(args.lstm_model, strict=False)
        
        result = test_model(lstm_agent, env, args.episodes, history_len=5, name="LSTM-DDPG V6")
        results.append(result)
        
    except Exception as e:
        print(f"❌ LSTM-DDPG测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ================================================================
    # 测试3: LSTM-DDPG + Attention
    # ================================================================
    try:
        print("\n正在加载 LSTM-DDPG + Attention 模型...")
        att_agent = LSTMDdpgAttAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            history_len=5,
            embed_dim=64,
            lstm_hidden_dim=64,
            use_spatial_att=True,
            use_temporal_att=True,
            sector_model_dim=32,
            spatial_att_heads=4,
            temporal_att_dim=64,
            temporal_att_heads=4,
        )
        att_agent.load(args.att_model, strict=False)
        
        result = test_model(att_agent, env, args.episodes, history_len=5, name="LSTM-DDPG-ATT")
        results.append(result)
        
    except Exception as e:
        print(f"❌ LSTM-DDPG-ATT测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # ================================================================
    # 最终对比
    # ================================================================
    if len(results) >= 2:
        print("\n" + "=" * 70)
        print("最终对比结果")
        print("=" * 70)
        
        for result in results:
            print(f"\n{result['name']}:")
            print(f"  成功率: {result['success_rate']:.1f}%")
            print(f"  平均奖励: {result['avg_reward']:.2f}")
        
        # 计算提升
        if len(results) >= 3:
            ddpg_sr = results[0]['success_rate']
            lstm_sr = results[1]['success_rate']
            att_sr = results[2]['success_rate']
            
            print(f"\n性能提升:")
            print(f"  LSTM-DDPG-ATT vs DDPG:      +{att_sr - ddpg_sr:.1f}%")
            print(f"  LSTM-DDPG-ATT vs LSTM-DDPG: +{att_sr - lstm_sr:.1f}%")
    
    env.close()
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()
