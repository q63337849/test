#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test_single_model.py

快速测试单个模型
"""

import argparse
import torch
import numpy as np
from environment import NavigationEnv
from lstm_ddpg_att import LSTMDdpgAgent

def test_model(model_path: str, episodes: int = 10, 
               speed_min: float = 0.20, speed_max: float = 0.40):
    """测试模型"""
    
    print("=" * 60)
    print(f"测试模型: {model_path}")
    print("=" * 60)
    print(f"测试轮数: {episodes}")
    print(f"速度范围: {speed_min}-{speed_max} m/s")
    print("=" * 60)
    
    # 创建环境
    env = NavigationEnv(
        dynamic_speed_min=speed_min,
        dynamic_speed_max=speed_max,
        n_sectors=16,
        sector_method="min",
    )
    
    # 创建Agent
    agent = LSTMDdpgAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        history_len=5,
    )
    
    # 加载模型
    try:
        agent.load(model_path, load_optimizers=False)
        agent.eval()
        print("\n✅ 模型加载成功")
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return
    
    # 测试
    success_count = 0
    collision_count = 0
    timeout_count = 0
    rewards = []
    
    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            action = agent.act(state, add_noise=False)
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            state = next_state
        
        rewards.append(ep_reward)
        
        if info["success"]:
            success_count += 1
            result = "✅"
        elif info["collision"]:
            collision_count += 1
            result = "❌"
        else:
            timeout_count += 1
            result = "⏱️"
        
        print(f"  Ep {ep+1:2d}: R={ep_reward:7.1f} {result}")
    
    # 结果
    print("\n" + "=" * 60)
    print("测试结果:")
    print("=" * 60)
    print(f"成功率: {success_count/episodes*100:.1f}% ({success_count}/{episodes})")
    print(f"碰撞率: {collision_count/episodes*100:.1f}% ({collision_count}/{episodes})")
    print(f"超时率: {timeout_count/episodes*100:.1f}% ({timeout_count}/{episodes})")
    print(f"平均奖励: {np.mean(rewards):.1f}")
    print("=" * 60)
    
    if success_count >= episodes * 0.5:
        print("\n✅ 模型表现正常！可以继续训练")
    else:
        print("\n⚠️ 成功率较低，建议检查模型")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--dynamic_speed_min", type=float, default=0.20)
    parser.add_argument("--dynamic_speed_max", type=float, default=0.40)
    args = parser.parse_args()
    
    test_model(args.model, args.episodes, 
               args.dynamic_speed_min, args.dynamic_speed_max)
