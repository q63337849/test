#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_v6_model.py - V6模型测试脚本

用法:
    python test_v6_model.py --model models/lstm_ddpg_v6_final_*.pth --episodes 100
    python test_v6_model.py --model models/lstm_ddpg_v6_stage7_checkpoint.pth --episodes 50 --render
"""

import argparse
import numpy as np
from collections import deque
from typing import Dict

from environment import NavigationEnv
from lstm_ddpg import LSTMDdpgAgent
from config import EnvConfig


def test_model(model_path: str, n_episodes: int = 100, render: bool = False, 
               enhanced_state: bool = True) -> Dict:
    """测试V6模型性能"""
    
    print("=" * 80)
    print("V6 模型测试")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"测试轮数: {n_episodes}")
    print(f"测试场景: 4静态 + 2动态 @ 100%速度")
    print(f"动态障碍物速度: {EnvConfig.DYNAMIC_OBS_VEL_MIN:.2f}-{EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f} m/s")
    print(f"无人机最大速度: {EnvConfig.MAX_LINEAR_VEL:.2f} m/s")
    print(f"速度比: {EnvConfig.MAX_LINEAR_VEL/EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f}:1")
    print("=" * 80)
    
    # 设置最难环境
    EnvConfig.NUM_STATIC_OBSTACLES = 4
    EnvConfig.NUM_DYNAMIC_OBSTACLES = 2
    
    # 创建环境（从config.py读取动态障碍物速度）
    env = NavigationEnv(
        use_enhanced_state=enhanced_state,
        render_mode='human' if render else None,
        dynamic_speed_min=EnvConfig.DYNAMIC_OBS_VEL_MIN,  # ← 从config.py读取
        dynamic_speed_max=EnvConfig.DYNAMIC_OBS_VEL_MAX,  # ← 从config.py读取
    )
    
    # 加载模型
    agent = LSTMDdpgAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        history_len=5,  # V6默认使用5
    )
    
    try:
        agent.load(model_path)
        print(f"✓ 模型加载成功\n")
    except Exception as e:
        print(f"✗ 模型加载失败: {e}")
        return {}
    
    # 统计数据
    successes = 0
    collisions = 0
    timeouts = 0
    
    episode_rewards = []
    episode_steps = []
    episode_results = []
    
    # 测试循环
    for ep in range(n_episodes):
        state = env.reset()
        state_queue = deque([state.copy() for _ in range(5)], maxlen=5)
        done = False
        
        episode_reward = 0.0
        step_count = 0
        
        while not done:
            state_seq = np.asarray(state_queue, dtype=np.float32)
            action = agent.act(state_seq, step=0, add_noise=False)  # 测试时不加噪声
            action_flat = action.reshape(-1)
            
            next_state, reward, done, info = env.step(action_flat)
            state_queue.append(next_state.copy())
            
            episode_reward += reward
            step_count += 1
            
            if done:
                break
        
        # 统计结果
        reason = info.get('reason', 'unknown')
        if reason == 'goal_reached':
            successes += 1
            result = 'S'
        elif reason in ('collision_obstacle', 'collision_wall'):
            collisions += 1
            result = 'C'
        else:
            timeouts += 1
            result = 'T'
        
        episode_rewards.append(episode_reward)
        episode_steps.append(step_count)
        episode_results.append(result)
        
        # 周期性输出
        if (ep + 1) % 10 == 0:
            current_success = successes / (ep + 1)
            current_collision = collisions / (ep + 1)
            current_timeout = timeouts / (ep + 1)
            
            print(f"[Episode {ep+1:3d}/{n_episodes}] "
                  f"S:{successes:3d}/{ep+1} C:{collisions:3d}/{ep+1} T:{timeouts:3d}/{ep+1} | "
                  f"SR:{current_success:.2%} CR:{current_collision:.2%} TR:{current_timeout:.2%}")
    
    # 计算统计
    success_rate = successes / n_episodes
    collision_rate = collisions / n_episodes
    timeout_rate = timeouts / n_episodes
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    print(f"场景: 4静态 + 2动态 @ 100%速度 | 测试回合数: {n_episodes}")
    print(f"动态障碍物速度: {EnvConfig.DYNAMIC_OBS_VEL_MIN:.2f}-{EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f} m/s")
    print(f"无人机速度: {EnvConfig.MAX_LINEAR_VEL:.2f} m/s | 速度比: {EnvConfig.MAX_LINEAR_VEL/EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f}:1")
    print(f"成功率: {success_rate:.2%} ({successes}/{n_episodes})")
    print(f"碰撞率: {collision_rate:.2%} ({collisions}/{n_episodes})")
    print(f"超时率: {timeout_rate:.2%} ({timeouts}/{n_episodes})")
    print(f"平均奖励: {avg_reward:.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均步数: {avg_steps:.2f} ± {np.std(episode_steps):.2f}")
    print("=" * 80)
    
    # 结果分布
    print("\n结果分布 (最近100轮):")
    recent_results = ''.join(episode_results[-100:])
    print(f"  {recent_results}")
    print(f"  S=成功, C=碰撞, T=超时")
    
    # 性能评估
    print("\n性能评估:")
    if success_rate >= 0.65:
        print(f"  ✓ 优秀 (≥65%): {success_rate:.2%}")
    elif success_rate >= 0.60:
        print(f"  ✓ 良好 (≥60%): {success_rate:.2%}")
    elif success_rate >= 0.55:
        print(f"  ~ 中等 (≥55%): {success_rate:.2%}")
    elif success_rate >= 0.50:
        print(f"  ~ 及格 (≥50%): {success_rate:.2%}")
    else:
        print(f"  ✗ 需要改进 (<50%): {success_rate:.2%}")
    
    if collision_rate <= 0.20:
        print(f"  ✓ 碰撞率优秀 (≤20%): {collision_rate:.2%}")
    elif collision_rate <= 0.30:
        print(f"  ✓ 碰撞率良好 (≤30%): {collision_rate:.2%}")
    elif collision_rate <= 0.35:
        print(f"  ~ 碰撞率可接受 (≤35%): {collision_rate:.2%}")
    else:
        print(f"  ✗ 碰撞率偏高 (>{35}%): {collision_rate:.2%}")
    
    env.close()
    
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_results': episode_results,
    }


def compare_models(model1_path: str, model2_path: str, n_episodes: int = 100) -> None:
    """对比两个模型的性能"""
    
    print("\n" + "=" * 80)
    print("模型对比测试")
    print("=" * 80)
    print(f"模型1: {model1_path}")
    print(f"模型2: {model2_path}")
    print("=" * 80 + "\n")
    
    print("测试模型1...")
    results1 = test_model(model1_path, n_episodes, render=False)
    
    print("\n" + "-" * 80 + "\n")
    
    print("测试模型2...")
    results2 = test_model(model2_path, n_episodes, render=False)
    
    # 对比结果
    print("\n" + "=" * 80)
    print("对比结果")
    print("=" * 80)
    print(f"{'指标':<20} | {'模型1':>15} | {'模型2':>15} | {'差异':>15}")
    print("-" * 80)
    
    sr1, sr2 = results1['success_rate'], results2['success_rate']
    print(f"{'成功率':<20} | {sr1:>14.2%} | {sr2:>14.2%} | {(sr2-sr1):>+14.2%}")
    
    cr1, cr2 = results1['collision_rate'], results2['collision_rate']
    print(f"{'碰撞率':<20} | {cr1:>14.2%} | {cr2:>14.2%} | {(cr2-cr1):>+14.2%}")
    
    tr1, tr2 = results1['timeout_rate'], results2['timeout_rate']
    print(f"{'超时率':<20} | {tr1:>14.2%} | {tr2:>14.2%} | {(tr2-tr1):>+14.2%}")
    
    ar1, ar2 = results1['avg_reward'], results2['avg_reward']
    print(f"{'平均奖励':<20} | {ar1:>14.2f} | {ar2:>14.2f} | {(ar2-ar1):>+14.2f}")
    
    as1, as2 = results1['avg_steps'], results2['avg_steps']
    print(f"{'平均步数':<20} | {as1:>14.2f} | {as2:>14.2f} | {(as2-as1):>+14.2f}")
    
    print("=" * 80)
    
    # 判断优劣
    if sr2 > sr1:
        improvement = (sr2 - sr1) / max(sr1, 0.01) * 100
        print(f"\n✓ 模型2更优: 成功率提升 {improvement:.1f}%")
    elif sr2 < sr1:
        decline = (sr1 - sr2) / max(sr1, 0.01) * 100
        print(f"\n✗ 模型2下降: 成功率下降 {decline:.1f}%")
    else:
        print(f"\n~ 两模型性能相当")


def main():
    parser = argparse.ArgumentParser(description="V6 Model Testing")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of test episodes (default: 100)")
    parser.add_argument("--render", action="store_true",
                       help="Enable rendering")
    parser.add_argument("--compare", type=str, default=None,
                       help="Path to second model for comparison")
    parser.add_argument("--legacy_state", action="store_true",
                       help="Use legacy state representation")
    
    args = parser.parse_args()
    
    if args.compare:
        # 对比模式
        compare_models(args.model, args.compare, args.episodes)
    else:
        # 单模型测试
        test_model(
            args.model, 
            args.episodes, 
            args.render,
            enhanced_state=(not args.legacy_state)
        )


if __name__ == "__main__":
    main()
