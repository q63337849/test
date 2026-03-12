"""
train_lstm_v6_comprehensive.py
LSTM-DDPG 综合优化版本

核心改进:
1. ✅ 增加Stage 7训练量: 800 → 1500 episodes
2. ✅ 添加Stage 6.5: 90%速度过渡阶段
3. ✅ 动态success_bonus: 根据难度调整 200/150/100
4. ✅ 自适应探索率衰减
5. ✅ 防御性评估机制
6. ✅ 早停策略

预期提升: 51% → 65-70%
训练时间: +3-4小时
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random
import time
from tqdm import tqdm

# 假设你的环境和Agent类已经定义
# from your_env import Sim2RealEnv, LiDARVelocityWrapper
# from your_agent import LSTMDDPGAgent
from lstm_ddpg import LSTMDdpgAgent

class AdaptiveRewardShapingWrapper:
    """
    自适应RewardShaping包装器
    根据训练阶段动态调整success_bonus
    """
    def __init__(self, env, stage_type='navigation', success_bonus=None):
        self.env = env
        self.stage_type = stage_type
        
        # 🆕 动态success_bonus策略
        if success_bonus is None:
            if stage_type == 'navigation':
                self.success_bonus = 200.0  # Stage 1-2: 高奖励鼓励探索
            elif stage_type in ['sparse', 'medium']:
                self.success_bonus = 150.0  # Stage 3-5: 中等奖励
            elif stage_type == 'dense':
                self.success_bonus = 100.0  # Stage 6-7: 低奖励关注安全
            else:
                self.success_bonus = 120.0  # 默认
        else:
            self.success_bonus = success_bonus
        
        print(f"  [RewardShaping] stage_type={stage_type}, success_bonus={self.success_bonus:.1f}")
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        
        # 成功时添加额外奖励
        if done and info.get('win', False):
            reward += self.success_bonus
        
        return obs, reward, done, info
    
    def close(self):
        self.env.close()
    
    def __getattr__(self, name):
        return getattr(self.env, name)


class DefensiveEvaluator:
    """
    防御性评估器 - 检测灾难性遗忘
    """
    def __init__(self, eval_env_config):
        self.eval_env_config = eval_env_config
        self.history = []
    
    def evaluate(self, agent, n_episodes=20):
        """评估Agent在简单场景上的表现"""
        from your_env import Sim2RealEnv, LiDARVelocityWrapper
        
        # 创建评估环境 (Scene 1, 无障碍)
        eval_env = Sim2RealEnv(
            scene_id=self.eval_env_config['scene'],
            max_steps=400,
            num_static=0,
            num_dynamic=0,
            num_lidar_beams=16,
        )
        eval_env = LiDARVelocityWrapper(eval_env)
        
        successes = []
        for _ in range(n_episodes):
            obs = eval_env.reset()
            hidden = None
            done = False
            
            while not done:
                with torch.no_grad():
                    action, hidden = agent.select_action(obs, hidden, evaluate=True)
                obs, _, done, info = eval_env.step(action)
            
            successes.append(1 if info.get('win', False) else 0)
        
        eval_env.close()
        
        success_rate = np.mean(successes)
        self.history.append(success_rate)
        
        return success_rate
    
    def check_forgetting(self, current_rate, threshold=0.90):
        """检查是否发生灾难性遗忘"""
        if len(self.history) < 2:
            return False
        
        # 如果当前成功率低于阈值，且比之前下降超过10%
        if current_rate < threshold:
            prev_rate = self.history[-2]
            if current_rate < prev_rate - 0.10:
                return True
        
        return False


def train_stage_comprehensive(env, agent, num_episodes, updates_per_ep, 
                              eval_interval, stage_name, stage_config,
                              defensive_evaluator=None):
    """
    综合优化的训练函数
    
    新增功能:
    - 自适应探索率衰减
    - 防御性评估
    - 早停策略
    """
    rewards = []
    success_rates = []
    collision_rates = []
    
    # 🆕 早停参数
    early_stop_threshold = stage_config.get('early_stop_threshold', None)
    early_stop_patience = stage_config.get('early_stop_patience', 200)
    best_success_rate = 0.0
    patience_counter = 0
    
    # 🆕 自适应探索率衰减参数
    adaptive_exploration = stage_config.get('adaptive_exploration', False)
    exploration_decay_start = stage_config.get('exploration_decay_start', 500)
    exploration_decay_rate = stage_config.get('exploration_decay_rate', 0.995)
    min_noise_std = stage_config.get('min_noise_std', 0.05)
    
    episode_buffer = deque(maxlen=100)
    
    pbar = tqdm(range(num_episodes), desc=f"{stage_name}")
    
    for episode in pbar:
        obs = env.reset()
        hidden = None
        episode_reward = 0
        done = False
        
        while not done:
            action, hidden = agent.select_action(obs, hidden, evaluate=False)
            next_obs, reward, done, info = env.step(action)
            
            agent.store_transition(obs, action, reward, next_obs, done, hidden)
            
            obs = next_obs
            episode_reward += reward
        
        rewards.append(episode_reward)
        win = info.get('win', False)
        collision = info.get('collision', False)
        
        episode_buffer.append({'win': win, 'collision': collision})
        
        # 训练网络
        for _ in range(updates_per_ep):
            agent.update()
        
        # 🆕 自适应探索率衰减
        if adaptive_exploration and episode > exploration_decay_start:
            old_noise = agent.noise_std
            agent.noise_std = max(min_noise_std, agent.noise_std * exploration_decay_rate)
            if episode % 100 == 0 and old_noise != agent.noise_std:
                pbar.write(f"  [探索衰减] ε: {old_noise:.4f} → {agent.noise_std:.4f}")
        
        # 周期性统计
        if (episode + 1) % eval_interval == 0:
            recent_success = np.mean([ep['win'] for ep in episode_buffer])
            recent_collision = np.mean([ep['collision'] for ep in episode_buffer])
            
            success_rates.append(recent_success)
            collision_rates.append(recent_collision)
            
            pbar.write(f"[{stage_name} | Ep {episode+1:4d}] "
                      f"Success: {recent_success:.2%} | "
                      f"Collision: {recent_collision:.2%} | "
                      f"AvgRet: {np.mean(list(rewards)[-100:]):.1f}")
            
            # 🆕 早停检查
            if early_stop_threshold is not None:
                if recent_success > best_success_rate:
                    best_success_rate = recent_success
                    patience_counter = 0
                else:
                    patience_counter += eval_interval
                
                if recent_success >= early_stop_threshold:
                    pbar.write(f"✅ 提前达到目标 {recent_success:.2%} >= {early_stop_threshold:.2%}, 停止训练")
                    break
                
                if patience_counter >= early_stop_patience:
                    pbar.write(f"⚠️ {early_stop_patience}轮无改善，提前停止")
                    break
        
        # 🆕 防御性评估 (每500轮)
        if defensive_evaluator is not None and (episode + 1) % 500 == 0:
            pbar.write("  [防御性评估] 检查Stage 1性能...")
            stage1_success = defensive_evaluator.evaluate(agent, n_episodes=20)
            pbar.write(f"  Stage 1 成功率: {stage1_success:.2%}")
            
            if defensive_evaluator.check_forgetting(stage1_success, threshold=0.90):
                pbar.write("  ⚠️ 检测到灾难性遗忘！建议检查训练")
                # 可选：这里可以添加回滚逻辑
    
    return {
        'rewards': rewards,
        'success': success_rates,
        'collision': collision_rates
    }


def train_comprehensive_curriculum(agent, eval_interval=50):
    """
    V6综合优化课程
    
    关键改进:
    1. Stage 6.5: 新增90%速度过渡
    2. Stage 7: 训练量 800→1500
    3. 动态success_bonus
    4. 自适应探索率
    5. 防御性评估
    6. 早停策略
    """
    from your_env import Sim2RealEnv, LiDARVelocityWrapper
    
    # 🆕 定义课程 - 增强版
    curriculum = [
        # Stage 1-2: 基础导航 (不使用RewardShaping)
        {'name': 'Stage 1: Navigation', 
         'episodes': 800,
         'updates_per_ep': 20,
         'scene': 1, 
         'num_static': 0, 
         'num_dynamic': 0, 
         'speed_scale': 1.0,
         'use_reward_shaping': False,
         'stage_type': 'navigation',
         'explore_reset': 0.25,
         'adaptive_exploration': False},
        
        {'name': 'Stage 2: Static Obstacles', 
         'episodes': 800,
         'updates_per_ep': 20,
         'scene': 3, 
         'num_static': 3, 
         'num_dynamic': 0, 
         'speed_scale': 1.0,
         'use_reward_shaping': False,
         'stage_type': 'navigation',
         'explore_reset': 0.25,
         'adaptive_exploration': False},
        
        # Stage 3-5: 动态障碍渐进 (使用RewardShaping)
        {'name': 'Stage 3: Dynamic@20%', 
         'episodes': 800,
         'updates_per_ep': 20,
         'scene': 3, 
         'num_static': 2, 
         'num_dynamic': 2, 
         'speed_scale': 0.20,
         'use_reward_shaping': True,
         'stage_type': 'sparse',
         'success_bonus': 150.0,
         'explore_reset': 0.20,
         'adaptive_exploration': False},
        
        {'name': 'Stage 4: Dynamic@40%', 
         'episodes': 800,
         'updates_per_ep': 20,
         'scene': 3, 
         'num_static': 2, 
         'num_dynamic': 2, 
         'speed_scale': 0.40,
         'use_reward_shaping': True,
         'stage_type': 'medium',
         'success_bonus': 150.0,
         'explore_reset': 0.18,
         'adaptive_exploration': False},
        
        {'name': 'Stage 5: Dynamic@60%', 
         'episodes': 800,
         'updates_per_ep': 20,
         'scene': 3, 
         'num_static': 2, 
         'num_dynamic': 2, 
         'speed_scale': 0.60,
         'use_reward_shaping': True,
         'stage_type': 'medium',
         'success_bonus': 150.0,
         'explore_reset': 0.16,
         'adaptive_exploration': False},
        
        # Stage 6: 密集障碍@80%
        {'name': 'Stage 6: Dense@80%', 
         'episodes': 900,  # 🆕 从800增加到900
         'updates_per_ep': 20,
         'scene': 3, 
         'num_static': 3, 
         'num_dynamic': 2, 
         'speed_scale': 0.80,
         'use_reward_shaping': True,
         'stage_type': 'dense',
         'success_bonus': 120.0,  # 🆕 降低success_bonus
         'explore_reset': 0.15,
         'adaptive_exploration': False},
        
        # 🆕 Stage 6.5: 90%速度过渡阶段 (新增!)
        {'name': 'Stage 6.5: Transition@90%', 
         'episodes': 600,
         'updates_per_ep': 20,
         'scene': 4,  # 🔥 提前适应Scene 4
         'num_static': 4, 
         'num_dynamic': 2, 
         'speed_scale': 0.90,  # 🔥 90%速度过渡
         'use_reward_shaping': True,
         'stage_type': 'dense',
         'success_bonus': 100.0,  # 🆕 进一步降低
         'explore_reset': 0.12,
         'adaptive_exploration': True,  # 🆕 启用自适应探索
         'exploration_decay_start': 300,
         'exploration_decay_rate': 0.995,
         'min_noise_std': 0.05},
        
        # 🆕 Stage 7: 最终挑战 (大幅增强!)
        {'name': 'Stage 7: Full Challenge@100%', 
         'episodes': 1500,  # 🔥 从800增加到1500
         'updates_per_ep': 20,
         'scene': 4, 
         'num_static': 4, 
         'num_dynamic': 2, 
         'speed_scale': 1.0,
         'use_reward_shaping': True,
         'stage_type': 'dense',
         'success_bonus': 100.0,  # 🆕 最低success_bonus, 关注安全
         'explore_reset': 0.10,
         'adaptive_exploration': True,  # 🆕 启用自适应探索
         'exploration_decay_start': 500,
         'exploration_decay_rate': 0.995,
         'min_noise_std': 0.05,
         'early_stop_threshold': 0.65,  # 🆕 达到65%自动停止
         'early_stop_patience': 300},    # 🆕 300轮无改善则停止
    ]
    
    # 🆕 创建防御性评估器
    defensive_evaluator = DefensiveEvaluator(
        eval_env_config={'scene': 1}
    )
    
    all_results = {'rewards': [], 'success': [], 'collision': []}
    stage_final_results = []
    
    for stage_idx, stage in enumerate(curriculum):
        print("\n" + "=" * 80)
        print(f"🎯 {stage['name']} ({stage['episodes']} episodes)")
        print(f"   Scene: {stage['scene']}, Static: {stage['num_static']}, "
              f"Dynamic: {stage['num_dynamic']}, Speed: {stage['speed_scale']:.0%}")
        print(f"   RewardShaping: {stage['use_reward_shaping']}, "
              f"Stage_type: {stage.get('stage_type', 'N/A')}")
        if stage['use_reward_shaping']:
            bonus = stage.get('success_bonus', 'auto')
            print(f"   Success_bonus: {bonus}")
        if stage.get('adaptive_exploration', False):
            print(f"   自适应探索: ON (衰减率={stage['exploration_decay_rate']})")
        if stage.get('early_stop_threshold', None):
            print(f"   早停策略: {stage['early_stop_threshold']:.0%}")
        print("=" * 80)
        
        # 创建环境
        base_env = Sim2RealEnv(
            scene_id=stage['scene'],
            max_steps=400,
            num_static=stage['num_static'],
            num_dynamic=stage['num_dynamic'],
            num_lidar_beams=16,
        )
        
        env = LiDARVelocityWrapper(base_env)
        
        # 应用RewardShaping
        if stage['use_reward_shaping']:
            stage_type = stage.get('stage_type', 'navigation')
            success_bonus = stage.get('success_bonus', None)
            env = AdaptiveRewardShapingWrapper(
                env, 
                stage_type=stage_type,
                success_bonus=success_bonus
            )
        
        # 应用速度缩放
        if stage['speed_scale'] != 1.0:
            from your_env import DynamicCurriculumWrapper
            env = DynamicCurriculumWrapper(env, initial_speed_scale=stage['speed_scale'])
        
        # 重置探索率
        explore_reset = stage['explore_reset']
        agent.reset_exploration(noise_std=explore_reset, epsilon=explore_reset)
        print(f"  [探索重置: ε={explore_reset:.2f}]")
        
        # 🆕 训练 - 传入stage_config和defensive_evaluator
        results = train_stage_comprehensive(
            env, agent, 
            num_episodes=stage['episodes'],
            updates_per_ep=stage['updates_per_ep'],
            eval_interval=eval_interval, 
            stage_name=stage['name'],
            stage_config=stage,
            defensive_evaluator=defensive_evaluator if stage_idx >= 2 else None  # Stage 3+才启用
        )
        
        all_results['rewards'].extend(results['rewards'])
        all_results['success'].extend(results['success'])
        all_results['collision'].extend(results['collision'])
        
        # 阶段总结
        last_n = min(100, len(results['success']))
        final_success = np.mean(results['success'][-last_n:]) if len(results['success']) > 0 else 0.0
        final_collision = np.mean(results['collision'][-last_n:]) if len(results['collision']) > 0 else 0.0
        
        stage_final_results.append({
            'stage': stage['name'],
            'success': final_success,
            'collision': final_collision
        })
        
        print(f"\n{stage['name']} 完成:")
        print(f"  最终成功率: {final_success:.2%}")
        print(f"  最终碰撞率: {final_collision:.2%}")
        
        # 🆕 保存阶段检查点
        checkpoint_path = f"lstm_ddpg_stage{stage_idx+1}_checkpoint.pth"
        torch.save({
            'actor_state_dict': agent.actor.state_dict(),
            'critic_state_dict': agent.critic.state_dict(),
            'stage': stage['name'],
            'final_success': final_success,
        }, checkpoint_path)
        print(f"  ✅ 检查点已保存: {checkpoint_path}")
        
        env.close()
    
    # 🆕 打印最终总结
    print("\n" + "=" * 80)
    print("🎉 全部训练完成！阶段总结:")
    print("=" * 80)
    for result in stage_final_results:
        print(f"{result['stage']:30s} | Success: {result['success']:6.2%} | Collision: {result['collision']:6.2%}")
    print("=" * 80)
    
    return agent, all_results, stage_final_results


def plot_comprehensive_results(results, stage_final_results, save_path='v6_comprehensive_results.png'):
    """
    绘制综合训练结果
    """
    fig = plt.figure(figsize=(18, 10))
    
    # 布局: 2行3列
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    window = 50
    
    # 1. Reward曲线
    ax1 = fig.add_subplot(gs[0, 0])
    rewards = results['rewards']
    if len(rewards) > window:
        smooth_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(smooth_rewards, label='Smoothed Reward', color='blue', alpha=0.7)
    ax1.plot(rewards, alpha=0.3, color='lightblue', label='Raw Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Success Rate曲线
    ax2 = fig.add_subplot(gs[0, 1])
    success_data = []
    for i, s in enumerate(results['success']):
        success_data.extend([s] * 50)  # 每50轮统计一次
    ax2.plot(success_data, label='Success Rate', color='green', linewidth=2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate')
    ax2.set_title('Success Rate over Training')
    ax2.set_ylim([0, 1])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Collision Rate曲线
    ax3 = fig.add_subplot(gs[0, 2])
    collision_data = []
    for i, c in enumerate(results['collision']):
        collision_data.extend([c] * 50)
    ax3.plot(collision_data, label='Collision Rate', color='red', linewidth=2)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Collision Rate')
    ax3.set_title('Collision Rate over Training')
    ax3.set_ylim([0, 1])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 阶段对比柱状图 - Success
    ax4 = fig.add_subplot(gs[1, 0])
    stages = [r['stage'].replace('Stage ', 'S') for r in stage_final_results]
    successes = [r['success'] for r in stage_final_results]
    colors = ['green' if s >= 0.65 else 'orange' if s >= 0.50 else 'red' for s in successes]
    bars = ax4.bar(range(len(stages)), successes, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(stages)))
    ax4.set_xticklabels(stages, rotation=45, ha='right')
    ax4.set_ylabel('Success Rate')
    ax4.set_title('Final Success Rate by Stage')
    ax4.set_ylim([0, 1])
    ax4.axhline(y=0.65, color='green', linestyle='--', alpha=0.5, label='Target (65%)')
    ax4.axhline(y=0.50, color='orange', linestyle='--', alpha=0.5, label='Baseline (50%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, successes)):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 5. 阶段对比柱状图 - Collision
    ax5 = fig.add_subplot(gs[1, 1])
    collisions = [r['collision'] for r in stage_final_results]
    colors = ['green' if c <= 0.20 else 'orange' if c <= 0.35 else 'red' for c in collisions]
    bars = ax5.bar(range(len(stages)), collisions, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(stages)))
    ax5.set_xticklabels(stages, rotation=45, ha='right')
    ax5.set_ylabel('Collision Rate')
    ax5.set_title('Final Collision Rate by Stage')
    ax5.set_ylim([0, 1])
    ax5.axhline(y=0.20, color='green', linestyle='--', alpha=0.5, label='Excellent (<20%)')
    ax5.axhline(y=0.35, color='orange', linestyle='--', alpha=0.5, label='Acceptable (<35%)')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    for i, (bar, val) in enumerate(zip(bars, collisions)):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.1%}', ha='center', va='bottom', fontsize=9)
    
    # 6. V5 vs V6 对比
    ax6 = fig.add_subplot(gs[1, 2])
    v5_results = [0.97, 0.77, 0.39, 0.71, 0.73, 0.58, 0.51]  # V5历史数据
    v6_results = successes
    
    x = np.arange(len(stages))
    width = 0.35
    
    bars1 = ax6.bar(x - width/2, v5_results, width, label='V5 Baseline', alpha=0.7, color='skyblue')
    bars2 = ax6.bar(x + width/2, v6_results, width, label='V6 Optimized', alpha=0.7, color='coral')
    
    ax6.set_xticks(x)
    ax6.set_xticklabels(stages, rotation=45, ha='right')
    ax6.set_ylabel('Success Rate')
    ax6.set_title('V5 vs V6 Comparison')
    ax6.set_ylim([0, 1])
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n📊 训练结果图表已保存: {save_path}")
    plt.close()


if __name__ == "__main__":
    print("=" * 80)
    print("LSTM-DDPG V6 综合优化训练")
    print("=" * 80)
    print("\n关键改进:")
    print("  1. ✅ Stage 7训练量: 800 → 1500 episodes")
    print("  2. ✅ 新增Stage 6.5: 90%速度过渡")
    print("  3. ✅ 动态success_bonus: 200/150/120/100")
    print("  4. ✅ 自适应探索率衰减")
    print("  5. ✅ 防御性评估机制")
    print("  6. ✅ 早停策略 (65%自动停止)")
    print("\n预期成果:")
    print("  - Stage 6.5: 58-60% (新增)")
    print("  - Stage 7:   62-68% (V5: 51%)")
    print("  - 训练时间: +3-4小时")
    print("=" * 80)
    
    # 初始化Agent (假设你已经有LSTMDDPGAgent类)
    # from your_agent import LSTMDDPGAgent
    # agent = LSTMDDPGAgent(
    #     state_dim=18,  # 16 LiDAR + 2 velocity
    #     action_dim=2,
    #     hidden_dim=128,
    #     lr_actor=1e-4,
    #     lr_critic=1e-3,
    #     gamma=0.99,
    #     tau=0.005,
    # )
    
    # 🆕 开始训练
    # trained_agent, results, stage_results = train_comprehensive_curriculum(
    #     agent, 
    #     eval_interval=50
    # )
    
    # 🆕 保存最终模型
    # torch.save(trained_agent.actor.state_dict(), 'lstm_ddpg_actor_v6_comprehensive.pth')
    # print("\n✅ 最终模型已保存: lstm_ddpg_actor_v6_comprehensive.pth")
    
    # 🆕 绘制结果
    # plot_comprehensive_results(results, stage_results)
    
    print("\n🎉 训练完成！")
