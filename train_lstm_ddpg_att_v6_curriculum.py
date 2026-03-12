#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_lstm_ddpg_att_v6_curriculum.py - 与LSTM-DDPG V6完全一致的课程学习

完全复制LSTM-DDPG V6的训练策略，确保公平对比:
1. ✅ 相同的8阶段课程
2. ✅ 相同的障碍物配置
3. ✅ 相同的速度渐进
4. ✅ 相同的RewardShaping策略
5. ✅ 相同的探索率衰减
6. ✅ 相同的早停策略

唯一区别: 使用LSTM-DDPG-ATT网络（带空间和时间注意力）
"""

from __future__ import annotations

import argparse
import os
import time
import copy
from collections import deque
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

from config import DDPGConfig, EnvConfig, LOG_DIR, MODEL_DIR, RESULT_DIR
from environment import NavigationEnv
from lstm_ddpg_att import LSTMDdpgAgent  # ← 使用ATT版本
from utils import Logger, plot_training_curves


class AdaptiveRewardShapingWrapper:
    """
    自适应RewardShaping包装器
    根据训练阶段动态调整success_bonus
    """

    def __init__(self, env: NavigationEnv, stage_type: str = 'navigation', success_bonus: float | None = None):
        self.env = env
        self.stage_type = stage_type

        # 动态success_bonus策略
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
        if done and info.get('reason') == 'goal_reached':
            reward += self.success_bonus

        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


class DefensiveEvaluator:
    """
    防御性评估器 - 检测灾难性遗忘
    """

    def __init__(self, env_config: Dict):
        self.env_config = env_config
        self.history: List[float] = []

    def evaluate(self, agent: LSTMDdpgAgent, n_episodes: int = 20) -> float:
        """评估Agent在简单场景(无障碍)上的表现"""
        # 创建简单评估环境
        eval_env = NavigationEnv(
            use_enhanced_state=self.env_config.get('use_enhanced_state', False),
            enhanced_state_config=self.env_config.get('enhanced_state_config'),
            dynamic_speed_min=EnvConfig.DYNAMIC_OBS_VEL_MIN,
            dynamic_speed_max=EnvConfig.DYNAMIC_OBS_VEL_MAX,
        )
        # 临时禁用障碍物
        original_static = EnvConfig.NUM_STATIC_OBSTACLES
        original_dynamic = EnvConfig.NUM_DYNAMIC_OBSTACLES
        EnvConfig.NUM_STATIC_OBSTACLES = 0
        EnvConfig.NUM_DYNAMIC_OBSTACLES = 0

        successes = 0
        for _ in range(n_episodes):
            state = eval_env.reset()
            state_queue = deque([state.copy() for _ in range(agent.history_len)], maxlen=agent.history_len)

            for _ in range(EnvConfig.MAX_STEPS):
                state_seq = np.asarray(state_queue, dtype=np.float32)
                action = agent.act(state_seq, add_noise=False)
                next_state, _, done, info = eval_env.step(action.reshape(-1))
                state_queue.append(next_state.copy())

                if done:
                    success, _ = eval_env.get_episode_status()
                    if success:
                        successes += 1
                    break

        # 恢复设置
        EnvConfig.NUM_STATIC_OBSTACLES = original_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = original_dynamic
        eval_env.close()

        return successes / n_episodes

    def check_forgetting(self, current_success: float, threshold: float = 0.90) -> bool:
        """检查是否发生灾难性遗忘"""
        self.history.append(current_success)

        if len(self.history) < 2:
            return False

        # 检测性能大幅下降
        if self.history[-1] < threshold and self.history[-1] < self.history[-2] - 0.10:
            return True

        return False


def train_one_stage(
    agent: LSTMDdpgAgent,
    env: NavigationEnv,
    stage_config: Dict,
    stage_name: str,
    defensive_evaluator: DefensiveEvaluator | None = None,
) -> Dict:
    """训练单个阶段"""

    num_episodes = stage_config['episodes']
    use_reward_shaping = stage_config.get('use_reward_shaping', False)
    stage_type = stage_config.get('stage_type', 'navigation')
    success_bonus = stage_config.get('success_bonus', None)
    explore_reset = stage_config.get('explore_reset', 0.25)
    adaptive_exploration = stage_config.get('adaptive_exploration', False)
    exploration_decay_start = stage_config.get('exploration_decay_start', 300)
    exploration_decay_rate = stage_config.get('exploration_decay_rate', 0.995)
    min_noise_sigma = stage_config.get('min_noise_sigma', 0.05)
    early_stop_threshold = stage_config.get('early_stop_threshold', None)
    early_stop_patience = stage_config.get('early_stop_patience', 300)

    print(f"\n{'='*80}")
    print(f"{stage_name}")
    print(f"{'='*80}")
    print(f"  训练轮数: {num_episodes}")
    print(f"  Reward Shaping: {use_reward_shaping}")
    if use_reward_shaping:
        print(f"  Success Bonus: {success_bonus if success_bonus else 'Auto'}")
    print(f"  探索率重置: {explore_reset}")
    if adaptive_exploration:
        print(f"  自适应探索: 开启 (从第{exploration_decay_start}轮衰减)")
    if early_stop_threshold:
        print(f"  早停策略: 达到{early_stop_threshold:.0%}成功率停止")

    # 可选包装器
    if use_reward_shaping:
        train_env = AdaptiveRewardShapingWrapper(env, stage_type=stage_type, success_bonus=success_bonus)
    else:
        train_env = env

    # 重置探索噪声
    agent.noise.max_sigma = explore_reset
    agent.noise.sigma = explore_reset
    agent.reset_noise()
    print(f"  已重置探索噪声: sigma={explore_reset}")

    # 训练统计
    rewards = []
    steps_list = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    best_avg_reward = -float('inf')
    patience_counter = 0

    start_time = time.time()

    for episode in range(num_episodes):
        state = train_env.reset()
        agent.reset_noise()

        state_queue = deque([state.copy() for _ in range(agent.history_len)], maxlen=agent.history_len)

        episode_reward = 0.0
        step = 0

        for step in range(EnvConfig.MAX_STEPS):
            state_seq = np.asarray(state_queue, dtype=np.float32)
            action = agent.act(state_seq, add_noise=True)
            action_flat = action.reshape(-1)

            next_state, reward, done, info = train_env.step(action_flat)

            agent.step(state, action_flat, reward, next_state, done)
            state_queue.append(next_state.copy())

            if agent.should_learn():
                agent.learn()

            episode_reward += float(reward)
            state = next_state

            if done:
                break

        # 记录统计
        rewards.append(episode_reward)
        steps_list.append(step + 1)

        success, _ = train_env.get_episode_status() if hasattr(train_env, 'get_episode_status') else env.get_episode_status()
        if success:
            success_count += 1
        elif info.get("reason") in ("collision_obstacle", "collision_wall"):
            collision_count += 1
        else:
            timeout_count += 1

        # 自适应探索率衰减
        if adaptive_exploration and episode >= exploration_decay_start:
            current_sigma = agent.noise.sigma
            new_sigma = max(min_noise_sigma, current_sigma * exploration_decay_rate)
            agent.noise.sigma = new_sigma
            agent.noise.max_sigma = new_sigma

        # 每10轮显示一次
        if (episode + 1) % 10 == 0:
            recent_rewards = rewards[-100:] if len(rewards) >= 100 else rewards
            avg_reward = np.mean(recent_rewards)
            recent_success = success_count / (episode + 1)

            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            eta_seconds = (num_episodes - episode - 1) / eps_per_sec if eps_per_sec > 0 else 0

            print(
                f"  Ep {episode+1:4d}/{num_episodes} | "
                f"R: {episode_reward:6.1f} | "
                f"Avg: {avg_reward:6.1f} | "
                f"SR: {recent_success:5.1%} | "
                f"ETA: {eta_seconds/60:.1f}min | "
                f"{info.get('reason', '')[:12]}"
            )

            # 早停逻辑
            if early_stop_threshold is not None:
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    patience_counter = 0
                else:
                    patience_counter += 1

                if recent_success >= early_stop_threshold and episode >= 100:
                    print(
                        f"✅ 提前达到目标 {recent_success:.2%} >= {early_stop_threshold:.2%} "
                        f"(已训练{episode + 1}轮), 停止训练")
                    break

                if patience_counter >= early_stop_patience:
                    print(f"⚠️ {early_stop_patience}轮无改善，提前停止")
                    break

        # 防御性评估 (每500轮)
        if defensive_evaluator is not None and (episode + 1) % 500 == 0:
            print("  [防御性评估] 检查简单场景性能...")
            simple_success = defensive_evaluator.evaluate(agent, n_episodes=20)
            print(f"  简单场景成功率: {simple_success:.2%}")

            if defensive_evaluator.check_forgetting(simple_success, threshold=0.90):
                print("  ⚠️ 检测到灾难性遗忘！建议检查训练")

    # 计算最终统计
    actual_episodes = episode + 1
    if actual_episodes > 0:
        final_success = success_count / actual_episodes
        final_collision = collision_count / actual_episodes
        final_timeout = timeout_count / actual_episodes
    else:
        final_success = 0.0
        final_collision = 0.0
        final_timeout = 0.0

    print(f"\n{stage_name} 完成:")
    if actual_episodes < num_episodes:
        print(f"  实际训练轮数: {actual_episodes}/{num_episodes} (提前停止)")
    else:
        print(f"  训练轮数: {actual_episodes}")
    print(f"  最终成功率: {final_success:.2%}")
    print(f"  最终碰撞率: {final_collision:.2%}")
    print(f"  最终超时率: {final_timeout:.2%}")

    return {
        'rewards': rewards,
        'steps': steps_list,
        'final_success': final_success,
        'final_collision': final_collision,
        'final_timeout': final_timeout,
    }


def train_att_v6_comprehensive(args: argparse.Namespace) -> None:
    """
    LSTM-DDPG-ATT V6综合课程训练（与LSTM-DDPG V6完全一致）
    """
    print("=" * 80)
    print("LSTM-DDPG + Attention V6 综合课程训练")
    print("=" * 80)
    print("\n🎯 训练目标: 与LSTM-DDPG V6使用完全相同的课程学习策略")
    print("   唯一区别: 使用带注意力机制的网络架构\n")

    # 环境配置
    enhanced_cfg = {
        "n_sectors": args.n_sectors,
        "sector_method": args.sector_method,
        "use_lidar_diff": (not args.disable_lidar_diff),
        "use_delta_yaw": (not args.disable_delta_yaw),
    }

    # 定义8阶段课程（与LSTM-DDPG V6完全一致）
    curriculum = [
        # =========== Stage 1-2: 基础导航 ===========
        {
            'name': 'Stage 1: Navigation',
            'episodes': 800,
            'num_static': 0,
            'num_dynamic': 0,
            'speed_scale': 1.0,
            'use_reward_shaping': False,
            'stage_type': 'navigation',
            'explore_reset': 0.25,
            'adaptive_exploration': False,
        },

        {
            'name': 'Stage 2: Static Obstacles',
            'episodes': 800,
            'num_static': 3,
            'num_dynamic': 0,
            'speed_scale': 1.0,
            'use_reward_shaping': False,
            'stage_type': 'navigation',
            'explore_reset': 0.25,
            'adaptive_exploration': False,
        },

        # =========== Stage 3-5: 动态障碍渐进 ===========
        {
            'name': 'Stage 3: Dynamic@20%',
            'episodes': 800,
            'num_static': 2,
            'num_dynamic': 2,
            'speed_scale': 0.20,
            'use_reward_shaping': True,
            'stage_type': 'sparse',
            'success_bonus': 150.0,
            'explore_reset': 0.20,
            'adaptive_exploration': False,
        },

        {
            'name': 'Stage 4: Dynamic@40%',
            'episodes': 800,
            'num_static': 2,
            'num_dynamic': 2,
            'speed_scale': 0.40,
            'use_reward_shaping': True,
            'stage_type': 'medium',
            'success_bonus': 150.0,
            'explore_reset': 0.18,
            'adaptive_exploration': False,
        },

        {
            'name': 'Stage 5: Dynamic@60%',
            'episodes': 800,
            'num_static': 2,
            'num_dynamic': 2,
            'speed_scale': 0.60,
            'use_reward_shaping': True,
            'stage_type': 'medium',
            'success_bonus': 150.0,
            'explore_reset': 0.16,
            'adaptive_exploration': False,
        },

        # =========== Stage 6: 密集障碍@80% ===========
        {
            'name': 'Stage 6: Dense@80%',
            'episodes': 900,
            'num_static': 3,
            'num_dynamic': 2,
            'speed_scale': 0.80,
            'use_reward_shaping': True,
            'stage_type': 'dense',
            'success_bonus': 120.0,
            'explore_reset': 0.15,
            'adaptive_exploration': False,
        },

        # =========== Stage 6.5: 90%速度过渡阶段 ===========
        {
            'name': 'Stage 6.5: Transition@90%',
            'episodes': 600,
            'num_static': 4,
            'num_dynamic': 2,
            'speed_scale': 0.90,
            'use_reward_shaping': True,
            'stage_type': 'dense',
            'success_bonus': 100.0,
            'explore_reset': 0.12,
            'adaptive_exploration': True,
            'exploration_decay_start': 300,
            'exploration_decay_rate': 0.995,
            'min_noise_sigma': 0.05,
        },

        # =========== Stage 7: 最终挑战@100% ===========
        {
            'name': 'Stage 7: Full Challenge@100%',
            'episodes': 1500,
            'num_static': 4,
            'num_dynamic': 2,
            'speed_scale': 1.0,
            'use_reward_shaping': True,
            'stage_type': 'dense',
            'success_bonus': 100.0,
            'explore_reset': 0.10,
            'adaptive_exploration': True,
            'exploration_decay_start': 500,
            'exploration_decay_rate': 0.995,
            'min_noise_sigma': 0.05,
            'early_stop_threshold': 0.85,  # 达到85%自动停止
            'early_stop_patience': 300,
        },
    ]

    print("\n课程设计 (与LSTM-DDPG V6完全一致):")
    for i, stage in enumerate(curriculum, 1):
        print(f"  {i}. {stage['name']}: {stage['episodes']}轮, "
              f"{stage['num_static']}静态+{stage['num_dynamic']}动态@{stage['speed_scale']:.0%}速度")

    total_episodes = sum(s['episodes'] for s in curriculum)
    print(f"\n总训练轮数: {total_episodes}")
    print(f"\n关键特性:")
    print(f"  1. ✅ Stage 7训练量: 1500 episodes")
    print(f"  2. ✅ 新增Stage 6.5: 90%速度过渡")
    print(f"  3. ✅ 动态success_bonus: 200/150/120/100")
    print(f"  4. ✅ 自适应探索率衰减")
    print(f"  5. ✅ 防御性评估机制")
    print(f"  6. ✅ 早停策略 (65%自动停止)")

    # 创建初始环境
    env = NavigationEnv(
        use_enhanced_state=(not args.legacy_state),
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=EnvConfig.DYNAMIC_OBS_VEL_MIN,
        dynamic_speed_max=EnvConfig.DYNAMIC_OBS_VEL_MAX,
    )

    print(f"\n环境信息:")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")

    # 创建LSTM-DDPG-ATT Agent（带注意力机制）
    print(f"\n创建 LSTM-DDPG + Attention Agent:")
    print(f"  History len: {args.history_len}")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  LSTM hidden: {args.lstm_hidden_dim}")
    print(f"  Spatial Attention: heads={args.spatial_att_heads}, dim={args.sector_model_dim}")
    print(f"  Temporal Attention: heads={args.temporal_att_heads}, dim={args.temporal_att_dim}")

    agent = LSTMDdpgAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        history_len=args.history_len,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        embed_dim=args.embed_dim,
        lstm_hidden_dim=args.lstm_hidden_dim,
        update_every=args.update_every,
        update_times=args.update_times,
        # Attention配置
        use_spatial_att=args.use_spatial_att,
        use_temporal_att=args.use_temporal_att,
        sector_model_dim=args.sector_model_dim,
        spatial_att_heads=args.spatial_att_heads,
        temporal_att_dim=args.temporal_att_dim,
        temporal_att_heads=args.temporal_att_heads,
        att_dropout=args.att_dropout,
    )

    # 防御性评估器
    env_config = {
        'use_enhanced_state': not args.legacy_state,
        'enhanced_state_config': enhanced_cfg,
    }
    defensive_evaluator = DefensiveEvaluator(env_config)

    # 保存配置
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    input("\n按Enter键开始训练...")

    # ================================================================
    # 逐阶段训练
    # ================================================================
    all_results = []
    overall_start = time.time()

    for stage_idx, stage in enumerate(curriculum):
        # 配置障碍物
        EnvConfig.NUM_STATIC_OBSTACLES = stage['num_static']
        EnvConfig.NUM_DYNAMIC_OBSTACLES = stage['num_dynamic']

        # 配置动态速度
        base_min = EnvConfig.DYNAMIC_OBS_VEL_MIN
        base_max = EnvConfig.DYNAMIC_OBS_VEL_MAX
        speed_scale = stage['speed_scale']
        
        # 根据speed_scale调整速度
        env.dynamic_speed_min = base_min * speed_scale
        env.dynamic_speed_max = base_max * speed_scale

        print(f"\n{'='*80}")
        print(f"阶段 {stage_idx + 1}/{len(curriculum)}")
        print(f"{'='*80}")
        print(f"障碍物: {stage['num_static']}静态 + {stage['num_dynamic']}动态")
        print(f"速度: {env.dynamic_speed_min:.2f}-{env.dynamic_speed_max:.2f} m/s (scale={speed_scale:.0%})")

        # 训练这个阶段
        result = train_one_stage(
            agent=agent,
            env=env,
            stage_config=stage,
            stage_name=stage['name'],
            defensive_evaluator=defensive_evaluator,
        )

        all_results.append({
            'stage': stage['name'],
            'success_rate': result['final_success'],
            'collision_rate': result['final_collision'],
            'timeout_rate': result['final_timeout'],
        })

        # 保存checkpoint
        checkpoint_path = os.path.join(MODEL_DIR, f"lstm_ddpg_att_v6_stage{stage_idx+1}_checkpoint.pth")
        agent.save(checkpoint_path)
        print(f"  ✅ 保存检查点: {checkpoint_path}")

    # ================================================================
    # 训练完成
    # ================================================================
    total_time = time.time() - overall_start

    print("\n" + "=" * 80)
    print("🎉 所有阶段训练完成！")
    print("=" * 80)

    print(f"\n各阶段成功率:")
    for result in all_results:
        print(f"  {result['stage']}: {result['success_rate']:.1%}")

    print(f"\n总训练时间: {total_time/3600:.2f} 小时")
    print(f"总训练轮数: {total_episodes}")

    # 保存最终模型
    final_path = os.path.join(MODEL_DIR, f"lstm_ddpg_att_v6_final_{timestamp}.pth")
    agent.save(final_path)
    print(f"\n✅ 最终模型保存: {final_path}")

    print("\n下一步建议:")
    print("1. 测试模型性能:")
    print(f"   python test_three_algorithms_simple.py \\")
    print(f"       --ddpg_model models/ddpg_final.pth \\")
    print(f"       --lstm_model models/lstm_ddpg_v6_final_20260125_011043.pth \\")
    print(f"       --att_model {final_path} \\")
    print(f"       --episodes 200")

    env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM-DDPG + Attention V6 Curriculum Training")

    # 训练参数
    p.add_argument("--batch_size", type=int, default=DDPGConfig.BATCH_SIZE)
    p.add_argument("--buffer_size", type=int, default=DDPGConfig.BUFFER_SIZE)
    p.add_argument("--history_len", type=int, default=5)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--lstm_hidden_dim", type=int, default=64)
    p.add_argument("--update_every", type=int, default=1)
    p.add_argument("--update_times", type=int, default=1)

    # 环境配置
    p.add_argument("--legacy_state", action="store_true")
    p.add_argument("--n_sectors", type=int, default=16)
    p.add_argument("--sector_method", type=str, default="min")
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    # Attention配置
    p.add_argument("--use_spatial_att", action="store_true", default=True)
    p.add_argument("--use_temporal_att", action="store_true", default=True)
    p.add_argument("--sector_model_dim", type=int, default=32)
    p.add_argument("--spatial_att_heads", type=int, default=4)
    p.add_argument("--temporal_att_dim", type=int, default=64)
    p.add_argument("--temporal_att_heads", type=int, default=4)
    p.add_argument("--att_dropout", type=float, default=0.0)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_att_v6_comprehensive(args)
