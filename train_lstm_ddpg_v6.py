#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_lstm_ddpg_v6.py - V6综合优化版

核心改进:
1. ✅ 增加Stage 7训练量: 800 → 1500 episodes
2. ✅ 添加Stage 6.5: 90%速度过渡阶段  
3. ✅ 动态success_bonus: 根据难度调整 200/150/100
4. ✅ 自适应探索率衰减
5. ✅ 防御性评估机制
6. ✅ 早停策略

预期提升: 51% → 65-70%
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
from lstm_ddpg import LSTMDdpgAgent
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
            dynamic_speed_min=EnvConfig.DYNAMIC_OBS_VEL_MIN,  # ← 从config读取
            dynamic_speed_max=EnvConfig.DYNAMIC_OBS_VEL_MAX,  # ← 从config读取
        )
        # 临时禁用障碍物
        original_static = EnvConfig.NUM_STATIC_OBSTACLES
        original_dynamic = EnvConfig.NUM_DYNAMIC_OBSTACLES
        EnvConfig.NUM_STATIC_OBSTACLES = 0
        EnvConfig.NUM_DYNAMIC_OBSTACLES = 0

        successes = []
        for _ in range(n_episodes):
            state = eval_env.reset()
            state_queue = deque([state.copy() for _ in range(agent.history_len)],
                                maxlen=agent.history_len)
            done = False

            while not done:
                state_seq = np.asarray(state_queue, dtype=np.float32)
                action = agent.act(state_seq, step=0, add_noise=False)
                action_flat = action.reshape(-1)

                next_state, _, done, info = eval_env.step(action_flat)
                state_queue.append(next_state.copy())

            successes.append(1 if info.get('reason') == 'goal_reached' else 0)

        # 恢复原始设置
        EnvConfig.NUM_STATIC_OBSTACLES = original_static
        EnvConfig.NUM_DYNAMIC_OBSTACLES = original_dynamic
        eval_env.close()

        success_rate = np.mean(successes)
        self.history.append(success_rate)

        return success_rate

    def check_forgetting(self, current_rate: float, threshold: float = 0.90) -> bool:
        """检查是否发生灾难性遗忘"""
        if len(self.history) < 2:
            return False

        # 如果当前成功率低于阈值，且比之前下降超过10%
        if current_rate < threshold:
            prev_rate = self.history[-2]
            if current_rate < prev_rate - 0.10:
                return True

        return False


def scale_dynamic_obstacle_speed(speed_scale: float) -> None:
    """调整动态障碍物速度"""
    # 保存原始值
    if not hasattr(scale_dynamic_obstacle_speed, 'original_min'):
        scale_dynamic_obstacle_speed.original_min = EnvConfig.DYNAMIC_OBS_VEL_MIN
        scale_dynamic_obstacle_speed.original_max = EnvConfig.DYNAMIC_OBS_VEL_MAX

    # 缩放速度
    EnvConfig.DYNAMIC_OBS_VEL_MIN = scale_dynamic_obstacle_speed.original_min * speed_scale
    EnvConfig.DYNAMIC_OBS_VEL_MAX = scale_dynamic_obstacle_speed.original_max * speed_scale


def train_stage_v6(
        env: NavigationEnv,
        agent: LSTMDdpgAgent,
        num_episodes: int,
        stage_name: str,
        stage_config: Dict,
        defensive_evaluator: DefensiveEvaluator | None = None,
        logger: Logger | None = None,
        global_episode_offset: int = 0,
) -> Dict:
    """
    V6增强训练函数

    新增功能:
    - 自适应探索率衰减
    - 防御性评估
    - 早停策略
    """
    rewards = []
    steps_list = []
    success_count = 0
    collision_count = 0
    timeout_count = 0

    # 早停参数
    early_stop_threshold = stage_config.get('early_stop_threshold', None)
    early_stop_patience = stage_config.get('early_stop_patience', 200)
    best_success_rate = 0.0
    patience_counter = 0

    # 自适应探索率衰减参数
    adaptive_exploration = stage_config.get('adaptive_exploration', False)
    exploration_decay_start = stage_config.get('exploration_decay_start', 500)
    exploration_decay_rate = stage_config.get('exploration_decay_rate', 0.995)
    min_noise_sigma = stage_config.get('min_noise_sigma', 0.05)

    # 学习开始步数
    learn_start = stage_config.get('learn_start', DDPGConfig.LEARN_START)
    total_steps = 0
    learn_count = 0

    episode_buffer = deque(maxlen=100)

    print(f"\n{'=' * 80}")
    print(f"🎯 {stage_name} ({num_episodes} episodes)")
    print(f"{'=' * 80}")

    for episode in range(num_episodes):
        state = env.reset()
        agent.reset_noise()

        state_queue = deque([state.copy() for _ in range(agent.history_len)],
                            maxlen=agent.history_len)

        episode_reward = 0.0

        for step in range(EnvConfig.MAX_STEPS):
            total_steps += 1

            state_seq = np.asarray(state_queue, dtype=np.float32)
            action = agent.act(state_seq, step=total_steps, add_noise=True)
            action_flat = action.reshape(-1)

            next_state, reward, done, info = env.step(action_flat)

            agent.step(state, action_flat, reward, next_state, done)

            state_queue.append(next_state.copy())

            # 学习
            if total_steps > learn_start and agent.should_learn():
                agent.learn()
                learn_count += 1

            episode_reward += float(reward)
            state = next_state

            if done:
                break

        rewards.append(episode_reward)
        steps_list.append(step + 1)

        # 统计
        success = (info.get('reason') == 'goal_reached')
        collision = info.get('reason') in ('collision_obstacle', 'collision_wall')
        timeout = info.get('reason') == 'max_steps'

        if success:
            success_count += 1
        elif collision:
            collision_count += 1
        elif timeout:
            timeout_count += 1

        episode_buffer.append({'success': success, 'collision': collision, 'timeout': timeout})

        # 自适应探索率衰减
        if adaptive_exploration and episode > exploration_decay_start:
            old_sigma = agent.noise.sigma
            agent.noise.sigma = max(min_noise_sigma, agent.noise.sigma * exploration_decay_rate)
            if episode % 100 == 0 and abs(old_sigma - agent.noise.sigma) > 1e-6:
                print(f"  [探索衰减] ε: {old_sigma:.4f} → {agent.noise.sigma:.4f}")

        # 周期性统计
        if (episode + 1) % 50 == 0:
            recent_success = np.mean([ep['success'] for ep in episode_buffer])
            recent_collision = np.mean([ep['collision'] for ep in episode_buffer])
            recent_timeout = np.mean([ep['timeout'] for ep in episode_buffer])
            avg_reward = np.mean(rewards[-100:])
            avg_steps = np.mean(steps_list[-100:])

            print(f"[{stage_name} | Ep {episode + 1:4d}] "
                  f"Success: {recent_success:.2%} | "
                  f"Collision: {recent_collision:.2%} | "
                  f"Timeout: {recent_timeout:.2%} | "
                  f"AvgRet: {avg_reward:.1f} | "
                  f"AvgSteps: {avg_steps:.1f}")

            # 日志记录
            if logger:
                logger.log({
                    'global_episode': global_episode_offset + episode + 1,
                    'stage': stage_name,
                    'episode_in_stage': episode + 1,
                    'success_rate': recent_success,
                    'collision_rate': recent_collision,
                    'timeout_rate': recent_timeout,
                    'avg_reward': avg_reward,
                    'avg_steps': avg_steps,
                    'total_steps': total_steps,
                    'learn_count': learn_count,
                })

            # 早停检查
            if early_stop_threshold is not None:
                # 🔧 修复：只在训练至少300轮后才允许早停
                if episode >= 300:
                    if recent_success > best_success_rate:
                        best_success_rate = recent_success
                        patience_counter = 0
                    else:
                        patience_counter += 50

                    if recent_success >= early_stop_threshold:
                        print(
                            f"✅ 提前达到目标 {recent_success:.2%} >= {early_stop_threshold:.2%} (已训练{episode + 1}轮), 停止训练")
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

    # 计算最终统计 (使用实际训练的episode数)
    actual_episodes = episode + 1  # episode是循环中的索引，0-based
    if actual_episodes > 0:
        final_success = success_count / actual_episodes
        final_collision = collision_count / actual_episodes
        final_timeout = timeout_count / actual_episodes
    else:
        final_success = 0.0
        final_collision = 0.0
        final_timeout = 0.0

    # 显示实际训练轮数
    actual_trained = episode + 1  # episode是0-based，所以+1
    print(f"\n{stage_name} 完成:")
    if actual_trained < num_episodes:
        print(f"  实际训练轮数: {actual_trained}/{num_episodes} (提前停止)")
    else:
        print(f"  训练轮数: {actual_trained}")
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


def train_v6_comprehensive(args: argparse.Namespace) -> None:
    """
    V6综合课程训练
    """
    print("=" * 80)
    print("LSTM-DDPG V6 综合优化训练")
    print("=" * 80)

    # 环境配置
    enhanced_cfg = {
        "n_sectors": args.n_sectors,
        "sector_method": args.sector_method,
        "use_lidar_diff": (not args.disable_lidar_diff),
        "use_delta_yaw": (not args.disable_delta_yaw),
    }

    # 定义8阶段课程
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
            'episodes': 900,  # 从800增加到900
            'num_static': 3,
            'num_dynamic': 2,
            'speed_scale': 0.80,
            'use_reward_shaping': True,
            'stage_type': 'dense',
            'success_bonus': 120.0,
            'explore_reset': 0.15,
            'adaptive_exploration': False,
        },

        # =========== Stage 6.5: 90%速度过渡阶段 (新增) ===========
        {
            'name': 'Stage 6.5: Transition@90%',
            'episodes': 600,
            'num_static': 4,  # 提前适应4个静态障碍
            'num_dynamic': 2,
            'speed_scale': 0.90,  # 90%速度过渡
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
            'episodes': 1500,  # 从800增加到1500
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
            'early_stop_threshold': 0.65,  # 达到65%自动停止
            'early_stop_patience': 300,
        },
    ]

    print("\nV6课程设计:")
    for i, stage in enumerate(curriculum, 1):
        print(f"  {i}. {stage['name']}: {stage['episodes']}轮, "
              f"{stage['num_static']}静态+{stage['num_dynamic']}动态@{stage['speed_scale']:.0%}速度")

    print(f"\n关键改进:")
    print(f"  1. ✅ Stage 7训练量: 800 → 1500 episodes")
    print(f"  2. ✅ 新增Stage 6.5: 90%速度过渡")
    print(f"  3. ✅ 动态success_bonus: 200/150/120/100")
    print(f"  4. ✅ 自适应探索率衰减")
    print(f"  5. ✅ 防御性评估机制")
    print(f"  6. ✅ 早停策略 (65%自动停止)")

    # 创建初始环境（用于初始化Agent）
    env = NavigationEnv(
        use_enhanced_state=(not args.legacy_state),
        enhanced_state_config=enhanced_cfg,
        dynamic_speed_min=EnvConfig.DYNAMIC_OBS_VEL_MIN,  # ← 从config读取
        dynamic_speed_max=EnvConfig.DYNAMIC_OBS_VEL_MAX,  # ← 从config读取
    )

    print(f"\nEnvironment:")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")

    # 创建Agent
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
    )

    # 保存配置
    agent.state_meta = {
        'legacy_state': bool(args.legacy_state),
        'n_sectors': int(args.n_sectors),
        'sector_method': str(args.sector_method),
        'disable_lidar_diff': bool(args.disable_lidar_diff),
        'disable_delta_yaw': bool(args.disable_delta_yaw),
        'history_len': int(args.history_len),
        'embed_dim': int(args.embed_dim),
        'lstm_hidden_dim': int(args.lstm_hidden_dim),
        'update_every': int(args.update_every),
        'curriculum': 'v6_comprehensive',
    }

    # Resume
    start_stage_idx = 0
    global_episode_offset = 0
    if args.resume and os.path.exists(args.resume):
        agent.load(args.resume)
        # 尝试解析stage
        try:
            base = os.path.basename(args.resume)
            if "stage" in base.lower():
                start_stage_idx = int(base.split("stage")[-1].split("_")[0]) - 1
        except Exception:
            start_stage_idx = 0
        print(f"Resumed from {args.resume}, starting at stage {start_stage_idx + 1}")

    # 创建防御性评估器
    defensive_evaluator = DefensiveEvaluator(
        env_config={
            'use_enhanced_state': not args.legacy_state,
            'enhanced_state_config': enhanced_cfg,
        }
    )

    # 创建Logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = Logger(os.path.join(LOG_DIR, f"lstm_ddpg_v6_training_{timestamp}.csv"))

    # 训练循环
    all_results = []
    stage_final_results = []

    train_start_time = time.time()

    for stage_idx in range(start_stage_idx, len(curriculum)):
        stage = curriculum[stage_idx]

        # 设置障碍物数量
        EnvConfig.NUM_STATIC_OBSTACLES = stage['num_static']
        EnvConfig.NUM_DYNAMIC_OBSTACLES = stage['num_dynamic']

        # 设置动态障碍物速度
        scale_dynamic_obstacle_speed(stage['speed_scale'])

        # 创建环境（从config读取速度）
        env = NavigationEnv(
            use_enhanced_state=(not args.legacy_state),
            enhanced_state_config=enhanced_cfg,
            dynamic_speed_min=EnvConfig.DYNAMIC_OBS_VEL_MIN,  # ← 从config读取（已缩放）
            dynamic_speed_max=EnvConfig.DYNAMIC_OBS_VEL_MAX,  # ← 从config读取（已缩放）
        )

        # 应用RewardShaping
        if stage['use_reward_shaping']:
            stage_type = stage.get('stage_type', 'navigation')
            success_bonus = stage.get('success_bonus', None)
            env = AdaptiveRewardShapingWrapper(
                env,
                stage_type=stage_type,
                success_bonus=success_bonus
            )

        # 重置探索率
        explore_reset = stage['explore_reset']
        agent.noise.sigma = explore_reset * (agent.noise.max_sigma - agent.noise.min_sigma) + agent.noise.min_sigma
        print(f"  [探索重置: ε={agent.noise.sigma:.2f}]")

        # 训练当前阶段
        results = train_stage_v6(
            env=env,
            agent=agent,
            num_episodes=stage['episodes'],
            stage_name=stage['name'],
            stage_config=stage,
            defensive_evaluator=defensive_evaluator if stage_idx >= 2 else None,  # Stage 3+才启用
            logger=logger,
            global_episode_offset=global_episode_offset,
        )

        all_results.append(results)
        stage_final_results.append({
            'stage': stage['name'],
            'success': results['final_success'],
            'collision': results['final_collision'],
            'timeout': results['final_timeout'],
        })

        global_episode_offset += stage['episodes']

        # 保存阶段检查点
        checkpoint_path = os.path.join(MODEL_DIR, f"lstm_ddpg_v6_stage{stage_idx + 1}_checkpoint.pth")
        agent.save(checkpoint_path)
        print(f"  ✅ 检查点已保存: {checkpoint_path}")

        env.close()

    total_time = time.time() - train_start_time

    # 打印最终总结
    print("\n" + "=" * 80)
    print("🎉 全部训练完成！阶段总结:")
    print("=" * 80)
    for result in stage_final_results:
        print(f"{result['stage']:30s} | Success: {result['success']:6.2%} | "
              f"Collision: {result['collision']:6.2%} | Timeout: {result['timeout']:6.2%}")
    print("=" * 80)
    print(f"总训练时间: {total_time / 3600:.2f} 小时")

    # 保存最终模型
    final_model_path = os.path.join(MODEL_DIR, f"lstm_ddpg_v6_final_{timestamp}.pth")
    agent.save(final_model_path)
    print(f"\n✅ 最终模型已保存: {final_model_path}")

    logger.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LSTM-DDPG V6 Comprehensive Training")

    # 训练参数
    p.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    p.add_argument("--batch_size", type=int, default=DDPGConfig.BATCH_SIZE)
    p.add_argument("--buffer_size", type=int, default=DDPGConfig.BUFFER_SIZE)

    # 加速参数
    p.add_argument("--history_len", type=int, default=DDPGConfig.HISTORY_LEN)
    p.add_argument("--embed_dim", type=int, default=DDPGConfig.LSTM_EMBED_DIM)
    p.add_argument("--lstm_hidden_dim", type=int, default=DDPGConfig.LSTM_HIDDEN_DIM)
    p.add_argument("--update_every", type=int, default=DDPGConfig.UPDATE_EVERY)
    p.add_argument("--update_times", type=int, default=DDPGConfig.UPDATE_TIMES)

    # 环境/状态配置
    p.add_argument("--legacy_state", action="store_true", help="Use original 39D state")
    p.add_argument("--n_sectors", type=int, default=16, choices=[8, 16])
    p.add_argument("--sector_method", type=str, default="min", choices=["min", "mean"])
    p.add_argument("--disable_lidar_diff", action="store_true")
    p.add_argument("--disable_delta_yaw", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_v6_comprehensive(args)