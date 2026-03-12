#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
test_v6_model_visual.py - V6模型测试脚本（带可视化）

功能:
1. 实时环境渲染
2. 测试结果可视化图表
3. 成功率/碰撞率变化曲线
4. 轨迹热力图
5. 详细统计信息

用法:
    # 基础测试（无可视化）
    python test_v6_model_visual.py --model models/lstm_ddpg_v6_final_*.pth --episodes 100
    
    # 实时渲染（观看前10个episode）
    python test_v6_model_visual.py --model models/lstm_ddpg_v6_final_*.pth --episodes 10 --render
    
    # 完整测试+结果可视化
    python test_v6_model_visual.py --model models/lstm_ddpg_v6_final_*.pth --episodes 100 --save_plots
    
    # 慢速渲染（观看详细过程）
    python test_v6_model_visual.py --model models/lstm_ddpg_v6_final_*.pth --episodes 5 --render --slow
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import deque
from typing import Dict, List
import time
import os

from environment import NavigationEnv
from lstm_ddpg import LSTMDdpgAgent
from config import EnvConfig, RESULT_DIR


def test_model_with_visualization(
    model_path: str, 
    n_episodes: int = 100, 
    render: bool = False,
    slow_render: bool = False,
    save_plots: bool = True,
    enhanced_state: bool = True
) -> Dict:
    """测试V6模型性能（带可视化）"""
    
    print("=" * 80)
    print("🎨 V6 模型测试 - 可视化版")
    print("=" * 80)
    print(f"模型路径: {model_path}")
    print(f"测试轮数: {n_episodes}")
    print(f"测试场景: 4静态 + 2动态 @ 100%速度")
    print(f"动态障碍物速度: {EnvConfig.DYNAMIC_OBS_VEL_MIN:.2f}-{EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f} m/s")
    print(f"无人机最大速度: {EnvConfig.MAX_LINEAR_VEL:.2f} m/s")
    print(f"速度比: {EnvConfig.MAX_LINEAR_VEL/EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f}:1")
    print(f"实时渲染: {'是' if render else '否'}")
    print(f"保存图表: {'是' if save_plots else '否'}")
    print("=" * 80 + "\n")
    
    # 设置环境
    EnvConfig.NUM_STATIC_OBSTACLES = 4
    EnvConfig.NUM_DYNAMIC_OBSTACLES = 2
    
    # 创建环境
    env = NavigationEnv(
        use_enhanced_state=enhanced_state,
        render_mode='human' if render else None,
        dynamic_speed_min=EnvConfig.DYNAMIC_OBS_VEL_MIN,
        dynamic_speed_max=EnvConfig.DYNAMIC_OBS_VEL_MAX,
    )
    
    # 加载模型
    agent = LSTMDdpgAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        history_len=5,
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
    
    # 可视化数据
    success_rates_over_time = []
    collision_rates_over_time = []
    trajectories = []  # 记录机器人轨迹
    
    # 测试循环
    print("开始测试...\n")
    for ep in range(n_episodes):
        state = env.reset()
        state_queue = deque([state.copy() for _ in range(5)], maxlen=5)
        done = False
        
        episode_reward = 0.0
        step_count = 0
        
        # 记录本episode的轨迹
        trajectory = []
        
        while not done:
            state_seq = np.asarray(state_queue, dtype=np.float32)
            action = agent.act(state_seq, step=0, add_noise=False)
            action_flat = action.reshape(-1)
            
            # 记录位置
            trajectory.append((env.robot.x, env.robot.y))
            
            next_state, reward, done, info = env.step(action_flat)
            state_queue.append(next_state.copy())
            
            episode_reward += reward
            step_count += 1
            
            # 慢速渲染
            if render and slow_render:
                time.sleep(0.05)
            
            if done:
                break
        
        # 保存轨迹
        trajectories.append({
            'path': trajectory,
            'success': info.get('reason') == 'goal_reached',
            'reason': info.get('reason', 'unknown')
        })
        
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
        
        # 计算累积统计
        current_success_rate = successes / (ep + 1)
        current_collision_rate = collisions / (ep + 1)
        success_rates_over_time.append(current_success_rate)
        collision_rates_over_time.append(current_collision_rate)
        
        # 周期性输出
        if (ep + 1) % 10 == 0:
            print(f"[Episode {ep+1:3d}/{n_episodes}] "
                  f"S:{successes:3d}/{ep+1} C:{collisions:3d}/{ep+1} T:{timeouts:3d}/{ep+1} | "
                  f"SR:{current_success_rate:.2%} CR:{current_collision_rate:.2%} TR:{timeouts/(ep+1):.2%}")
    
    # 计算最终统计
    success_rate = successes / n_episodes
    collision_rate = collisions / n_episodes
    timeout_rate = timeouts / n_episodes
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    # 打印最终结果
    print("\n" + "=" * 80)
    print("📊 测试结果汇总")
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
    print("\n📈 结果分布 (最近100轮):")
    recent_results = ''.join(episode_results[-100:])
    print(f"  {recent_results}")
    print(f"  S=成功, C=碰撞, T=超时")
    
    # 性能评估
    print("\n⭐ 性能评估:")
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
        print(f"  ✗ 碰撞率偏高 (>35%): {collision_rate:.2%}")
    
    env.close()
    
    # 生成可视化图表
    if save_plots:
        print(f"\n🎨 生成可视化图表...")
        plot_results(
            episode_rewards, 
            episode_steps, 
            success_rates_over_time,
            collision_rates_over_time,
            trajectories,
            model_path
        )
    
    return {
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'timeout_rate': timeout_rate,
        'avg_reward': avg_reward,
        'avg_steps': avg_steps,
        'episode_results': episode_results,
        'trajectories': trajectories,
    }


def plot_results(
    episode_rewards: List[float],
    episode_steps: List[int],
    success_rates: List[float],
    collision_rates: List[float],
    trajectories: List[Dict],
    model_name: str
):
    """生成可视化图表"""
    
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    
    # 创建2x3子图
    fig = plt.figure(figsize=(18, 10))
    
    # 1. 奖励曲线
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(episode_rewards, alpha=0.6, linewidth=0.5, color='blue')
    # 移动平均
    window = min(20, len(episode_rewards) // 5)
    if window > 1:
        rewards_ma = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), rewards_ma, 
                color='red', linewidth=2, label=f'{window}-episode MA')
        ax1.legend()
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.grid(True, alpha=0.3)
    
    # 2. 步数曲线
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(episode_steps, alpha=0.6, linewidth=0.5, color='green')
    if window > 1:
        steps_ma = np.convolve(episode_steps, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_steps)), steps_ma,
                color='darkgreen', linewidth=2, label=f'{window}-episode MA')
        ax2.legend()
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Steps')
    ax2.grid(True, alpha=0.3)
    
    # 3. 成功率/碰撞率变化
    ax3 = plt.subplot(2, 3, 3)
    episodes = range(1, len(success_rates) + 1)
    ax3.plot(episodes, success_rates, color='green', linewidth=2, label='Success Rate')
    ax3.plot(episodes, collision_rates, color='red', linewidth=2, label='Collision Rate')
    ax3.axhline(y=0.65, color='green', linestyle='--', alpha=0.5, label='65% Target')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Rate')
    ax3.set_title('Success & Collision Rates Over Time')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # 4. 轨迹热力图
    ax4 = plt.subplot(2, 3, 4)
    plot_trajectory_heatmap(ax4, trajectories)
    
    # 5. 结果分布饼图
    ax5 = plt.subplot(2, 3, 5)
    success_count = sum(1 for t in trajectories if t['success'])
    collision_count = sum(1 for t in trajectories if 'collision' in t['reason'])
    timeout_count = len(trajectories) - success_count - collision_count
    
    colors = ['#4CAF50', '#F44336', '#FFC107']
    labels = [f'Success\n{success_count}', f'Collision\n{collision_count}', f'Timeout\n{timeout_count}']
    sizes = [success_count, collision_count, timeout_count]
    
    wedges, texts, autotexts = ax5.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                                        startangle=90, textprops={'fontsize': 11})
    ax5.set_title('Result Distribution')
    
    # 6. 统计信息
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # 计算统计
    final_success_rate = success_count / len(trajectories)
    final_collision_rate = collision_count / len(trajectories)
    avg_reward = np.mean(episode_rewards)
    avg_steps = np.mean(episode_steps)
    
    stats_text = f"""
    📊 Test Statistics
    {'='*30}
    
    Total Episodes:     {len(trajectories)}
    
    Success Rate:       {final_success_rate:.1%}
    Collision Rate:     {final_collision_rate:.1%}
    Timeout Rate:       {timeout_count/len(trajectories):.1%}
    
    Avg Reward:         {avg_reward:.2f}
    Reward Std:         {np.std(episode_rewards):.2f}
    
    Avg Steps:          {avg_steps:.2f}
    Steps Std:          {np.std(episode_steps):.2f}
    
    Min Steps:          {np.min(episode_steps)}
    Max Steps:          {np.max(episode_steps)}
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Configuration:
    
    Obstacle Speed:     {EnvConfig.DYNAMIC_OBS_VEL_MIN:.2f}-{EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f} m/s
    UAV Speed:          {EnvConfig.MAX_LINEAR_VEL:.2f} m/s
    Speed Ratio:        {EnvConfig.MAX_LINEAR_VEL/EnvConfig.DYNAMIC_OBS_VEL_MAX:.2f}:1
    """
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图表
    save_name = os.path.basename(model_name).replace('.pth', '')
    save_path = os.path.join(RESULT_DIR, f'test_results_{save_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  ✓ 图表已保存: {save_path}")
    
    plt.close()


def plot_trajectory_heatmap(ax, trajectories: List[Dict]):
    """绘制轨迹热力图"""
    
    # 创建网格
    map_size = (EnvConfig.MAP_WIDTH, EnvConfig.MAP_HEIGHT)
    grid_size = 50
    heatmap = np.zeros((grid_size, grid_size))
    
    # 统计每个网格的访问次数
    for traj_data in trajectories:
        for x, y in traj_data['path']:
            i = int(x / map_size[0] * grid_size)
            j = int(y / map_size[1] * grid_size)
            if 0 <= i < grid_size and 0 <= j < grid_size:
                heatmap[j, i] += 1
    
    # 绘制热力图
    im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear', 
                   origin='lower', extent=[0, map_size[0], 0, map_size[1]])
    
    # 绘制成功和失败的终点
    for traj_data in trajectories:
        if len(traj_data['path']) > 0:
            end_x, end_y = traj_data['path'][-1]
            if traj_data['success']:
                ax.scatter(end_x, end_y, c='lime', s=20, alpha=0.6, marker='o', edgecolors='white', linewidths=0.5)
            else:
                ax.scatter(end_x, end_y, c='red', s=20, alpha=0.6, marker='x', linewidths=1)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Trajectory Heatmap\n(Green=Success, Red=Failure)')
    ax.set_xlim([0, map_size[0]])
    ax.set_ylim([0, map_size[1]])
    
    # 添加颜色条
    plt.colorbar(im, ax=ax, label='Visit Count')


def compare_models(model1_path: str, model2_path: str, n_episodes: int = 100) -> None:
    """对比两个模型的性能"""
    
    print("\n" + "=" * 80)
    print("🔬 模型对比测试")
    print("=" * 80)
    print(f"模型1: {model1_path}")
    print(f"模型2: {model2_path}")
    print("=" * 80 + "\n")
    
    print("测试模型1...")
    results1 = test_model_with_visualization(model1_path, n_episodes, render=False, save_plots=False)
    
    print("\n" + "-" * 80 + "\n")
    
    print("测试模型2...")
    results2 = test_model_with_visualization(model2_path, n_episodes, render=False, save_plots=False)
    
    # 对比结果
    print("\n" + "=" * 80)
    print("📊 对比结果")
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
    parser = argparse.ArgumentParser(description="V6 Model Testing with Visualization")
    parser.add_argument("--model", type=str, required=True, 
                       help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of test episodes (default: 100)")
    parser.add_argument("--render", action="store_true",
                       help="Enable real-time rendering")
    parser.add_argument("--slow", action="store_true",
                       help="Slow down rendering for observation")
    parser.add_argument("--save_plots", action="store_true", default=True,
                       help="Save visualization plots (default: True)")
    parser.add_argument("--no_plots", action="store_true",
                       help="Disable plot generation")
    parser.add_argument("--compare", type=str, default=None,
                       help="Path to second model for comparison")
    parser.add_argument("--legacy_state", action="store_true",
                       help="Use legacy state representation")
    
    args = parser.parse_args()
    
    # 处理save_plots参数
    save_plots = args.save_plots and not args.no_plots
    
    if args.compare:
        # 对比模式
        compare_models(args.model, args.compare, args.episodes)
    else:
        # 单模型测试
        test_model_with_visualization(
            args.model, 
            args.episodes, 
            render=args.render,
            slow_render=args.slow,
            save_plots=save_plots,
            enhanced_state=(not args.legacy_state)
        )


if __name__ == "__main__":
    main()
