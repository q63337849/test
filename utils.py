#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
工具函数
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


class Logger:
    """训练日志记录器"""
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.writer = None
        self.headers = None
        
    def log(self, data_dict):
        """记录一条数据"""
        if self.file is None:
            self.headers = list(data_dict.keys())
            self.file = open(self.filepath, 'w', newline='')
            self.writer = csv.DictWriter(self.file, fieldnames=self.headers)
            self.writer.writeheader()
        
        self.writer.writerow(data_dict)
        self.file.flush()
    
    def close(self):
        if self.file:
            self.file.close()


def plot_training_curves(rewards, steps, save_path=None, window=100):
    """绘制训练曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    episodes = np.arange(1, len(rewards) + 1)
    
    # 1. 奖励曲线
    ax1 = axes[0, 0]
    ax1.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
    if len(rewards) >= window:
        avg_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(episodes[window-1:], avg_rewards, color='red', linewidth=2, 
                 label=f'{window}-Episode Average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 步数曲线
    ax2 = axes[0, 1]
    ax2.plot(episodes, steps, alpha=0.3, color='green', label='Episode Steps')
    if len(steps) >= window:
        avg_steps = np.convolve(steps, np.ones(window)/window, mode='valid')
        ax2.plot(episodes[window-1:], avg_steps, color='orange', linewidth=2,
                 label=f'{window}-Episode Average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 奖励分布
    ax3 = axes[1, 0]
    ax3.hist(rewards, bins=50, color='blue', alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.2f}')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Reward Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 成功率曲线 (假设奖励>0为成功)
    ax4 = axes[1, 1]
    success = np.array(rewards) > 100  # 假设完成任务奖励>100
    if len(success) >= window:
        success_rate = np.convolve(success.astype(float), np.ones(window)/window, mode='valid') * 100
        ax4.plot(episodes[window-1:], success_rate, color='purple', linewidth=2)
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Success Rate (%)')
    ax4.set_title(f'{window}-Episode Success Rate')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 100])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def load_training_log(filepath):
    """加载训练日志"""
    data = {}
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key not in data:
                    data[key] = []
                try:
                    data[key].append(float(value))
                except ValueError:
                    data[key].append(value)
    return data


def moving_average(data, window):
    """计算移动平均"""
    return np.convolve(data, np.ones(window)/window, mode='valid')


def calculate_statistics(rewards, steps):
    """计算统计信息"""
    stats = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'max_reward': np.max(rewards),
        'min_reward': np.min(rewards),
        'mean_steps': np.mean(steps),
        'std_steps': np.std(steps),
    }
    return stats


# 测试代码
if __name__ == '__main__':
    # 测试绘图功能
    np.random.seed(42)
    
    # 模拟训练数据
    n_episodes = 500
    rewards = []
    steps = []
    
    for i in range(n_episodes):
        # 模拟逐渐改善的奖励
        base_reward = -200 + (200 / n_episodes) * i
        noise = np.random.normal(0, 50)
        rewards.append(base_reward + noise)
        
        # 模拟逐渐减少的步数
        base_steps = 500 - (300 / n_episodes) * i
        steps.append(int(max(50, base_steps + np.random.normal(0, 50))))
    
    # 绘制曲线
    plot_training_curves(rewards, steps)
    
    # 打印统计信息
    stats = calculate_statistics(rewards, steps)
    print("\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
