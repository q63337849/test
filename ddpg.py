#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG算法实现
基于原项目 ddpg.py，适配32线激光雷达的状态空间
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import DDPGConfig, EnvConfig

# 使用CUDA GPU（如果可用）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def add(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class OUNoise:
    """
    Ornstein-Uhlenbeck噪声
    用于连续动作空间的探索
    基于原项目实现
    """
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.2, min_sigma=0.05, decay_period=100000):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()
    
    def reset(self):
        """重置噪声状态"""
        self.state = copy.copy(self.mu)
    
    def sample(self, step=0):
        """采样噪声"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for _ in range(len(x))])
        self.state = x + dx
        
        # 噪声衰减
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, step / self.decay_period)
        
        return self.state


class Actor(nn.Module):
    """
    Actor网络
    输入: 状态
    输出: 动作 [线速度, 角速度]
    """
    def __init__(self, state_dim, action_dim, hidden_dim, max_lin_vel, max_ang_vel, init_w=3e-3):
        super(Actor, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)
        
        # 权重初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
        self.max_lin_vel = max_lin_vel
        self.max_ang_vel = max_ang_vel
    
    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        action = self.linear3(x)
        
        # 动作输出约束
        # 线速度: [0, max_lin_vel] 使用sigmoid
        # 角速度: [-max_ang_vel, max_ang_vel] 使用tanh
        action_out = torch.zeros_like(action)
        action_out[:, 0] = torch.sigmoid(action[:, 0]) * self.max_lin_vel
        action_out[:, 1] = torch.tanh(action[:, 1]) * self.max_ang_vel
        
        return action_out


class Critic(nn.Module):
    """
    Critic网络
    输入: 状态, 动作
    输出: Q值
    """
    def __init__(self, state_dim, action_dim, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        # 权重初始化
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        value = self.linear3(x)
        return value


class DDPGAgent:
    """
    DDPG智能体
    基于原项目的实现
    """
    def __init__(self, state_dim=None, action_dim=None, hidden_dim=None,
                 actor_lr=None, critic_lr=None, gamma=None, tau=None,
                 buffer_size=None, batch_size=None,
                 max_lin_vel=None, max_ang_vel=None):
        
        # 使用配置文件中的默认值
        self.state_dim = state_dim or DDPGConfig.STATE_DIM
        self.action_dim = action_dim or DDPGConfig.ACTION_DIM
        self.hidden_dim = hidden_dim or DDPGConfig.HIDDEN_DIM
        self.actor_lr = actor_lr or DDPGConfig.ACTOR_LR
        self.critic_lr = critic_lr or DDPGConfig.CRITIC_LR
        self.gamma = gamma or DDPGConfig.GAMMA
        self.tau = tau or DDPGConfig.TAU
        self.buffer_size = buffer_size or DDPGConfig.BUFFER_SIZE
        self.batch_size = batch_size or DDPGConfig.BATCH_SIZE
        self.max_lin_vel = max_lin_vel or EnvConfig.MAX_LINEAR_VEL
        self.max_ang_vel = max_ang_vel or EnvConfig.MAX_ANGULAR_VEL
        
        # Actor网络
        self.actor_local = Actor(self.state_dim, self.action_dim, self.hidden_dim,
                                  self.max_lin_vel, self.max_ang_vel).to(device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dim,
                                   self.max_lin_vel, self.max_ang_vel).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_lr)
        
        # Critic网络
        self.critic_local = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_lr)
        
        # 损失函数
        self.loss_function = nn.MSELoss()
        
        # 噪声
        self.noise = OUNoise(
            self.action_dim,
            mu=DDPGConfig.OU_MU,
            theta=DDPGConfig.OU_THETA,
            max_sigma=DDPGConfig.OU_SIGMA,
            min_sigma=DDPGConfig.OU_SIGMA_MIN,
            decay_period=DDPGConfig.OU_DECAY
        )
        
        # 经验回放
        self.memory = ReplayBuffer(self.buffer_size)
        
        # 硬更新目标网络
        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)
        
        # 训练统计
        self.actor_loss_history = []
        self.critic_loss_history = []
    
    def reset_noise(self):
        """重置噪声"""
        self.noise.reset()
    
    def act(self, state, step=0, add_noise=True):
        """
        选择动作
        state: 状态
        step: 当前步数（用于噪声衰减）
        add_noise: 是否添加探索噪声
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            noise = self.noise.sample(step)
            action += noise
        
        # 裁剪动作到有效范围
        action[0, 0] = np.clip(action[0, 0], 0.0, self.max_lin_vel)
        action[0, 1] = np.clip(action[0, 1], -self.max_ang_vel, self.max_ang_vel)
        
        return action
    
    def step(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self):
        """
        从经验回放中学习
        基于原项目的learn函数
        """
        if len(self.memory) < self.batch_size:
            return
        
        # 采样
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        state = torch.FloatTensor(states).to(device)
        next_state = torch.FloatTensor(next_states).to(device)
        action = torch.FloatTensor(actions).to(device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(device)
        
        # 修正action维度 (原项目中的处理)
        action = torch.squeeze(action, 1)
        
        # ========== 更新Critic ==========
        next_action = self.actor_target(next_state)
        target_Q = self.critic_target(next_state, next_action.detach())
        expected_Q = reward + (1.0 - done) * self.gamma * target_Q
        expected_Q = torch.clamp(expected_Q, -np.inf, np.inf)
        
        current_Q = self.critic_local(state, action)
        critic_loss = self.loss_function(current_Q, expected_Q.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ========== 更新Actor ==========
        actor_loss = -self.critic_local(state, self.actor_local(state)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ========== 软更新目标网络 ==========
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)
        
        # 记录损失
        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())
    
    def soft_update(self, local_model, target_model):
        """
        软更新目标网络
        θ_target = τ * θ_local + (1 - τ) * θ_target
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
    
    def hard_update(self, target_model, local_model):
        """硬更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)
    
    def save(self, filepath):
        """保存模型"""
        torch.save({
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath, strict=True, load_optimizers=True):
        """加载模型
        
        Args:
            filepath: 模型文件路径
            strict: 是否严格匹配网络结构（默认True）
            load_optimizers: 是否加载优化器状态（测试时可设为False）
        """
        # PyTorch 2.6+ 需要 weights_only=False
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=device)
            
        self.actor_local.load_state_dict(checkpoint['actor_local'], strict=strict)
        self.actor_target.load_state_dict(checkpoint['actor_target'], strict=strict)
        self.critic_local.load_state_dict(checkpoint['critic_local'], strict=strict)
        self.critic_target.load_state_dict(checkpoint['critic_target'], strict=strict)
        
        if load_optimizers:
            if 'actor_optimizer' in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            if 'critic_optimizer' in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        print(f"Model loaded from {filepath}")
    
    def save_actor(self, filepath):
        """只保存Actor模型（用于部署）"""
        torch.save(self.actor_local.state_dict(), filepath)
        
    def load_actor(self, filepath):
        """只加载Actor模型"""
        try:
            state_dict = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            state_dict = torch.load(filepath, map_location=device)
        self.actor_local.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)


# 测试代码
if __name__ == '__main__':
    # 创建agent
    agent = DDPGAgent()
    
    # 测试动作选择
    state = np.random.randn(DDPGConfig.STATE_DIM).astype(np.float32)
    action = agent.act(state, step=0, add_noise=True)
    print(f"State shape: {state.shape}")
    print(f"Action: {action}")
    print(f"Action shape: {action.shape}")
    
    # 测试存储和学习
    next_state = np.random.randn(DDPGConfig.STATE_DIM).astype(np.float32)
    reward = 1.0
    done = False
    
    for i in range(200):
        agent.step(state, action, reward, next_state, done)
    
    print(f"Memory size: {len(agent.memory)}")
    agent.learn()
    print("Learning step completed!")
