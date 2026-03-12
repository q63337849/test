#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM-DDPG 优化版：
1. UPDATE_EVERY - 不是每步都更新
2. 更小的网络规模 (64 vs 128)
3. 更短的历史长度 (5 vs 10)
4. 优化的 replay buffer 采样
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import DDPGConfig, EnvConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OUNoise:
    """Ornstein-Uhlenbeck 噪声"""

    def __init__(
        self,
        action_dim: int,
        mu: float = 0.0,
        theta: float = 0.15,
        max_sigma: float = 0.2,
        min_sigma: float = 0.05,
        decay_period: int = 100000,
    ):
        self.mu = mu * np.ones(action_dim, dtype=np.float32)
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self) -> None:
        self.state = copy.copy(self.mu)

    def sample(self, step: int = 0) -> np.ndarray:
        x = self.state
        # 修正：使用零均值高斯噪声
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x)).astype(np.float32)
        self.state = x + dx
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, float(step) / float(self.decay_period)
        )
        return self.state


@dataclass
class _Episode:
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]


class EpisodeSequenceReplayBuffer:
    """优化版 Episode Replay Buffer
    
    优化点：
    1. 使用 numpy 数组预分配
    2. 缓存已完成 episode 的索引
    """

    def __init__(self, capacity_transitions: int, state_dim: int, action_dim: int, history_len: int):
        self.capacity_transitions = int(capacity_transitions)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)

        self.episodes: Deque[_Episode] = deque()
        self._current: Optional[_Episode] = None
        self._num_transitions: int = 0
        
        # 缓存：(episode_idx, transition_idx) 的扁平列表，加速采样
        self._transition_index: List[Tuple[int, int]] = []
        self._index_dirty = True

    def __len__(self) -> int:
        return self._num_transitions

    def _start_episode_if_needed(self, first_state: np.ndarray) -> None:
        if self._current is None:
            self._current = _Episode(states=[first_state.copy()], actions=[], rewards=[], dones=[])

    def add(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        state = np.asarray(state, dtype=np.float32)
        next_state = np.asarray(next_state, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        self._start_episode_if_needed(state)
        assert self._current is not None

        self._current.actions.append(action.copy())
        self._current.rewards.append(float(reward))
        self._current.dones.append(bool(done))
        self._current.states.append(next_state.copy())

        self._num_transitions += 1

        while self._num_transitions > self.capacity_transitions and len(self.episodes) > 0:
            old = self.episodes.popleft()
            self._num_transitions -= len(old.actions)
            self._index_dirty = True

        if done:
            self.episodes.append(self._current)
            self._current = None
            self._index_dirty = True

    def _rebuild_index(self) -> None:
        """重建 transition 索引"""
        if not self._index_dirty:
            return
        self._transition_index = []
        for ep_idx, ep in enumerate(self.episodes):
            for t_idx in range(len(ep.actions)):
                self._transition_index.append((ep_idx, t_idx))
        self._index_dirty = False

    def _pad_left(self, seq: List[np.ndarray], pad_value: np.ndarray) -> np.ndarray:
        if len(seq) >= self.history_len:
            return np.stack(seq[-self.history_len:], axis=0)
        pad_n = self.history_len - len(seq)
        pads = [pad_value.copy() for _ in range(pad_n)]
        return np.stack(pads + seq, axis=0)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """优化采样：使用预建索引"""
        if len(self.episodes) == 0:
            raise RuntimeError("ReplayBuffer has no completed episodes yet.")

        self._rebuild_index()
        
        if len(self._transition_index) < batch_size:
            indices = self._transition_index.copy()
        else:
            indices = random.sample(self._transition_index, batch_size)

        batch_state_seq = []
        batch_next_state_seq = []
        batch_actions = []
        batch_rewards = []
        batch_dones = []

        episodes_list = list(self.episodes)
        
        for ep_idx, t in indices:
            ep = episodes_list[ep_idx]
            
            # state_seq ends at state_t
            start = max(0, t - self.history_len + 1)
            state_seq = self._pad_left(ep.states[start:t + 1], ep.states[0])

            # next_state_seq ends at state_{t+1}
            t2 = t + 1
            start2 = max(0, t2 - self.history_len + 1)
            next_state_seq = self._pad_left(ep.states[start2:t2 + 1], ep.states[0])

            batch_state_seq.append(state_seq)
            batch_next_state_seq.append(next_state_seq)
            batch_actions.append(ep.actions[t])
            batch_rewards.append(ep.rewards[t])
            batch_dones.append(ep.dones[t])

        return (
            np.asarray(batch_state_seq, dtype=np.float32),
            np.asarray(batch_actions, dtype=np.float32),
            np.asarray(batch_rewards, dtype=np.float32),
            np.asarray(batch_next_state_seq, dtype=np.float32),
            np.asarray(batch_dones, dtype=np.float32),
        )


class LSTMActor(nn.Module):
    """LSTM-Actor（优化版：更小网络）"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 64,
        lstm_hidden_dim: int = 64,
        mlp_hidden_dim: int = 256,
        max_lin_vel: float = 0.22,
        max_ang_vel: float = 2.0,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc_out = nn.Linear(mlp_hidden_dim, action_dim)

        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_in(state_seq))
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        action = self.fc_out(h)

        action_out = torch.zeros_like(action)
        action_out[:, 0] = torch.sigmoid(action[:, 0]) * self.max_lin_vel
        action_out[:, 1] = torch.tanh(action[:, 1]) * self.max_ang_vel
        return action_out


class LSTMCritic(nn.Module):
    """LSTM-Critic（优化版：更小网络）"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 64,
        lstm_hidden_dim: int = 64,
        mlp_hidden_dim: int = 256,
        init_w: float = 3e-3,
    ):
        super().__init__()
        self.fc_in = nn.Linear(state_dim, embed_dim)
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(lstm_hidden_dim + action_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc_out = nn.Linear(mlp_hidden_dim, 1)

        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc_in(state_seq))
        out, _ = self.lstm(x)
        h = out[:, -1, :]
        x = torch.cat([h, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class LSTMDdpgAgent:
    """LSTM-DDPG Agent（优化版）
    
    关键优化：
    1. UPDATE_EVERY: 不是每步都更新
    2. 更小的网络 (embed_dim=64, lstm_hidden_dim=64)
    3. 更短的历史 (history_len=5)
    """

    def __init__(
        self,
        state_dim: int = DDPGConfig.STATE_DIM,
        action_dim: int = DDPGConfig.ACTION_DIM,
        hidden_dim: int = DDPGConfig.HIDDEN_DIM,
        actor_lr: float = DDPGConfig.ACTOR_LR,
        critic_lr: float = DDPGConfig.CRITIC_LR,
        gamma: float = DDPGConfig.GAMMA,
        tau: float = DDPGConfig.TAU,
        buffer_size: int = DDPGConfig.BUFFER_SIZE,
        batch_size: int = DDPGConfig.BATCH_SIZE,
        history_len: int = DDPGConfig.HISTORY_LEN,
        embed_dim: int = DDPGConfig.LSTM_EMBED_DIM,
        lstm_hidden_dim: int = DDPGConfig.LSTM_HIDDEN_DIM,
        max_lin_vel: float = EnvConfig.MAX_LINEAR_VEL,
        max_ang_vel: float = EnvConfig.MAX_ANGULAR_VEL,
        grad_clip_norm: float = DDPGConfig.GRAD_CLIP_NORM,
        update_every: int = DDPGConfig.UPDATE_EVERY,
        update_times: int = DDPGConfig.UPDATE_TIMES,
    ):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)
        self.history_len = int(history_len)
        self.grad_clip_norm = float(grad_clip_norm)
        self.update_every = int(update_every)
        self.update_times = int(update_times)

        self.embed_dim = int(embed_dim)
        self.lstm_hidden_dim = int(lstm_hidden_dim)
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)

        self.state_meta = {}
        
        # 步数计数器
        self._step_count = 0

        # Networks
        self.actor_local = LSTMActor(
            self.state_dim, self.action_dim,
            embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
            max_lin_vel=max_lin_vel, max_ang_vel=max_ang_vel,
        ).to(device)
        self.actor_target = LSTMActor(
            self.state_dim, self.action_dim,
            embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
            max_lin_vel=max_lin_vel, max_ang_vel=max_ang_vel,
        ).to(device)

        self.critic_local = LSTMCritic(
            self.state_dim, self.action_dim,
            embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
        ).to(device)
        self.critic_target = LSTMCritic(
            self.state_dim, self.action_dim,
            embed_dim=embed_dim, lstm_hidden_dim=lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
        ).to(device)

        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=critic_lr)
        self.loss_function = nn.MSELoss()

        self.noise = OUNoise(
            self.action_dim,
            mu=DDPGConfig.OU_MU,
            theta=DDPGConfig.OU_THETA,
            max_sigma=DDPGConfig.OU_SIGMA,
            min_sigma=DDPGConfig.OU_SIGMA_MIN,
            decay_period=DDPGConfig.OU_DECAY,
        )

        self.memory = EpisodeSequenceReplayBuffer(
            capacity_transitions=buffer_size,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            history_len=self.history_len,
        )

        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        self.actor_loss_history: List[float] = []
        self.critic_loss_history: List[float] = []

    def reset_noise(self) -> None:
        self.noise.reset()

    @torch.no_grad()
    def act(self, state_seq: np.ndarray, step: int = 0, add_noise: bool = True) -> np.ndarray:
        state_seq = np.asarray(state_seq, dtype=np.float32)
        if state_seq.ndim == 2:
            state_seq_t = torch.from_numpy(state_seq).float().unsqueeze(0).to(device)
        else:
            state_seq_t = torch.from_numpy(state_seq).float().to(device)

        self.actor_local.eval()
        action = self.actor_local(state_seq_t).cpu().numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample(step)

        action[0, 0] = np.clip(action[0, 0], 0.0, self.max_lin_vel)
        action[0, 1] = np.clip(action[0, 1], -self.max_ang_vel, self.max_ang_vel)
        return action

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self._step_count += 1

    def should_learn(self) -> bool:
        """检查是否应该学习（基于 UPDATE_EVERY）"""
        return (
            self._step_count % self.update_every == 0 and
            len(self.memory) >= self.batch_size and
            len(self.memory.episodes) > 0
        )

    def learn(self) -> None:
        """执行一次学习更新"""
        if len(self.memory) < self.batch_size or len(self.memory.episodes) == 0:
            return

        for _ in range(self.update_times):
            self._learn_once()

    def _learn_once(self) -> None:
        states_seq, actions, rewards, next_states_seq, dones = self.memory.sample(self.batch_size)

        state_seq = torch.from_numpy(states_seq).float().to(device)
        next_state_seq = torch.from_numpy(next_states_seq).float().to(device)
        action = torch.from_numpy(actions).float().to(device)
        reward = torch.from_numpy(rewards).float().unsqueeze(1).to(device)
        done = torch.from_numpy(dones).float().unsqueeze(1).to(device)

        # Critic update
        with torch.no_grad():
            next_action = self.actor_target(next_state_seq)
            target_q = self.critic_target(next_state_seq, next_action)
            expected_q = reward + (1.0 - done) * self.gamma * target_q

        current_q = self.critic_local(state_seq, action)
        critic_loss = self.loss_function(current_q, expected_q)

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), self.grad_clip_norm)
        self.critic_optimizer.step()

        # Actor update
        actor_loss = -self.critic_local(state_seq, self.actor_local(state_seq)).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), self.grad_clip_norm)
        self.actor_optimizer.step()

        # Soft update
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)

        self.actor_loss_history.append(float(actor_loss.item()))
        self.critic_loss_history.append(float(critic_loss.item()))

    def soft_update(self, local_model: nn.Module, target_model: nn.Module) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    @staticmethod
    def hard_update(target_model: nn.Module, local_model: nn.Module) -> None:
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "actor_local": self.actor_local.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_local": self.critic_local.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "history_len": self.history_len,
                "net_cfg": {
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim,
                    "mlp_hidden_dim": self.hidden_dim,
                    "embed_dim": self.embed_dim,
                    "lstm_hidden_dim": self.lstm_hidden_dim,
                    "max_lin_vel": self.max_lin_vel,
                    "max_ang_vel": self.max_ang_vel,
                },
                "state_meta": getattr(self, "state_meta", {}),
            },
            filepath,
        )

    def load(self, filepath: str, strict: bool = True, load_optimizers: bool = True) -> dict:
        """加载模型检查点
        
        Args:
            filepath: 模型文件路径
            strict: 是否严格匹配网络结构（默认True）
            load_optimizers: 是否加载优化器状态（测试时可设为False）
        
        Returns:
            checkpoint dict
        """
        # PyTorch 2.6+ 需要 weights_only=False
        try:
            checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(filepath, map_location=device)

        actor_sd = checkpoint.get("actor_local", {})
        if "fc_in.weight" in actor_sd:
            ckpt_state_dim = int(actor_sd["fc_in.weight"].shape[1])
        else:
            ckpt_state_dim = checkpoint.get("net_cfg", {}).get("state_dim", None)

        if ckpt_state_dim is not None and int(ckpt_state_dim) != int(self.state_dim):
            raise RuntimeError(
                f"Checkpoint state_dim={ckpt_state_dim} != current {self.state_dim}. "
                "Check your state config flags match the checkpoint."
            )

        self.actor_local.load_state_dict(checkpoint["actor_local"], strict=strict)
        self.actor_target.load_state_dict(checkpoint["actor_target"], strict=strict)
        self.critic_local.load_state_dict(checkpoint["critic_local"], strict=strict)
        self.critic_target.load_state_dict(checkpoint["critic_target"], strict=strict)
        
        if load_optimizers:
            if "actor_optimizer" in checkpoint:
                self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            if "critic_optimizer" in checkpoint:
                self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

        if "state_meta" in checkpoint:
            self.state_meta = checkpoint.get("state_meta", {}) or {}
        return checkpoint
