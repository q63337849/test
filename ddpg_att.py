#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDPG + Spatial Attention 算法实现 (V3 — 真正的多头注意力)
修复要点：
  1. 消除重复类定义 (旧版 Actor/Critic 在文件中出现两次)
  2. SpatialSectorAttentionV2 → V3: 真正的多头注意力 + 丰富 token 特征
  3. Token 从 2D [range, diff] 升级为 6D [range, valid, diff, closing, pos_sin, pos_cos]
  4. Pre-LayerNorm + 残差 + FFN 提升训练稳定性
  5. 硬掩码排除无回波扇区 (all-invalid 时回退均匀权重)
  6. 可学习温度 (clamped 0.3-2.0) 实现锐利/平滑注意力自适应
  7. 保持与 train_ddpg_att_sr100_logmeta_seed.py 完全兼容的接口
"""

import copy
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import DDPGConfig, EnvConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ========================== Replay Buffer ==========================

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


# ========================== OU Noise ==========================

class OUNoise:
    """Ornstein-Uhlenbeck 噪声"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, max_sigma=0.2,
                 min_sigma=0.05, decay_period=100000):
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_dim
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self, step=0):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array(
            [random.random() for _ in range(len(x))])
        self.state = x + dx
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, step / self.decay_period)
        return self.state


# ========================== State Layout ==========================

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StateLayout:
    """Enhanced state layout inference."""
    n_sectors: int
    sectors_slice: slice
    lidar_diff_slice: Optional[slice]
    non_lidar_slice: slice
    delta_yaw_index: Optional[int]

    @property
    def non_lidar_dim(self) -> int:
        return self.non_lidar_slice.stop - self.non_lidar_slice.start

    @property
    def has_lidar_diff(self) -> bool:
        return self.lidar_diff_slice is not None

    @property
    def has_delta_yaw(self) -> bool:
        return self.delta_yaw_index is not None


def infer_state_layout(state_dim: int) -> StateLayout:
    """Infer enhanced-state layout from state_dim."""
    for n in (16, 8, 32):
        enhanced_dim = n + n + 3 + 2 + 2 + 1 + 1 + 2
        enhanced_dim_no_dyaw = n + n + 3 + 2 + 2 + 1 + 2
        if state_dim == enhanced_dim:
            return StateLayout(
                n, slice(0, n), slice(n, 2 * n), slice(2 * n, state_dim),
                2 * n + (3 + 2 + 2 + 1))
        if state_dim == enhanced_dim_no_dyaw:
            return StateLayout(
                n, slice(0, n), slice(n, 2 * n), slice(2 * n, state_dim), None)

    n_guess = getattr(EnvConfig, "N_SECTORS", None)
    if isinstance(n_guess, int) and 0 < n_guess < state_dim:
        n = n_guess
    else:
        n = 16 if state_dim > 16 else max(1, state_dim // 2)
    return StateLayout(n, slice(0, n), None, slice(n, state_dim), None)


def _split_flat_state(layout: StateLayout, state: torch.Tensor):
    """Split flat state (B,D) → sectors, lidar_diff, non_lidar_wo_dyaw, dyaw, other_feat."""
    sectors = state[:, layout.sectors_slice]
    lidar_diff = state[:, layout.lidar_diff_slice] if layout.lidar_diff_slice else None
    non_lidar = state[:, layout.non_lidar_slice]

    if layout.delta_yaw_index is not None:
        idx = layout.delta_yaw_index - layout.non_lidar_slice.start
        dyaw = non_lidar[:, idx:idx + 1]
        non_lidar_wo = torch.cat([non_lidar[:, :idx], non_lidar[:, idx + 1:]], dim=-1)
        other_feat = torch.cat([non_lidar_wo, dyaw], dim=-1)
    else:
        dyaw = None
        non_lidar_wo = non_lidar
        other_feat = non_lidar
    return sectors, lidar_diff, non_lidar_wo, dyaw, other_feat


# ========================== Multi-Head Spatial Attention V3 ==========================

class SpatialAttentionV3(nn.Module):
    """真正的多头空间注意力 — 专为单步 DDPG 设计.

    vs V2 (旧版):
      - V2: heads 参数被忽略, 实际只有 1 个 query → 注意力表达力极弱
      - V3: 真正的 H 个注意力头, 每个头可独立聚焦不同方向/障碍

    Token 特征 (6D):
      [range, valid_conf, diff, closing_speed, pos_sin, pos_cos]
      - valid_conf: 1.0=近处有回波, 0.0=远处/无回波
      - closing_speed: clamp(diff, -1, 0) — 只关注"正在逼近"的信号
      - pos_sin/cos: 扇区方向的位置编码

    稳定性机制:
      - Pre-LayerNorm (token + query)
      - 硬掩码: 无回波扇区从 softmax 中排除 (all-invalid → 回退均匀权重)
      - 可学习温度: 控制注意力锐利度 (clamped 0.3-2.0)
      - 残差连接: attention_output + mean_pooling → 防止梯度消失
      - Post FFN + LayerNorm
    """
    TOKEN_DIM = 6  # [range, valid, diff, closing, pos_sin, pos_cos]

    def __init__(self, n_sectors: int, non_lidar_query_dim: int,
                 model_dim: int = 32, heads: int = 4,
                 dropout: float = 0.0, no_echo_thresh: float = 0.95):
        super().__init__()
        self.n_sectors = n_sectors
        self.model_dim = model_dim
        self.heads = heads
        self.head_dim = model_dim // heads
        assert model_dim % heads == 0, f"model_dim({model_dim}) must be divisible by heads({heads})"
        self.no_echo_thresh = no_echo_thresh

        # Token projection: 6D → model_dim
        self.token_proj = nn.Sequential(
            nn.Linear(self.TOKEN_DIM, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.token_ln = nn.LayerNorm(model_dim)

        # Query projection: non_lidar → model_dim
        self.query_proj = nn.Sequential(
            nn.Linear(non_lidar_query_dim, model_dim),
            nn.ReLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.query_ln = nn.LayerNorm(model_dim)

        # Learnable temperature (one per head)
        self.log_temp = nn.Parameter(torch.zeros(heads))  # init temp=1.0

        # Output projection (multi-head concat → model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

        # Mean-pool residual projection
        self.mean_proj = nn.Linear(model_dim, model_dim)

        # FFN + LayerNorm
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * 2),
            nn.ReLU(),
            nn.Linear(model_dim * 2, model_dim),
        )
        self.out_ln = nn.LayerNorm(model_dim)

        self.dropout = nn.Dropout(dropout)

        # Pre-compute positional encoding (fixed)
        angles = torch.arange(n_sectors, dtype=torch.float32) * (
            2.0 * math.pi / n_sectors)
        self.register_buffer('pos_sin', torch.sin(angles))  # (N,)
        self.register_buffer('pos_cos', torch.cos(angles))  # (N,)

    def forward(self, sectors: torch.Tensor, lidar_diff: Optional[torch.Tensor],
                non_lidar: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sectors: (B, N) range values [0,1], 1.0 = max range / no echo
            lidar_diff: (B, N) or None — temporal difference
            non_lidar: (B, Dq) — non-lidar features (query source)
        Returns:
            (B, model_dim) — spatial attention embedding
        """
        B, N = sectors.shape
        assert N == self.n_sectors

        # === Build 6D token features ===
        valid = (sectors < self.no_echo_thresh).float()          # (B,N)
        diff = lidar_diff if lidar_diff is not None else torch.zeros_like(sectors)
        closing = torch.clamp(diff, -1.0, 0.0)                  # only approach signal

        pos_sin = self.pos_sin.unsqueeze(0).expand(B, -1)       # (B,N)
        pos_cos = self.pos_cos.unsqueeze(0).expand(B, -1)       # (B,N)

        # (B, N, 6)
        tokens_raw = torch.stack([sectors, valid, diff, closing, pos_sin, pos_cos], dim=-1)

        # === Project tokens and query ===
        tokens = self.token_ln(self.token_proj(tokens_raw))      # (B, N, D)
        query = self.query_ln(self.query_proj(non_lidar))        # (B, D)

        # === Multi-head attention ===
        H = self.heads
        d_h = self.head_dim

        # (B, N, H, d_h) → (B, H, N, d_h)
        t_mh = tokens.view(B, N, H, d_h).transpose(1, 2)
        # (B, H, d_h) → (B, H, 1, d_h)
        q_mh = query.view(B, H, d_h).unsqueeze(2)

        # Scaled dot-product scores with learnable temperature
        temp = torch.clamp(self.log_temp.exp(), 0.3, 2.0)       # (H,)
        scale = (d_h ** 0.5) * temp.view(1, H, 1)               # (1,H,1)
        scores = (q_mh * t_mh).sum(dim=-1) / scale              # (B,H,N)

        # === Hard mask: exclude invalid tokens ===
        hard_mask = (sectors < self.no_echo_thresh)              # (B,N) bool
        hard_mask_mh = hard_mask.unsqueeze(1).expand(-1, H, -1) # (B,H,N)
        any_valid = hard_mask_mh.any(dim=-1, keepdim=True)      # (B,H,1)

        scores = scores.masked_fill(~hard_mask_mh, -1e9)
        weights = torch.softmax(scores, dim=-1)                  # (B,H,N)

        # Fallback: all invalid → uniform
        uniform = torch.ones(B, H, N, device=sectors.device) / N
        weights = torch.where(any_valid.expand_as(weights), weights, uniform)
        weights = self.dropout(weights)

        # Weighted sum: (B,H,d_h)
        attended = torch.einsum('bhn,bhnd->bhd', weights, t_mh)
        attended = attended.reshape(B, self.model_dim)           # concat heads
        attended = self.out_proj(attended)                        # (B,D)

        # === Residual from mean pooling ===
        mean_pool = tokens.mean(dim=1)                           # (B,D)
        x = attended + self.mean_proj(mean_pool)

        # === FFN + LayerNorm ===
        x = self.out_ln(x + self.ffn(self.dropout(x)))

        return x


# ========================== Actor / Critic ==========================

class ActorWithAttention(nn.Module):
    """Actor: SpatialAttentionV3 + non_lidar MLP → action."""
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 n_sectors=16, sector_embed_dim=32, spatial_att_heads=4,
                 att_dropout=0.0,
                 max_lin_vel: float = 1.0, max_ang_vel: float = 0.8):
        super().__init__()
        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)

        self.layout = infer_state_layout(int(state_dim))
        self.n_sectors = self.layout.n_sectors

        query_dim = self.layout.non_lidar_dim - (1 if self.layout.has_delta_yaw else 0)

        self.spatial_att = SpatialAttentionV3(
            n_sectors=self.n_sectors,
            non_lidar_query_dim=query_dim,
            model_dim=sector_embed_dim,
            heads=spatial_att_heads,
            dropout=att_dropout,
        )

        other_dim = self.layout.non_lidar_dim
        self.other_fc = nn.Sequential(
            nn.Linear(other_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(hidden_dim + sector_embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state):
        sectors, lidar_diff, non_lidar_wo, dyaw, other_feat = _split_flat_state(
            self.layout, state)

        att_feat = self.spatial_att(sectors, lidar_diff, non_lidar_wo)
        other_h = self.other_fc(other_feat)

        x = torch.cat([att_feat, other_h], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        raw = self.fc3(x)

        lin = torch.sigmoid(raw[:, 0:1]) * self.max_lin_vel
        ang = torch.tanh(raw[:, 1:2]) * self.max_ang_vel
        return torch.cat([lin, ang], dim=1)


class CriticWithAttention(nn.Module):
    """Critic: SpatialAttentionV3 + non_lidar + action → Q."""
    def __init__(self, state_dim, action_dim, hidden_dim=256,
                 n_sectors=16, sector_embed_dim=32, spatial_att_heads=4,
                 att_dropout=0.0,
                 max_lin_vel: float = 1.0, max_ang_vel: float = 0.8):
        super().__init__()

        self.layout = infer_state_layout(int(state_dim))
        self.n_sectors = self.layout.n_sectors

        query_dim = self.layout.non_lidar_dim - (1 if self.layout.has_delta_yaw else 0)

        self.spatial_att = SpatialAttentionV3(
            n_sectors=self.n_sectors,
            non_lidar_query_dim=query_dim,
            model_dim=sector_embed_dim,
            heads=spatial_att_heads,
            dropout=att_dropout,
        )

        other_dim = self.layout.non_lidar_dim
        self.other_fc = nn.Sequential(
            nn.Linear(other_dim, hidden_dim),
            nn.ReLU(),
        )

        self.fc1 = nn.Linear(hidden_dim + sector_embed_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state, action):
        sectors, lidar_diff, non_lidar_wo, dyaw, other_feat = _split_flat_state(
            self.layout, state)

        att_feat = self.spatial_att(sectors, lidar_diff, non_lidar_wo)
        other_h = self.other_fc(other_feat)

        x = torch.cat([att_feat, other_h, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ========================== DDPG Agent ==========================

class DDPGAttentionAgent:
    """DDPG + 空间注意力智能体 (接口与训练脚本完全兼容)."""
    def __init__(self, state_dim=None, action_dim=None, hidden_dim=None,
                 n_sectors=16, sector_embed_dim=32, spatial_att_heads=4,
                 att_dropout=0.0,
                 actor_lr=None, critic_lr=None, gamma=None, tau=None,
                 buffer_size=None, batch_size=None,
                 max_lin_vel=None, max_ang_vel=None):

        self.state_dim = state_dim or DDPGConfig.STATE_DIM
        self.action_dim = action_dim or DDPGConfig.ACTION_DIM
        self.hidden_dim = hidden_dim or DDPGConfig.HIDDEN_DIM
        self.n_sectors = n_sectors
        self.sector_embed_dim = sector_embed_dim
        self.spatial_att_heads = spatial_att_heads
        self.att_dropout = att_dropout

        self.actor_lr = actor_lr or DDPGConfig.ACTOR_LR
        self.critic_lr = critic_lr or DDPGConfig.CRITIC_LR
        self.gamma = gamma or DDPGConfig.GAMMA
        self.tau = tau or DDPGConfig.TAU
        self.buffer_size = buffer_size or DDPGConfig.BUFFER_SIZE
        self.batch_size = batch_size or DDPGConfig.BATCH_SIZE
        self.max_lin_vel = max_lin_vel or EnvConfig.MAX_LINEAR_VEL
        self.max_ang_vel = max_ang_vel or EnvConfig.MAX_ANGULAR_VEL

        _layout = infer_state_layout(self.state_dim)

        # Actor
        self.actor_local = ActorWithAttention(
            self.state_dim, self.action_dim, self.hidden_dim,
            self.n_sectors, self.sector_embed_dim, self.spatial_att_heads,
            self.att_dropout, self.max_lin_vel, self.max_ang_vel
        ).to(device)
        self.actor_target = ActorWithAttention(
            self.state_dim, self.action_dim, self.hidden_dim,
            self.n_sectors, self.sector_embed_dim, self.spatial_att_heads,
            self.att_dropout, self.max_lin_vel, self.max_ang_vel
        ).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=self.actor_lr)

        # Critic
        self.critic_local = CriticWithAttention(
            self.state_dim, self.action_dim, self.hidden_dim,
            self.n_sectors, self.sector_embed_dim, self.spatial_att_heads,
            self.att_dropout
        ).to(device)
        self.critic_target = CriticWithAttention(
            self.state_dim, self.action_dim, self.hidden_dim,
            self.n_sectors, self.sector_embed_dim, self.spatial_att_heads,
            self.att_dropout
        ).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.critic_lr)

        self.loss_function = nn.MSELoss()

        # Noise
        self.noise = OUNoise(
            self.action_dim,
            mu=DDPGConfig.OU_MU,
            theta=DDPGConfig.OU_THETA,
            max_sigma=DDPGConfig.OU_SIGMA,
            min_sigma=DDPGConfig.OU_SIGMA_MIN,
            decay_period=DDPGConfig.OU_DECAY,
        )

        self.memory = ReplayBuffer(self.buffer_size)

        self.hard_update(self.actor_target, self.actor_local)
        self.hard_update(self.critic_target, self.critic_local)

        self.actor_loss_history = []
        self.critic_loss_history = []

        # Log
        actor_params = sum(p.numel() for p in self.actor_local.parameters())
        critic_params = sum(p.numel() for p in self.critic_local.parameters())
        print(f"[DDPGAttentionAgent] state_dim={self.state_dim}, "
              f"layout={'enhanced' if _layout.has_lidar_diff else 'legacy'}, "
              f"n_sectors={_layout.n_sectors}, "
              f"has_diff={_layout.has_lidar_diff}, "
              f"has_dyaw={_layout.has_delta_yaw}")
        print(f"[DDPGAttentionAgent] V3 multi-head attention: "
              f"heads={self.spatial_att_heads}, "
              f"model_dim={self.sector_embed_dim}, "
              f"token_dim=6")
        print(f"[DDPGAttentionAgent] Total parameters "
              f"(actor+critic local): {actor_params + critic_params:,}")

    def reset_noise(self):
        self.noise.reset()

    def act(self, state, step=0, add_noise=True):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().numpy()
        self.actor_local.train()

        if add_noise:
            noise = self.noise.sample(step)
            action += noise

        action[0, 0] = np.clip(action[0, 0], 0.0, self.max_lin_vel)
        action[0, 1] = np.clip(action[0, 1], -self.max_ang_vel, self.max_ang_vel)
        return action

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size)

        state = torch.FloatTensor(states).to(device)
        next_state = torch.FloatTensor(next_states).to(device)
        action = torch.FloatTensor(actions).to(device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(device)
        done = torch.FloatTensor(np.float32(dones)).unsqueeze(1).to(device)

        action = torch.squeeze(action, 1)

        # ========== Update Critic ==========
        self.actor_target.eval()
        self.critic_target.eval()
        with torch.no_grad():
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            expected_Q = reward + (1.0 - done) * self.gamma * target_Q
        self.actor_target.train()
        self.critic_target.train()

        current_Q = self.critic_local(state, action)
        critic_loss = self.loss_function(current_Q, expected_Q.detach())

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1.0)
        self.critic_optimizer.step()

        # ========== Update Actor ==========
        actor_loss = -self.critic_local(state, self.actor_local(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1.0)
        self.actor_optimizer.step()

        # ========== Soft update targets ==========
        self.soft_update(self.actor_local, self.actor_target)
        self.soft_update(self.critic_local, self.critic_target)

        self.actor_loss_history.append(actor_loss.item())
        self.critic_loss_history.append(critic_loss.item())

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(),
                                              local_model.parameters()):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def hard_update(self, target_model, local_model):
        for target_param, local_param in zip(target_model.parameters(),
                                              local_model.parameters()):
            target_param.data.copy_(local_param.data)

    def save(self, filepath):
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
        torch.save(self.actor_local.state_dict(), filepath)

    def load_actor(self, filepath):
        try:
            state_dict = torch.load(filepath, map_location=device, weights_only=False)
        except TypeError:
            state_dict = torch.load(filepath, map_location=device)
        self.actor_local.load_state_dict(state_dict)
        self.actor_target.load_state_dict(state_dict)


# ========================== Test ==========================

if __name__ == '__main__':
    agent = DDPGAttentionAgent(
        state_dim=43, action_dim=2,
        n_sectors=16, sector_embed_dim=32, spatial_att_heads=4)

    state = np.random.randn(43).astype(np.float32)
    action = agent.act(state, step=0, add_noise=True)
    print(f"State shape: {state.shape}")
    print(f"Action: {action}, shape: {action.shape}")

    next_state = np.random.randn(43).astype(np.float32)
    for i in range(200):
        agent.step(state, action, 1.0, next_state, False)
    print(f"Memory size: {len(agent.memory)}")
    agent.learn()
    print("Learning step completed!")