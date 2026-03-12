#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""lstm_ddpg_att.py

LSTM-DDPG + Attention（为你的动态环境导航/避障状态设计定制）

本文件可作为 lstm_ddpg.py 的“即插即用”替代：
- 训练脚本只需把 `from lstm_ddpg import LSTMDdpgAgent` 改为 `from lstm_ddpg_att import LSTMDdpgAgent`。
- Agent 的接口与更新逻辑保持一致；仅网络前端引入空间注意力与时间注意力。

Attention 设计
1) 空间注意力（Beam/Sector-level）
   - 将每个时间步的 `lidar_sectors`（n_sectors=8/16 或 legacy 的 32 beams）视为 token 序列。
   - token 特征：
       [range_norm, (可选) diff, valid_flag]
     valid_flag 使用 EnhancedSim2RealStateV2 的无回波判定：range_norm < no_echo_threshold（默认 0.95）。
   - 不做 hard mask（因为空旷场景可能全无回波），而是：
       - 对 invalid token 施加可学习负偏置
       - 加入“更近更重要”的距离偏置 (1 - range_norm)

2) 时间注意力（Temporal）
   - 作用在 LSTM 的输出序列 out(B,H,hidden) 上。
   - attention score 除了依赖 out 与 last_hidden，还额外引入：
       - mean(|lidar_diff|)（若存在）：强调“变化显著”的时刻
       - |delta_yaw|（若存在）：在快速自转时降低 LiDAR 差分的可信度
       - recency bias（可学习）：在 replay padding/重复起始帧时更偏向最近

与 lidar_diff / delta_yaw 的配合
- lidar_diff：
  - 作为空间 token 的输入特征（若存在），让空间注意力更敏感“靠近/远离”的方向。
  - 同时计算每个时间步 diff_mag = mean(|diff|) 作为时间注意力的加权项。
- delta_yaw：
  - 作为非 LiDAR 特征参与空间 query（因为 query 来自 non-lidar 分支）。
  - 同时作为时间注意力的可靠度项：score += yaw_w * (1 - |delta_yaw|)。

备注
- 本实现自动根据 state_dim 推断 layout：
  - Enhanced 格式：n + (n if diff) + 3 + 2 + 2 + 1 + (1 if dyaw) + 2
  - Legacy 格式：32 + 2 + 2 + 1 + 2 + 2 = 39（见 environment.py）

"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from collections import deque
from typing import Deque, List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import DDPGConfig, EnvConfig


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================ utils ================================

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
    """与 lstm_ddpg.py 一致的 Episode ReplayBuffer（按 episode 存储，并采样定长历史窗口）。"""

    def __init__(self, capacity_transitions: int, state_dim: int, action_dim: int, history_len: int):
        self.capacity_transitions = int(capacity_transitions)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.history_len = int(history_len)

        self.episodes: Deque[_Episode] = deque()
        self._current: Optional[_Episode] = None
        self._num_transitions: int = 0

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

            start = max(0, t - self.history_len + 1)
            state_seq = self._pad_left(ep.states[start : t + 1], ep.states[0])

            t2 = t + 1
            start2 = max(0, t2 - self.history_len + 1)
            next_state_seq = self._pad_left(ep.states[start2 : t2 + 1], ep.states[0])

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


# ============================ state layout =============================

@dataclass
class StateLayout:
    mode: str  # 'enhanced' | 'legacy' | 'unknown'
    n_sectors: int
    has_lidar_diff: bool
    has_delta_yaw: bool

    sectors_slice: slice
    diff_slice: Optional[slice]
    non_lidar_slice: slice

    delta_yaw_index: Optional[int]


def infer_state_layout(state_dim: int) -> StateLayout:
    """仅根据 state_dim 推断 Enhanced / Legacy 状态布局。

    Enhanced 维度：n + (n if diff) + 3 + 2 + 2 + 1 + (1 if dyaw) + 2
    Legacy 维度（environment.py）：32 + goal(2) + pos(2) + theta(1) + vel(2) = 39
    """
    D = int(state_dim)

    # Enhanced 尝试（常用 16/8 sectors；也兼容把 32 beams 当作 sectors 的情况）
    for n in (16, 8, 32):
        for has_diff in (True, False):
            for has_dyaw in (True, False):
                exp = n + (n if has_diff else 0) + 3 + 2 + 2 + 1 + (1 if has_dyaw else 0) + 2
                if exp == D:
                    sectors_slice = slice(0, n)
                    diff_slice = slice(n, 2 * n) if has_diff else None
                    non_start = n + (n if has_diff else 0)
                    non_lidar_slice = slice(non_start, D)
                    # delta_yaw 在 non_lidar 的 (target3 + vel2 + attitude2 + yaw_rate1) 之后
                    delta_yaw_index = (non_start + 3 + 2 + 2 + 1) if has_dyaw else None
                    return StateLayout(
                        mode="enhanced",
                        n_sectors=n,
                        has_lidar_diff=has_diff,
                        has_delta_yaw=has_dyaw,
                        sectors_slice=sectors_slice,
                        diff_slice=diff_slice,
                        non_lidar_slice=non_lidar_slice,
                        delta_yaw_index=delta_yaw_index,
                    )

    # Legacy（严格按 environment.py 的拼接顺序）
    legacy_dim = int(getattr(EnvConfig, "LIDAR_RAYS", 32)) + 2 + 2 + 1 + 2
    if D == legacy_dim:
        n = int(getattr(EnvConfig, "LIDAR_RAYS", 32))
        sectors_slice = slice(0, n)
        non_lidar_slice = slice(n, D)
        return StateLayout(
            mode="legacy",
            n_sectors=n,
            has_lidar_diff=False,
            has_delta_yaw=False,
            sectors_slice=sectors_slice,
            diff_slice=None,
            non_lidar_slice=non_lidar_slice,
            delta_yaw_index=None,
        )

    # unknown fallback：尽量把前 32 维当作 beams
    n = min(int(getattr(EnvConfig, "LIDAR_RAYS", 32)), D)
    return StateLayout(
        mode="unknown",
        n_sectors=n,
        has_lidar_diff=False,
        has_delta_yaw=False,
        sectors_slice=slice(0, n),
        diff_slice=None,
        non_lidar_slice=slice(n, D),
        delta_yaw_index=None,
    )


# ============================== Attention ==============================

class SpatialSectorAttention(nn.Module):
    """Beam/Sector-level 空间注意力：每个时间步对 N 个 sectors 做注意力池化。"""

    def __init__(
        self,
        layout: StateLayout,
        non_lidar_dim: int,
        sector_model_dim: int = 32,
        no_echo_threshold: float = 0.95,
        use_distance_bias: bool = True,
        use_valid_bias: bool = True,
    ):
        super().__init__()
        self.layout = layout
        self.n = int(layout.n_sectors)
        self.no_echo_threshold = float(no_echo_threshold)
        self.use_distance_bias = bool(use_distance_bias)
        self.use_valid_bias = bool(use_valid_bias)

        token_in = 2 + (1 if layout.has_lidar_diff else 0)  # range, valid, (+diff)

        self.token_proj = nn.Linear(token_in, sector_model_dim)
        self.query_proj = nn.Linear(non_lidar_dim, sector_model_dim)

        # 可学习偏置（初始化为：更近更重要、无回波略惩罚）
        self.range_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.valid_bias = nn.Parameter(torch.tensor(0.0))
        self.invalid_bias = nn.Parameter(torch.tensor(-1.0))

        self.out_ln = nn.LayerNorm(sector_model_dim)

    def forward(self, sectors: torch.Tensor, lidar_diff: Optional[torch.Tensor], non_lidar: torch.Tensor) -> torch.Tensor:
        """Args:
        - sectors: (B,H,N) range_norm in [0,1]
        - lidar_diff: (B,H,N) in [-1,1] or None
        - non_lidar: (B,H,D_non)

        Returns:
        - spatial_ctx: (B,H,sector_model_dim)
        """
        B, H, N = sectors.shape
        assert N == self.n

        valid = (sectors < self.no_echo_threshold).float()  # (B,H,N)

        # token features
        feats = [sectors.unsqueeze(-1)]
        if lidar_diff is not None:
            feats.append(lidar_diff.unsqueeze(-1))
        feats.append(valid.unsqueeze(-1))
        tok = torch.cat(feats, dim=-1)  # (B,H,N,token_in)

        tok_emb = F.relu(self.token_proj(tok))  # (B,H,N,Dm)
        q = F.relu(self.query_proj(non_lidar)).unsqueeze(2)  # (B,H,1,Dm)

        # dot attention score
        score = (tok_emb * q).sum(dim=-1) / math.sqrt(tok_emb.shape[-1])  # (B,H,N)

        if self.use_distance_bias:
            score = score + self.range_bias_scale * (1.0 - sectors)

        if self.use_valid_bias:
            score = score + self.valid_bias * valid + self.invalid_bias * (1.0 - valid)

        w = torch.softmax(score, dim=-1)  # (B,H,N)
        ctx = (w.unsqueeze(-1) * tok_emb).sum(dim=2)  # (B,H,Dm)

        # residual LN：让 ctx 更像“加权聚合后的方向风险表示”
        ctx = self.out_ln(ctx)
        return ctx


class TemporalAttentionPool(nn.Module):
    """时间注意力池化：对 LSTM 输出序列做加权求和。"""

    def __init__(
        self,
        hidden_dim: int,
        att_dim: int = 64,
        use_time_features: bool = True,
        att_dropout: float = 0.0,
    ):
        super().__init__()
        self.use_time_features = bool(use_time_features)

        self.dropout = nn.Dropout(float(att_dropout))

        self.k_proj = nn.Linear(hidden_dim, att_dim)
        self.q_proj = nn.Linear(hidden_dim, att_dim)
        self.v_proj = nn.Linear(att_dim, 1)

        # time-feature bias（可学习标量）
        self.diff_w = nn.Parameter(torch.tensor(0.5))   # diff_mag 越大越重要
        self.yaw_w = nn.Parameter(torch.tensor(0.5))    # (1-|dyaw|) 越大越可靠
        self.recency_scale = nn.Parameter(torch.tensor(0.5))  # 越靠后越重要

        self.out_ln = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        lstm_out: torch.Tensor,
        last_hidden: torch.Tensor,
        time_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
        - lstm_out: (B,H,hidden)
        - last_hidden: (B,hidden)
        - time_feats: (B,H,2) where [:,:,0]=diff_mag, [:,:,1]=dyaw_abs; can be None

        Returns:
        - ctx: (B,hidden)
        """
        B, H, Hd = lstm_out.shape

        k = self.k_proj(lstm_out)  # (B,H,att)
        q = self.q_proj(last_hidden).unsqueeze(1)  # (B,1,att)
        e = torch.tanh(k + q)  # (B,H,att)
        score = self.v_proj(e).squeeze(-1)  # (B,H)

        # recency bias: position in [0,1], later larger
        pos = torch.linspace(0.0, 1.0, H, device=lstm_out.device, dtype=lstm_out.dtype).unsqueeze(0)  # (1,H)
        score = score + self.recency_scale * pos

        if self.use_time_features and time_feats is not None and time_feats.shape[-1] >= 2:
            diff_mag = time_feats[:, :, 0]
            dyaw_abs = time_feats[:, :, 1]
            score = score + self.diff_w * diff_mag + self.yaw_w * (1.0 - dyaw_abs)

        w = torch.softmax(score, dim=1)  # (B,H)
        ctx = (w.unsqueeze(-1) * lstm_out).sum(dim=1)  # (B,hidden)

        ctx = self.dropout(ctx)

        # residual LN（避免 ctx 偏移过大）
        ctx = self.out_ln(ctx)
        return ctx


class AttStateEncoder(nn.Module):
    """把原始 state_seq(B,H,D) 编码为 LSTM 输入 embedding(B,H,embed_dim)。

    - 空间注意力用于对 lidar tokens 做聚合，得到 spatial_ctx(B,H,Dm)
    - non-lidar + lidar 统计量做成 query 与辅助特征
    """

    def __init__(
        self,
        state_dim: int,
        embed_dim: int,
        sector_model_dim: int = 32,
        no_echo_threshold: float = 0.95,
        use_spatial_att: bool = True,
        att_dropout: float = 0.0,
        sp_gate_init: float = -4.0,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.embed_dim = int(embed_dim)
        self.layout = infer_state_layout(state_dim)
        self.use_spatial_att = bool(use_spatial_att)
        self.no_echo_threshold = float(no_echo_threshold)

        self.dropout = nn.Dropout(float(att_dropout))

        non_lidar_dim = int(self.layout.non_lidar_slice.stop - self.layout.non_lidar_slice.start)

        # aux: min_range, mean_range, valid_ratio, diff_abs_mean, diff_abs_max, dyaw_abs
        aux_dim = 6
        self.non_lidar_aug_dim = non_lidar_dim + aux_dim

        half = max(8, self.embed_dim // 2)
        self.non_proj = nn.Linear(self.non_lidar_aug_dim, half)

        # fallback stats -> half (used both when spatial-att off, and as a stable anchor when on)
        self.sp_fallback_proj = nn.Linear(3, half)

        # gate to blend (fallback) and (attention) embeddings; init negative => start close to fallback
        self.sp_gate = nn.Parameter(torch.tensor(float(sp_gate_init)))

        self.spatial_att = SpatialSectorAttention(
            layout=self.layout,
            non_lidar_dim=self.non_lidar_aug_dim,
            sector_model_dim=sector_model_dim,
            no_echo_threshold=no_echo_threshold,
        ) if self.use_spatial_att else None

        self.spatial_proj = nn.Linear(sector_model_dim, half) if self.use_spatial_att else None

        self.fuse = nn.Linear(half * 2, self.embed_dim)
        self.ln = nn.LayerNorm(self.embed_dim)

    def forward(self, state_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns:
        - x_embed: (B,H,embed_dim)
        - time_feats: (B,H,2) where [:,:,0]=diff_mag, [:,:,1]=dyaw_abs
        """
        B, H, D = state_seq.shape
        assert D == self.state_dim

        sectors = state_seq[:, :, self.layout.sectors_slice]  # (B,H,N)
        lidar_diff = state_seq[:, :, self.layout.diff_slice] if self.layout.diff_slice is not None else None
        non_lidar = state_seq[:, :, self.layout.non_lidar_slice]  # (B,H,D_non)

        # validity & aux stats
        valid = (sectors < self.no_echo_threshold).float()
        min_r = sectors.min(dim=-1).values
        mean_r = sectors.mean(dim=-1)
        valid_ratio = valid.mean(dim=-1)

        if lidar_diff is not None:
            diff_abs = lidar_diff.abs()
            diff_abs_mean = diff_abs.mean(dim=-1)
            diff_abs_max = diff_abs.max(dim=-1).values
            diff_mag = diff_abs_mean
        else:
            diff_abs_mean = torch.zeros((B, H), device=state_seq.device, dtype=state_seq.dtype)
            diff_abs_max = torch.zeros((B, H), device=state_seq.device, dtype=state_seq.dtype)
            diff_mag = diff_abs_mean

        if self.layout.delta_yaw_index is not None:
            dyaw = state_seq[:, :, int(self.layout.delta_yaw_index)]
            dyaw_abs = dyaw.abs()
        else:
            dyaw_abs = torch.zeros((B, H), device=state_seq.device, dtype=state_seq.dtype)

        # clamp to keep temporal bias numerically stable
        diff_mag = torch.clamp(diff_mag, 0.0, 1.0)
        dyaw_abs = torch.clamp(dyaw_abs, 0.0, 1.0)

        aux = torch.stack([min_r, mean_r, valid_ratio, diff_abs_mean, diff_abs_max, dyaw_abs], dim=-1)  # (B,H,6)
        non_aug = torch.cat([non_lidar, aux], dim=-1)  # (B,H,D_non+6)

        non_emb = F.relu(self.non_proj(non_aug))  # (B,H,half)

        sp_fallback = torch.stack([min_r, mean_r, valid_ratio], dim=-1)  # (B,H,3)
        sp_fallback_emb = F.relu(self.sp_fallback_proj(sp_fallback))  # (B,H,half)

        if self.spatial_att is not None and self.spatial_proj is not None:
            spatial_ctx = self.spatial_att(sectors=sectors, lidar_diff=lidar_diff, non_lidar=non_aug)  # (B,H,Dm)
            sp_att_emb = F.relu(self.spatial_proj(spatial_ctx))  # (B,H,half)
            alpha = torch.sigmoid(self.sp_gate)  # scalar
            sp_emb = alpha * sp_att_emb + (1.0 - alpha) * sp_fallback_emb
        else:
            sp_emb = sp_fallback_emb

        sp_emb = self.dropout(sp_emb)

        x = torch.cat([non_emb, sp_emb], dim=-1)  # (B,H,2*half)
        x = F.relu(self.fuse(x))
        x = self.dropout(x)
        x = self.ln(x)

        time_feats = torch.stack([diff_mag, dyaw_abs], dim=-1)  # (B,H,2)
        return x, time_feats


# ============================== Networks ===============================

class LSTMActorAtt(nn.Module):
    """LSTM-Actor + (Spatial + Temporal) Attention"""

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
        use_spatial_att: bool = True,
        use_temporal_att: bool = True,
        sector_model_dim: int = 32,
        temporal_att_dim: int = 64,
        spatial_att_heads: int = 4,
        temporal_att_heads: int = 2,
        att_dropout: float = 0.0,
        sp_gate_init: float = -4.0,
        temporal_gate_init: float = -4.0,
    ):
        super().__init__()

        # NOTE: 当前 SpatialSectorAttention 为单头实现；spatial_att_heads 先保留为超参占位，便于后续升级。
        _ = int(spatial_att_heads)

        self.encoder = AttStateEncoder(
            state_dim=state_dim,
            embed_dim=embed_dim,
            sector_model_dim=sector_model_dim,
            no_echo_threshold=0.95,
            use_spatial_att=use_spatial_att,
            att_dropout=att_dropout,
            sp_gate_init=sp_gate_init,
        )
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)

        self.use_temporal_att = bool(use_temporal_att)
        self.temporal_att = TemporalAttentionPool(
            hidden_dim=lstm_hidden_dim,
            att_dim=temporal_att_dim,
            use_time_features=True,
            att_dropout=att_dropout,
        ) if self.use_temporal_att else None

        # NOTE: 当前 TemporalAttentionPool 为单头实现；temporal_att_heads 先保留为超参占位。
        _ = int(temporal_att_heads)

        self.temporal_gate = nn.Parameter(torch.tensor(float(temporal_gate_init))) if self.use_temporal_att else None

        self.fuse = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim) if self.use_temporal_att else None

        self.fc1 = nn.Linear(lstm_hidden_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc_out = nn.Linear(mlp_hidden_dim, action_dim)

        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

        self.max_lin_vel = float(max_lin_vel)
        self.max_ang_vel = float(max_ang_vel)

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        x, time_feats = self.encoder(state_seq)  # (B,H,embed)
        out, _ = self.lstm(x)  # (B,H,hid)
        last = out[:, -1, :]

        if self.temporal_att is not None and self.fuse is not None and self.temporal_gate is not None:
            ctx = self.temporal_att(out, last, time_feats=time_feats)
            fused = F.relu(self.fuse(torch.cat([last, ctx], dim=1)))
            alpha = torch.sigmoid(self.temporal_gate)
            h = alpha * fused + (1.0 - alpha) * last
        else:
            h = last

        h = F.relu(self.fc1(h))
        h = F.relu(self.fc2(h))
        action = self.fc_out(h)

        action_out = torch.zeros_like(action)
        action_out[:, 0] = torch.sigmoid(action[:, 0]) * self.max_lin_vel
        action_out[:, 1] = torch.tanh(action[:, 1]) * self.max_ang_vel
        return action_out


class LSTMCriticAtt(nn.Module):
    """LSTM-Critic + (Spatial + Temporal) Attention"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        embed_dim: int = 64,
        lstm_hidden_dim: int = 64,
        mlp_hidden_dim: int = 256,
        init_w: float = 3e-3,
        use_spatial_att: bool = True,
        use_temporal_att: bool = True,
        sector_model_dim: int = 32,
        temporal_att_dim: int = 64,
        spatial_att_heads: int = 4,
        temporal_att_heads: int = 2,
        att_dropout: float = 0.0,
        sp_gate_init: float = -4.0,
        temporal_gate_init: float = -4.0,
    ):
        super().__init__()

        self.encoder = AttStateEncoder(
            state_dim=state_dim,
            embed_dim=embed_dim,
            sector_model_dim=sector_model_dim,
            no_echo_threshold=0.95,
            use_spatial_att=use_spatial_att,
            att_dropout=att_dropout,
            sp_gate_init=sp_gate_init,
        )
        self.lstm = nn.LSTM(embed_dim, lstm_hidden_dim, batch_first=True)

        self.use_temporal_att = bool(use_temporal_att)
        self.temporal_att = TemporalAttentionPool(
            hidden_dim=lstm_hidden_dim,
            att_dim=temporal_att_dim,
            use_time_features=True,
            att_dropout=att_dropout,
        ) if self.use_temporal_att else None

        self.fuse = nn.Linear(lstm_hidden_dim * 2, lstm_hidden_dim) if self.use_temporal_att else None

        # temporal gate: start close to "no temporal attention" for stability
        self.temporal_gate = nn.Parameter(torch.tensor(float(temporal_gate_init))) if self.use_temporal_att else None

        self.fc1 = nn.Linear(lstm_hidden_dim + action_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc_out = nn.Linear(mlp_hidden_dim, 1)

        self.fc_out.weight.data.uniform_(-init_w, init_w)
        self.fc_out.bias.data.uniform_(-init_w, init_w)

    def forward(self, state_seq: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x, time_feats = self.encoder(state_seq)
        out, _ = self.lstm(x)
        last = out[:, -1, :]

        if self.temporal_att is not None and self.fuse is not None:
            ctx = self.temporal_att(out, last, time_feats=time_feats)
            fused = F.relu(self.fuse(torch.cat([last, ctx], dim=1)))
            if self.temporal_gate is not None:
                alpha = torch.sigmoid(self.temporal_gate)
                h = alpha * fused + (1.0 - alpha) * last
            else:
                h = fused
        else:
            h = last

        x = torch.cat([h, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


# ================================ Agent ================================

class LSTMDdpgAgent:
    """LSTM-DDPG Agent + Attention（保持训练流程与 lstm_ddpg.py 一致）。"""

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
        # Attention options
        use_spatial_att: bool = True,
        use_temporal_att: bool = True,
        sector_model_dim: int = 32,
        temporal_att_dim: int = 64,
        spatial_att_heads: int = 4,
        temporal_att_heads: int = 4,
        att_dropout: float = 0.0,
        sp_gate_init: float = -2.0,
        temporal_gate_init: float = -2.0,
        # Exploration scaling
        noise_lin_scale: float = 1.0,
        noise_ang_scale: float = 1.0,
        min_lin_explore: float = 0.0,
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

        self.use_spatial_att = bool(use_spatial_att)
        self.use_temporal_att = bool(use_temporal_att)
        self.sector_model_dim = int(sector_model_dim)
        self.temporal_att_dim = int(temporal_att_dim)
        self.spatial_att_heads = int(spatial_att_heads)
        self.temporal_att_heads = int(temporal_att_heads)

        # exploration scaling (per-action component)
        self.noise_lin_scale = float(noise_lin_scale)
        self.noise_ang_scale = float(noise_ang_scale)
        self.min_lin_explore = float(min_lin_explore)
        self.min_lin_explore = float(min_lin_explore)

        self.state_meta: Dict[str, Any] = {}
        self._step_count = 0

        # Networks
        self.actor_local = LSTMActorAtt(
            self.state_dim,
            self.action_dim,
            embed_dim=self.embed_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
            max_lin_vel=self.max_lin_vel,
            max_ang_vel=self.max_ang_vel,
            use_spatial_att=self.use_spatial_att,
            use_temporal_att=self.use_temporal_att,
            sector_model_dim=self.sector_model_dim,
            temporal_att_dim=self.temporal_att_dim,
            spatial_att_heads=self.spatial_att_heads,
            temporal_att_heads=self.temporal_att_heads,
            att_dropout=float(att_dropout),
            sp_gate_init=float(sp_gate_init),
            temporal_gate_init=float(temporal_gate_init),
        ).to(device)
        self.actor_target = LSTMActorAtt(
            self.state_dim,
            self.action_dim,
            embed_dim=self.embed_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
            max_lin_vel=self.max_lin_vel,
            max_ang_vel=self.max_ang_vel,
            use_spatial_att=self.use_spatial_att,
            use_temporal_att=self.use_temporal_att,
            sector_model_dim=self.sector_model_dim,
            temporal_att_dim=self.temporal_att_dim,
            spatial_att_heads=self.spatial_att_heads,
            temporal_att_heads=self.temporal_att_heads,
            att_dropout=float(att_dropout),
            sp_gate_init=float(sp_gate_init),
            temporal_gate_init=float(temporal_gate_init),
        ).to(device)

        self.critic_local = LSTMCriticAtt(
            self.state_dim,
            self.action_dim,
            embed_dim=self.embed_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
            use_spatial_att=self.use_spatial_att,
            use_temporal_att=self.use_temporal_att,
            sector_model_dim=self.sector_model_dim,
            temporal_att_dim=self.temporal_att_dim,
            spatial_att_heads=self.spatial_att_heads,
            temporal_att_heads=self.temporal_att_heads,
            att_dropout=float(att_dropout),
            sp_gate_init=float(sp_gate_init),
            temporal_gate_init=float(temporal_gate_init),
        ).to(device)
        self.critic_target = LSTMCriticAtt(
            self.state_dim,
            self.action_dim,
            embed_dim=self.embed_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            mlp_hidden_dim=self.hidden_dim,
            use_spatial_att=self.use_spatial_att,
            use_temporal_att=self.use_temporal_att,
            sector_model_dim=self.sector_model_dim,
            temporal_att_dim=self.temporal_att_dim,
            spatial_att_heads=self.spatial_att_heads,
            temporal_att_heads=self.temporal_att_heads,
            att_dropout=float(att_dropout),
            sp_gate_init=float(sp_gate_init),
            temporal_gate_init=float(temporal_gate_init),
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
            n = self.noise.sample(step)
            # scale OU noise per action dimension: [linear, angular]
            if n.shape[0] >= 1:
                n[0] *= self.noise_lin_scale
            if n.shape[0] >= 2:
                n[1] *= self.noise_ang_scale
            action += n

            # keep a minimum forward action during exploration to avoid being stuck
            if self.min_lin_explore > 0.0:
                action[0, 0] = max(action[0, 0], self.min_lin_explore)

        action[0, 0] = np.clip(action[0, 0], 0.0, self.max_lin_vel)
        action[0, 1] = np.clip(action[0, 1], -self.max_ang_vel, self.max_ang_vel)
        return action

    def step(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool) -> None:
        self.memory.add(state, action, reward, next_state, done)
        self._step_count += 1

    def should_learn(self) -> bool:
        return (
            self._step_count % self.update_every == 0
            and len(self.memory) >= self.batch_size
            and len(self.memory.episodes) > 0
        )

    def learn(self) -> None:
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
                    "use_spatial_att": self.use_spatial_att,
                    "use_temporal_att": self.use_temporal_att,
                    "sector_model_dim": self.sector_model_dim,
                    "temporal_att_dim": self.temporal_att_dim,
                    "spatial_att_heads": self.spatial_att_heads,
                    "temporal_att_heads": self.temporal_att_heads,
                    "layout": infer_state_layout(self.state_dim).__dict__,
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
        # PyTorch 2.6+ 需要 weights_only=False 来加载包含非张量对象的checkpoint
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        actor_sd = checkpoint.get("actor_local", {})
        if "encoder.non_proj.weight" in actor_sd:
            ckpt_state_dim = int(actor_sd["encoder.non_proj.weight"].shape[1])  # this is non_aug, not state_dim
            # state_dim 不能从 non_proj 反推；改用 net_cfg
            ckpt_state_dim = checkpoint.get("net_cfg", {}).get("state_dim", None)
        else:
            ckpt_state_dim = checkpoint.get("net_cfg", {}).get("state_dim", None)

        if ckpt_state_dim is not None and int(ckpt_state_dim) != int(self.state_dim):
            raise RuntimeError(
                f"Checkpoint state_dim={ckpt_state_dim} != current {self.state_dim}. "
                "请确认 n_sectors / lidar_diff / delta_yaw 与 checkpoint 一致。"
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
