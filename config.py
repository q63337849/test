#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置文件 - 包含所有超参数和环境参数
修复版：修正奖励函数参数
"""

import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
RESULT_DIR = os.path.join(BASE_DIR, 'results')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

for dir_path in [MODEL_DIR, RESULT_DIR, LOG_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# ==================== 环境参数 ====================
class EnvConfig:
    # 地图尺寸 (米)
    MAP_WIDTH = 10.0
    MAP_HEIGHT = 10.0
    
    # 机器人参数（提速版）
    ROBOT_RADIUS = 0.16
    MAX_LINEAR_VEL = 0.5          # 0.22 → 0.5 m/s（提速2.3倍）
    MAX_ANGULAR_VEL = 2.5         # 2.0 → 2.5（相应提高转向速度）
    
    # 激光雷达参数 (32线)
    LIDAR_RAYS = 32
    LIDAR_MAX_RANGE = 3.5
    LIDAR_MIN_RANGE = 0.12
    LIDAR_FOV = 360
    
    # Sim2Real 噪声（可选）
    LIDAR_NOISE_STD = 0.0
    LIDAR_DROPOUT_PROB = 0.0
    
    # 目标参数
    GOAL_RADIUS = 0.5              # 0.3 → 0.5（增大目标，更容易到达）
    
    # 障碍物参数（简化版，先验证算法有效性）
    NUM_STATIC_OBSTACLES = 8       # 5 → 3
    KNOWN_STATIC_OBSTACLES = 8
    NUM_DYNAMIC_OBSTACLES = 6      # 4 → 2（总共5个，原来9个）
    OBSTACLE_RADIUS_MIN = 0.1
    OBSTACLE_RADIUS_MAX = 0.3
    DYNAMIC_OBS_VEL_MIN = 0.50    # 0.05 → 0.10（相应提速）
    DYNAMIC_OBS_VEL_MAX = 0.70    # 0.15 → 0.30

    # 动态障碍物增强配置
    DYNAMIC_PROFILE_DEFAULT = "baseline"
    DYNAMIC_OBS_VEL_MIN_FAST = 0.30   # 0.15 → 0.30
    DYNAMIC_OBS_VEL_MAX_FAST = 0.70   # 0.40 → 0.80
    DYNAMIC_OBS_PATTERNS_FAST = ("bounce", "random_walk", "stop_and_go")
    DYNAMIC_OBS_STOP_PROB_FAST = 0.08

    # EnhancedSim2RealStateV2 默认开关
    ENHANCED_N_SECTORS = 16
    ENHANCED_USE_LIDAR_DIFF_DEFAULT = True
    ENHANCED_USE_DELTA_YAW_DEFAULT = True

    # 仿真参数
    DT = 0.1
    MAX_STEPS = 500               # 保持500步
    COLLISION_DISTANCE = 0.12


# ==================== DDPG参数 ====================
class DDPGConfig:
    # 网络结构
    STATE_DIM = 32 + 2 + 2 + 1 + 2
    ACTION_DIM = 2
    HIDDEN_DIM = 256
    
    # 学习率
    ACTOR_LR = 3e-4
    CRITIC_LR = 3e-4
    
    # 训练参数
    GAMMA = 0.99
    TAU = 0.005
    BATCH_SIZE = 128
    BUFFER_SIZE = 1000000
    
    # 探索噪声 (OU噪声)
    OU_MU = 0.0
    OU_THETA = 0.15
    OU_SIGMA = 0.2
    OU_SIGMA_MIN = 0.05
    OU_DECAY = 100000
    
    # 训练
    NUM_EPISODES = 3000
    LEARN_START = 1000
    SAVE_INTERVAL = 100

    # LSTM-DDPG（序列建模）
    HISTORY_LEN = 3
    LSTM_EMBED_DIM = 64
    LSTM_HIDDEN_DIM = 64
    GRAD_CLIP_NORM = 1.0
    
    UPDATE_EVERY = 4
    UPDATE_TIMES = 1


# ==================== 奖励参数（V5连续奖励版）====================
class RewardConfig:
    # 时间惩罚（鼓励快速到达）
    STEP_PENALTY = -0.5
    
    # 终端奖励
    GOAL_REWARD = 200           # 到达目标
    COLLISION_PENALTY = -200    # 碰撞惩罚
    TIMEOUT_PENALTY = -100      # ⚡ 新增：超时惩罚！
    
    # 距离奖励现在在 environment.py 中用 DISTANCE_SCALE=10 计算
    # 总距离奖励 ≈ 初始距离(8m) * 10 = 80
    # DTG_REWARD/PENALTY 不再使用（保留兼容性）
    DTG_REWARD = 0
    DTG_PENALTY = 0
    
    HTG_REWARD = 0
    HTG_PENALTY = 0
    
    FORWARD_REWARD = 0
    TURN_REWARD = 0


# ==================== 可视化参数 ====================
class VisConfig:
    WINDOW_WIDTH = 800
    WINDOW_HEIGHT = 800
    FPS = 30
    SHOW_LIDAR = True
    SHOW_TRAJECTORY = True
    TRAJECTORY_LENGTH = 100
