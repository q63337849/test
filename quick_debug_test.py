#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""quick_debug_test.py

快速诊断LSTM-ATT的问题
不需要完整训练，只需几个step就能发现问题

使用方法:
python quick_debug_test.py
"""

import numpy as np
import torch
from collections import deque

# 导入你的模块
from lstm_ddpg_att import LSTMDdpgAgent
from environment import NavigationEnv
from config import EnvConfig

def print_section(title):
    """打印分节标题"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80)

def check_tensor_stats(tensor, name):
    """检查张量统计"""
    with torch.no_grad():
        mean = tensor.mean().item()
        std = tensor.std().item()
        max_val = tensor.max().item()
        min_val = tensor.min().item()
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        print(f"\n{name}:")
        print(f"  Shape: {list(tensor.shape)}")
        print(f"  Mean: {mean:.6f}  Std: {std:.6f}")
        print(f"  Range: [{min_val:.6f}, {max_val:.6f}]")
        
        if has_nan:
            print(f"  ⚠️  包含 NaN!")
        if has_inf:
            print(f"  ⚠️  包含 Inf!")
        if std < 1e-6:
            print(f"  ⚠️  标准差接近0!")
        
        return mean, std, max_val, min_val, has_nan, has_inf

def test_forward_pass():
    """测试前向传播"""
    print_section("1. 测试前向传播")
    
    # 创建环境
    env = NavigationEnv(use_enhanced_state=True)
    print(f"✓ 环境创建成功")
    print(f"  State dim: {env.state_dim}")
    print(f"  Action dim: {env.action_dim}")
    print(f"  Obstacles: {EnvConfig.NUM_STATIC_OBSTACLES} static + {EnvConfig.NUM_DYNAMIC_OBSTACLES} dynamic = {EnvConfig.NUM_STATIC_OBSTACLES + EnvConfig.NUM_DYNAMIC_OBSTACLES} total")
    
    # 创建Agent
    agent = LSTMDdpgAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        history_len=3,
        use_spatial_att=True,
        use_temporal_att=True,
    )
    print(f"✓ Agent创建成功")
    
    # 收集一个episode的数据
    state = env.reset()
    state_queue = deque([state.copy() for _ in range(3)], maxlen=3)
    
    print(f"\n测试单次前向传播...")
    state_seq = np.asarray(state_queue, dtype=np.float32)
    print(f"  Input shape: {state_seq.shape}")
    
    # 转换为torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_seq_t = torch.from_numpy(state_seq).float().unsqueeze(0).to(device)
    
    # 前向传播
    agent.actor_local.eval()
    with torch.no_grad():
        action = agent.actor_local(state_seq_t)
    agent.actor_local.train()
    
    print(f"✓ 前向传播成功")
    print(f"  Output shape: {action.shape}")
    print(f"  Actions: linear={action[0,0].item():.4f}, angular={action[0,1].item():.4f}")
    
    return agent, env

def test_attention_weights(agent, env):
    """测试注意力权重"""
    print_section("2. 测试注意力权重")
    
    # 收集一些数据
    state = env.reset()
    state_queue = deque([state.copy() for _ in range(3)], maxlen=3)
    
    state_seq = np.asarray(state_queue, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_seq_t = torch.from_numpy(state_seq).float().unsqueeze(0).to(device)
    
    # 手动调用encoder检查空间注意力
    agent.actor_local.eval()
    with torch.no_grad():
        x, time_feats = agent.actor_local.encoder(state_seq_t)
        
        # 尝试获取空间注意力权重
        if hasattr(agent.actor_local.encoder, 'spatial_att'):
            print("\n✓ 检测到空间注意力模块")
            
            # 手动计算空间注意力
            B, H, D = state_seq_t.shape
            layout = agent.actor_local.encoder.layout
            
            sectors = state_seq_t[:, :, layout.sectors_slice]
            lidar_diff = state_seq_t[:, :, layout.diff_slice] if layout.diff_slice else None
            non_lidar = state_seq_t[:, :, layout.non_lidar_slice]
            
            # 计算valid
            valid = (sectors < 0.95).float()
            valid_ratio = valid.mean(dim=-1)
            min_r = sectors.min(dim=-1).values
            mean_r = sectors.mean(dim=-1)
            
            print(f"\nLiDAR统计:")
            print(f"  扇区数: {sectors.shape[-1]}")
            print(f"  有效比例: {valid_ratio.mean().item():.2%}")
            print(f"  最小距离: {min_r.mean().item():.3f}m")
            print(f"  平均距离: {mean_r.mean().item():.3f}m")
            
            # 计算attention score (简化版)
            spatial_att = agent.actor_local.encoder.spatial_att
            
            # Token features
            feats = [sectors.unsqueeze(-1)]
            if lidar_diff is not None:
                feats.append(lidar_diff.unsqueeze(-1))
            feats.append(valid.unsqueeze(-1))
            tok = torch.cat(feats, dim=-1)
            
            tok_emb = torch.relu(spatial_att.token_proj(tok))
            
            # Non-lidar augmentation
            aux = torch.stack([min_r, mean_r, valid_ratio,
                             torch.zeros_like(min_r), torch.zeros_like(min_r), torch.zeros_like(min_r)], dim=-1)
            non_aug = torch.cat([non_lidar, aux], dim=-1)
            
            q = torch.relu(spatial_att.query_proj(non_aug)).unsqueeze(2)
            
            score = (tok_emb * q).sum(dim=-1) / (tok_emb.shape[-1] ** 0.5)
            score = score + spatial_att.range_bias_scale * (1.0 - sectors)
            score = score + spatial_att.valid_bias * valid + spatial_att.invalid_bias * (1.0 - valid)
            
            w = torch.softmax(score, dim=-1)
            
            # 检查权重
            check_tensor_stats(w, "空间注意力权重")
            
            # 额外检查
            n_sectors = w.shape[-1]
            expected_mean = 1.0 / n_sectors
            actual_mean = w.mean().item()
            deviation = abs(actual_mean - expected_mean) / expected_mean
            
            print(f"\n均匀性检查:")
            print(f"  期望均值: {expected_mean:.4f}")
            print(f"  实际均值: {actual_mean:.4f}")
            print(f"  偏差: {deviation:.1%}")
            
            if deviation < 0.1:
                print(f"  ✓ 权重接近均匀 (偏差<10%)")
            elif deviation < 0.3:
                print(f"  ⚠️  权重有一定偏向 (10%<偏差<30%)")
            else:
                print(f"  ⚠️  权重严重不均匀 (偏差>30%)")
            
            # 检查最大权重
            max_weights = w.max(dim=-1).values
            print(f"\n集中度检查:")
            print(f"  最大权重均值: {max_weights.mean().item():.4f}")
            if max_weights.mean().item() > 0.9:
                print(f"  ⚠️  注意力过度集中在单个扇区!")
            elif max_weights.mean().item() > 0.5:
                print(f"  ⚠️  注意力较为集中")
            else:
                print(f"  ✓ 注意力分散合理")
        
        # 检查时间注意力
        if hasattr(agent.actor_local, 'temporal_att'):
            print("\n✓ 检测到时间注意力模块")
            
            out, _ = agent.actor_local.lstm(x)
            last = out[:, -1, :]
            
            temporal_att = agent.actor_local.temporal_att
            
            # 计算temporal attention
            k = temporal_att.k_proj(out)
            q = temporal_att.q_proj(last).unsqueeze(1)
            e = torch.tanh(k + q)
            score = temporal_att.v_proj(e).squeeze(-1)
            
            H = out.shape[1]
            pos = torch.linspace(0.0, 1.0, H, device=out.device).unsqueeze(0)
            score = score + temporal_att.recency_scale * pos
            
            if time_feats is not None:
                diff_mag = time_feats[:, :, 0]
                dyaw_abs = time_feats[:, :, 1]
                score = score + temporal_att.diff_w * diff_mag + temporal_att.yaw_w * (1.0 - dyaw_abs)
            
            w = torch.softmax(score, dim=1)
            
            # 检查权重
            check_tensor_stats(w, "时间注意力权重")
            
            # 额外检查
            expected_mean = 1.0 / H
            actual_mean = w.mean().item()
            deviation = abs(actual_mean - expected_mean) / expected_mean
            
            print(f"\n均匀性检查:")
            print(f"  期望均值: {expected_mean:.4f} (1/{H})")
            print(f"  实际均值: {actual_mean:.4f}")
            print(f"  偏差: {deviation:.1%}")
            
            # 检查recency bias
            print(f"\nRecency bias: {temporal_att.recency_scale.item():.4f}")
            if temporal_att.recency_scale.item() > 2.0:
                print(f"  ⚠️  Recency bias过大，可能过度偏向最近时刻")
    
    agent.actor_local.train()

def test_gradients(agent, env):
    """测试梯度流动"""
    print_section("3. 测试梯度流动")
    
    # 收集一些经验
    print("收集经验...")
    for _ in range(100):
        state = env.reset()
        for step in range(50):
            action = agent.act(np.array([state]*3), add_noise=True)
            next_state, reward, done, _ = env.step(action[0])
            agent.step(state, action[0], reward, next_state, done)
            state = next_state
            if done:
                break
    
    print(f"✓ 收集了 {len(agent.memory)} 个transitions")
    
    # 执行一次学习
    print("\n执行一次学习...")
    agent._learn_once()
    print("✓ 学习完成")
    
    # 检查梯度
    print("\n检查Actor梯度:")
    total_norm = 0.0
    zero_grads = []
    large_grads = []
    
    for name, param in agent.actor_local.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            
            if param_norm < 1e-8:
                zero_grads.append(name)
            if param_norm > 100:
                large_grads.append(name)
    
    total_norm = total_norm ** 0.5
    print(f"  Total norm: {total_norm:.6f}")
    
    if total_norm < 1e-6:
        print(f"  ⚠️  梯度消失! norm={total_norm:.10f}")
    elif total_norm > 100:
        print(f"  ⚠️  梯度爆炸! norm={total_norm:.2f}")
    else:
        print(f"  ✓ 梯度正常")
    
    if zero_grads:
        print(f"  ⚠️  {len(zero_grads)}个参数梯度接近0:")
        for zg in zero_grads[:3]:
            print(f"     - {zg}")
    
    if large_grads:
        print(f"  ⚠️  {len(large_grads)}个参数梯度过大:")
        for lg in large_grads[:3]:
            print(f"     - {lg}")
    
    # 检查Critic梯度
    print("\n检查Critic梯度:")
    total_norm = 0.0
    for param in agent.critic_local.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5
    print(f"  Total norm: {total_norm:.6f}")
    
    # 检查Loss
    print("\n检查Loss:")
    print(f"  Actor loss: {agent.actor_loss_history[-1]:.6f}")
    print(f"  Critic loss: {agent.critic_loss_history[-1]:.6f}")

def test_gates(agent):
    """测试门控值"""
    print_section("4. 测试门控机制")
    
    if hasattr(agent.actor_local.encoder, 'sp_gate'):
        sp_alpha = torch.sigmoid(agent.actor_local.encoder.sp_gate).item()
        print(f"\n空间注意力门控:")
        print(f"  α = {sp_alpha:.6f}")
        print(f"  Raw value = {agent.actor_local.encoder.sp_gate.item():.6f}")
        
        if sp_alpha < 0.05:
            print(f"  ⚠️  门控几乎关闭 (α<5%)")
            print(f"     → 空间注意力未被使用")
        elif sp_alpha > 0.95:
            print(f"  ✓ 门控几乎完全打开 (α>95%)")
            print(f"     → 完全使用空间注意力")
        else:
            print(f"  ✓ 门控部分打开 ({sp_alpha:.1%})")
            print(f"     → 混合使用attention和fallback")
    
    if hasattr(agent.actor_local, 'temporal_gate'):
        temp_alpha = torch.sigmoid(agent.actor_local.temporal_gate).item()
        print(f"\n时间注意力门控:")
        print(f"  α = {temp_alpha:.6f}")
        print(f"  Raw value = {agent.actor_local.temporal_gate.item():.6f}")
        
        if temp_alpha < 0.05:
            print(f"  ⚠️  门控几乎关闭 (α<5%)")
            print(f"     → 时间注意力未被使用")
        elif temp_alpha > 0.95:
            print(f"  ✓ 门控几乎完全打开 (α>95%)")
            print(f"     → 完全使用时间注意力")
        else:
            print(f"  ✓ 门控部分打开 ({temp_alpha:.1%})")
            print(f"     → 混合使用attention和LSTM last")

def main():
    """主函数"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "LSTM-ATT 快速诊断工具" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    
    try:
        # 测试1: 前向传播
        agent, env = test_forward_pass()
        
        # 测试2: 注意力权重
        test_attention_weights(agent, env)
        
        # 测试3: 门控
        test_gates(agent)
        
        # 测试4: 梯度
        test_gradients(agent, env)
        
        # 总结
        print_section("📊 诊断总结")
        print("\n如果看到以下问题，说明存在bug:")
        print("  ❌ NaN或Inf")
        print("  ❌ 梯度消失 (norm < 1e-6)")
        print("  ❌ 梯度爆炸 (norm > 100)")
        print("  ❌ 注意力权重过度集中 (max > 0.9)")
        print("  ❌ 门控长时间不增长 (α < 0.05)")
        
        print("\n如果一切正常:")
        print("  ✅ 权重在合理范围")
        print("  ✅ 梯度正常流动")
        print("  ✅ 注意力有合理的选择性")
        print("  ✅ 门控逐渐增长")
        
        print("\n" + "="*80)
        print("诊断完成! 请根据上面的结果判断是否有问题。")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n⚠️  发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n请将错误信息发给我，我会帮你分析!")

if __name__ == "__main__":
    main()
