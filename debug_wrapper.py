#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""debug_wrapper_v2.py - 修复版"""

import torch


class DebugConfig:
    enabled = False
    print_every_episode = 10  # 每N个episode打印一次
    _episode_counter = 0


def enable_debug(print_every_episode=10):
    """启用调试模式"""
    DebugConfig.enabled = True
    DebugConfig.print_every_episode = print_every_episode
    print("\n" + "=" * 80)
    print("🔧 调试模式已启用 (v2 - 修复版)")
    print("=" * 80)
    print(f"  打印频率: 每 {print_every_episode} episodes")
    print("=" * 80 + "\n")


def wrap_agent_with_debug(agent):
    """包装agent启用调试"""
    if not DebugConfig.enabled:
        print("⚠️  调试未启用!")
        return agent

    print("🔧 正在包装Agent启用调试...")

    # 保存原始方法
    original_learn = agent._learn_once

    # 包装learn方法
    def wrapped_learn():
        # 调用原始learn
        original_learn()

        # 每N个episode打印一次
        DebugConfig._episode_counter += 1
        if DebugConfig._episode_counter % (DebugConfig.print_every_episode * 100) == 0:

            print("\n" + "=" * 80)
            print(f"🔍 调试信息 (Episode ~{DebugConfig._episode_counter // 100})")
            print("=" * 80)

            # 检查门控值
            if hasattr(agent.actor_local.encoder, 'sp_gate'):
                sp_alpha = torch.sigmoid(agent.actor_local.encoder.sp_gate).item()
                print(f"  空间门控 α = {sp_alpha:.4f}")

            if hasattr(agent.actor_local, 'temporal_gate'):
                temp_alpha = torch.sigmoid(agent.actor_local.temporal_gate).item()
                print(f"  时间门控 α = {temp_alpha:.4f}")

            # 检查梯度
            actor_norm = 0.0
            for p in agent.actor_local.parameters():
                if p.grad is not None:
                    actor_norm += p.grad.data.norm(2).item() ** 2
            actor_norm = actor_norm ** 0.5
            print(f"  Actor梯度 norm = {actor_norm:.6f}")

            critic_norm = 0.0
            for p in agent.critic_local.parameters():
                if p.grad is not None:
                    critic_norm += p.grad.data.norm(2).item() ** 2
            critic_norm = critic_norm ** 0.5
            print(f"  Critic梯度 norm = {critic_norm:.6f}")

            # 检查loss
            if hasattr(agent, 'actor_loss_history') and len(agent.actor_loss_history) > 0:
                print(f"  Actor loss = {agent.actor_loss_history[-1]:.6f}")
                print(f"  Critic loss = {agent.critic_loss_history[-1]:.6f}")

            print("=" * 80)

    agent._learn_once = wrapped_learn
    print("   ✓ Learn方法已包装")
    print("✅ Agent调试包装完成!\n")

    return agent