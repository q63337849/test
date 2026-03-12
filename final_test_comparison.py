#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""final_test_comparison.py

5阶段训练完成后的最终对比测试
"""

import subprocess
import sys

print("=" * 80)
print("🎉 LSTM-DDPG + Attention 最终性能对比测试")
print("=" * 80)

print("\n✅ 5阶段课程学习训练已完成：")
print("  Stage 1: 0.20-0.40 m/s - 800轮 ✅")
print("  Stage 2: 0.25-0.50 m/s - 800轮 ✅")
print("  Stage 3: 0.30-0.60 m/s - 800轮 ✅")
print("  Stage 4: 0.35-0.70 m/s - 800轮 ✅")
print("  Stage 5: 0.30-0.70 m/s - 1000轮 ✅")
print("\n  总训练轮数: 4200")
print("  最终成功率: 72.4% (训练场景)")

print("\n" + "=" * 80)
print("测试配置")
print("=" * 80)

# 测试配置
test_episodes = 200
speed_min = 0.50
speed_max = 0.70
seed = 42

print(f"  测试轮数: {test_episodes}")
print(f"  速度范围: {speed_min:.2f}-{speed_max:.2f} m/s (高速场景)")
print(f"  随机种子: {seed}")

print("\n模型配置:")
print("  DDPG:          models/ddpg_final.pth")
print("  LSTM-DDPG V6:  models/lstm_ddpg_v6_final_20260125_011043.pth")
print("  LSTM-DDPG-ATT: models/lstm_ddpg_att_final.pth")

print("\n" + "=" * 80)
print("预期结果")
print("=" * 80)
print("  DDPG:           79.5%")
print("  LSTM-DDPG V6:   83.5%")
print("  LSTM-DDPG-ATT:  88-92% ⭐")
print("\n  相比DDPG:      +8.5-12.5%")
print("  相比LSTM-DDPG:  +4.5-8.5%")

input("\n按Enter键开始测试...")

# 构建测试命令
cmd = [
    sys.executable,
    "test_ddpg_lstm_lstmatt_two_scenarios.py",
    "--ddpg_model", "models/ddpg_final.pth",
    "--lstm_model", "models/lstm_ddpg_v6_final_20260125_011043.pth",
    "--att_model", "models/lstm_ddpg_att_final.pth",
    "--episodes", str(test_episodes),
    "--dynamic_speed_min", str(speed_min),
    "--dynamic_speed_max", str(speed_max),
    "--seed", str(seed),
]

print("\n" + "=" * 80)
print("开始测试...")
print("=" * 80)

try:
    # 运行测试
    result = subprocess.run(cmd, check=True)
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！")
    print("=" * 80)
    
    print("\n结果分析:")
    print("  1. 查看输出中的成功率对比")
    print("  2. 查看results目录中的对比图表")
    print("  3. 记录数据用于论文")
    
    print("\n如果需要更多测试:")
    print("  - 不同速度范围：修改 --dynamic_speed_min/max")
    print("  - 更多测试轮数：修改 --episodes")
    print("  - 可视化测试：添加 --render")

except subprocess.CalledProcessError as e:
    print(f"\n❌ 测试失败: {e}")
    print("\n请检查:")
    print("  1. 所有模型文件是否存在")
    print("  2. test_ddpg_lstm_lstmatt_two_scenarios.py是否可用")
    print("  3. 环境配置是否正确")
    sys.exit(1)

except KeyboardInterrupt:
    print("\n\n⚠️ 测试被用户中断")
    sys.exit(0)

print("\n" + "=" * 80)
print("🎉 5阶段训练和测试全部完成！")
print("=" * 80)
