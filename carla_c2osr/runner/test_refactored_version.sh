#!/bin/bash
# 对比测试脚本：验证简化版本和原版本产生相同的结果

set -e

echo "========================================="
echo "对比测试：原版本 vs 简化版本"
echo "========================================="

# 清理旧的输出
rm -rf outputs/replay_experiment_original
rm -rf outputs/replay_experiment_simple

# 测试参数
EPISODES=3
SEED=2025
PRESET="fast"

echo ""
echo "1. 运行原版本..."
python carla_c2osr/runner/replay_openloop_lattice.py \
    --episodes $EPISODES \
    --seed $SEED \
    --config-preset $PRESET 2>&1 | tee /tmp/original_output.log

# 保存输出
mv outputs/replay_experiment outputs/replay_experiment_original

echo ""
echo "2. 运行简化版本..."
python carla_c2osr/runner/replay_openloop_lattice_simple.py \
    --episodes $EPISODES \
    --seed $SEED \
    --config-preset $PRESET 2>&1 | tee /tmp/simple_output.log

# 保存输出
mv outputs/replay_experiment outputs/replay_experiment_simple

echo ""
echo "========================================="
echo "3. 对比结果"
echo "========================================="

# 提取关键指标
echo ""
echo "原版本结果:"
grep -A 3 "轨迹选择改进:" /tmp/original_output.log | tail -3
grep "Dirichlet学习:" /tmp/original_output.log
grep "Buffer:" /tmp/original_output.log

echo ""
echo "简化版本结果:"
grep -A 3 "轨迹选择改进:" /tmp/simple_output.log | tail -3
grep "Dirichlet学习:" /tmp/simple_output.log
grep "Buffer:" /tmp/simple_output.log

echo ""
echo "========================================="
echo "4. 文件对比"
echo "========================================="
echo "原版本输出: outputs/replay_experiment_original/"
echo "简化版本输出: outputs/replay_experiment_simple/"

# 对比Q值数据
echo ""
echo "Q值数据对比:"
python3 << 'EOF'
import json

# 读取两个版本的Q值数据
with open('outputs/replay_experiment_original/q_distribution_data.json', 'r') as f:
    original = json.load(f)
with open('outputs/replay_experiment_simple/q_distribution_data.json', 'r') as f:
    simple = json.load(f)

# 对比Q值历史
print(f"Q值历史 (原版本): {original['q_value_history']}")
print(f"Q值历史 (简化版): {simple['q_value_history']}")

# 检查是否相同
if original['q_value_history'] == simple['q_value_history']:
    print("✅ Q值历史完全一致")
else:
    print("⚠️ Q值历史存在差异")
    for i, (o, s) in enumerate(zip(original['q_value_history'], simple['q_value_history'])):
        diff = abs(o - s)
        if diff > 0.001:
            print(f"  Episode {i}: diff={diff:.6f}")

# 对比碰撞率
print(f"\n碰撞率历史 (原版本): {original['collision_rate_history']}")
print(f"碰撞率历史 (简化版): {simple['collision_rate_history']}")

if original['collision_rate_history'] == simple['collision_rate_history']:
    print("✅ 碰撞率历史完全一致")
else:
    print("⚠️ 碰撞率历史存在差异")
EOF

echo ""
echo "========================================="
echo "测试完成！"
echo "========================================="
