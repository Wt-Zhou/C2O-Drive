#!/bin/bash
# S1场景所有Baseline测试脚本

SCENARIO="s1"
EPISODES=10
MAX_STEPS=100
OUTPUT_BASE="outputs/s1_comparison"

echo "=========================================="
echo " S1场景Baseline对比测试"
echo "=========================================="
echo "场景: $SCENARIO"
echo "Episodes: $EPISODES"
echo "Max steps: $MAX_STEPS"
echo "=========================================="
echo ""

# 创建输出目录
mkdir -p $OUTPUT_BASE

# 1. C2OSR
echo "[1/5] Running C2OSR..."
python examples/run_c2osr_carla.py \
  --scenario $SCENARIO \
  --episodes $EPISODES \
  --output-dir ${OUTPUT_BASE}/c2osr \
  --quiet

# 2. PPO
echo "[2/5] Running PPO..."
python examples/run_ppo_carla.py \
  --scenario $SCENARIO \
  --episodes $EPISODES \
  --max-steps $MAX_STEPS \
  --output-dir ${OUTPUT_BASE}/ppo \
  --no-rendering \
  --quiet

# 3. SAC
echo "[3/5] Running SAC..."
python examples/run_sac_carla.py \
  --scenario $SCENARIO \
  --episodes $EPISODES \
  --max-steps $MAX_STEPS \
  --output-dir ${OUTPUT_BASE}/sac \
  --no-rendering \
  --quiet

# 4. Rainbow DQN
echo "[4/5] Running Rainbow DQN..."
python examples/run_rainbow_dqn_carla.py \
  --scenario $SCENARIO \
  --episodes $EPISODES \
  --max-steps $MAX_STEPS \
  --output-dir ${OUTPUT_BASE}/rainbow_dqn \
  --no-rendering \
  --quiet

# 5. RCRL
echo "[5/5] Running RCRL..."
python examples/run_rcrl_carla.py \
  --scenario $SCENARIO \
  --episodes $EPISODES \
  --max-steps $MAX_STEPS \
  --constraint-mode soft \
  --output-dir ${OUTPUT_BASE}/rcrl \
  --no-rendering \
  --quiet

echo ""
echo "=========================================="
echo " 所有测试完成！"
echo "=========================================="
echo "结果保存在: $OUTPUT_BASE"
echo ""
echo "目录结构:"
ls -lh $OUTPUT_BASE

echo ""
echo "查看TensorBoard日志:"
echo "  tensorboard --logdir logs/"
