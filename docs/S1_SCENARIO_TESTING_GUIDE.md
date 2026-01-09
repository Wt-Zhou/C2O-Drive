# S1场景Baseline测试完整指南

## 📋 目录

- [场景说明](#场景说明)
- [前置准备](#前置准备)
- [所有算法运行命令](#所有算法运行命令)
- [批量测试脚本](#批量测试脚本)
- [结果查看](#结果查看)
- [问题排查](#问题排查)

---

## 场景说明

**S1场景（s1_scenario）**: 环境车逆行场景
- **描述**: 对向逆行车辆切入本车道
- **地图**: Town03
- **难度**: 困难 (Hard)
- **自车位置**: (5.5, -90.0, 0.5) 朝南(-90°)
- **环境车**: (12.8, -123.0, 1.0) 逆行切入(100°)
- **测试重点**: 算法对突发危险情况的响应能力

---

## 前置准备

### 1. 启动CARLA服务器

```bash
# 在第一个终端
cd /path/to/CARLA
./CarlaUE4.sh

# 或指定特定端口
./CarlaUE4.sh -carla-rpc-port=2000
```

### 2. 进入项目目录

```bash
cd /home/zwt/code/C2O-Drive
```

---

## 所有算法运行命令

### ✅ 算法1: C2OSR

**快速测试 (3 episodes)**:
```bash
python examples/run_c2osr_carla.py \
  --scenario s1 \
  --episodes 3 \
  --output-dir outputs/s1_test/c2osr
```

**标准测试 (10 episodes)**:
```bash
python examples/run_c2osr_carla.py \
  --scenario s1 \
  --episodes 10 \
  --config-preset default \
  --output-dir outputs/s1_test/c2osr \
  --host localhost \
  --port 2000
```

**完整训练 (100 episodes, 高精度)**:
```bash
python examples/run_c2osr_carla.py \
  --scenario s1 \
  --episodes 100 \
  --config-preset high-precision \
  --horizon 10 \
  --dt 1.0 \
  --grid-size 200.0 \
  --visualize-distributions \
  --output-dir outputs/s1_train/c2osr \
  --vis-interval 5
```

---

### ✅ 算法2: PPO

**快速测试 (5 episodes)**:
```bash
python examples/run_ppo_carla.py \
  --scenario s1 \
  --episodes 5 \
  --max-steps 100 \
  --output-dir outputs/s1_test/ppo \
  --no-rendering
```

**标准训练 (100 episodes)**:
```bash
python examples/run_ppo_carla.py \
  --scenario s1 \
  --episodes 100 \
  --max-steps 100 \
  --lr 3e-4 \
  --gamma 0.99 \
  --clip-epsilon 0.2 \
  --batch-size 64 \
  --output-dir outputs/s1_train/ppo \
  --log-dir logs/ppo_s1 \
  --save-interval 20 \
  --host localhost \
  --port 2000
```

**使用全局配置**:
```bash
python examples/run_ppo_carla.py \
  --scenario s1 \
  --episodes 50 \
  --use-global-config \
  --output-dir outputs/s1_train/ppo \
  --no-rendering
```

---

### ✅ 算法3: SAC

**快速测试 (5 episodes)**:
```bash
python examples/run_sac_carla.py \
  --scenario s1 \
  --episodes 5 \
  --max-steps 100 \
  --output-dir outputs/s1_test/sac \
  --no-rendering
```

**标准训练 (100 episodes)**:
```bash
python examples/run_sac_carla.py \
  --scenario s1 \
  --episodes 100 \
  --max-steps 100 \
  --lr 3e-4 \
  --gamma 0.99 \
  --tau 0.005 \
  --batch-size 256 \
  --buffer-size 100000 \
  --output-dir outputs/s1_train/sac \
  --log-dir logs/sac_s1 \
  --save-interval 20 \
  --host localhost \
  --port 2000
```

**使用全局配置**:
```bash
python examples/run_sac_carla.py \
  --scenario s1 \
  --episodes 50 \
  --use-global-config \
  --output-dir outputs/s1_train/sac \
  --no-rendering
```

---

### ✅ 算法4: Rainbow DQN

**快速测试 (5 episodes)**:
```bash
python examples/run_rainbow_dqn_carla.py \
  --scenario s1 \
  --episodes 5 \
  --max-steps 100 \
  --output-dir outputs/s1_test/rainbow_dqn \
  --no-rendering
```

**标准训练 (100 episodes)**:
```bash
python examples/run_rainbow_dqn_carla.py \
  --scenario s1 \
  --episodes 100 \
  --max-steps 100 \
  --lr 6.25e-5 \
  --gamma 0.99 \
  --batch-size 32 \
  --buffer-size 100000 \
  --output-dir outputs/s1_train/rainbow_dqn \
  --log-dir logs/rainbow_dqn_s1 \
  --save-interval 20 \
  --host localhost \
  --port 2000
```

**使用全局配置**:
```bash
python examples/run_rainbow_dqn_carla.py \
  --scenario s1 \
  --episodes 50 \
  --use-global-config \
  --output-dir outputs/s1_train/rainbow_dqn \
  --no-rendering
```

**关键特性**:
- 结合6种DQN改进: Double DQN, Dueling, PER, Multi-step, C51, Noisy Nets
- 分布式值函数估计
- 优先经验回放

---

### ✅ 算法5: RCRL

**快速测试 (5 episodes, 软约束)**:
```bash
python examples/run_rcrl_carla.py \
  --scenario s1 \
  --episodes 5 \
  --max-steps 100 \
  --constraint-mode soft \
  --output-dir outputs/s1_test/rcrl \
  --no-rendering
```

**标准训练 (100 episodes, 软约束)**:
```bash
python examples/run_rcrl_carla.py \
  --scenario s1 \
  --episodes 100 \
  --max-steps 100 \
  --constraint-mode soft \
  --lr 3e-4 \
  --gamma 0.99 \
  --batch-size 64 \
  --buffer-size 50000 \
  --output-dir outputs/s1_train/rcrl_soft \
  --log-dir logs/rcrl_s1_soft \
  --save-interval 20 \
  --host localhost \
  --port 2000
```

**标准训练 (100 episodes, 硬约束)**:
```bash
python examples/run_rcrl_carla.py \
  --scenario s1 \
  --episodes 100 \
  --max-steps 100 \
  --constraint-mode hard \
  --lr 3e-4 \
  --gamma 0.99 \
  --batch-size 64 \
  --buffer-size 50000 \
  --output-dir outputs/s1_train/rcrl_hard \
  --log-dir logs/rcrl_s1_hard \
  --save-interval 20 \
  --host localhost \
  --port 2000
```

**使用全局配置**:
```bash
python examples/run_rcrl_carla.py \
  --scenario s1 \
  --episodes 50 \
  --use-global-config \
  --constraint-mode soft \
  --output-dir outputs/s1_train/rcrl \
  --no-rendering
```

**关键特性**:
- 前向可达集计算
- 硬约束: 过滤不安全动作
- 软约束: 安全性惩罚项
- 实时安全验证

---

## 批量测试脚本

### 方法1: 顺序测试所有算法

创建 `test_all_baselines_s1.sh`:

```bash
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
ls -lh $OUTPUT_BASE
```

**运行方式**:
```bash
chmod +x test_all_baselines_s1.sh
./test_all_baselines_s1.sh
```

---

### 方法2: 手动依次运行

```bash
# 1. C2OSR
python examples/run_c2osr_carla.py --scenario s1 --episodes 10 --output-dir outputs/s1_comparison/c2osr

# 2. PPO
python examples/run_ppo_carla.py --scenario s1 --episodes 10 --max-steps 100 --output-dir outputs/s1_comparison/ppo --no-rendering

# 3. SAC
python examples/run_sac_carla.py --scenario s1 --episodes 10 --max-steps 100 --output-dir outputs/s1_comparison/sac --no-rendering

# 4. Rainbow DQN
python examples/run_rainbow_dqn_carla.py --scenario s1 --episodes 10 --max-steps 100 --output-dir outputs/s1_comparison/rainbow_dqn --no-rendering

# 5. RCRL
python examples/run_rcrl_carla.py --scenario s1 --episodes 10 --max-steps 100 --constraint-mode soft --output-dir outputs/s1_comparison/rcrl --no-rendering
```

---

## 结果查看

### 输出目录结构

```
outputs/s1_comparison/
├── c2osr/
│   ├── episode_0/
│   ├── episode_1/
│   ├── ...
│   ├── metrics.json
│   └── summary.txt
├── ppo/
│   ├── checkpoints/
│   │   ├── ppo_episode_20.pt
│   │   └── ppo_final.pt
│   ├── tensorboard/
│   └── metrics.json
├── sac/
│   ├── checkpoints/
│   │   ├── sac_episode_20.pt
│   │   └── sac_final.pt
│   ├── tensorboard/
│   └── metrics.json
├── rainbow_dqn/
│   ├── checkpoints/
│   │   ├── rainbow_dqn_episode_20.pt
│   │   └── rainbow_dqn_final.pt
│   ├── tensorboard/
│   └── metrics.json
└── rcrl/
    ├── checkpoints/
    │   ├── rcrl_episode_20.pt
    │   └── rcrl_final.pt
    ├── tensorboard/
    └── metrics.json
```

### 查看TensorBoard日志

```bash
# PPO
tensorboard --logdir logs/ppo_s1 --port 6006

# SAC
tensorboard --logdir logs/sac_s1 --port 6007

# Rainbow DQN
tensorboard --logdir logs/rainbow_dqn_s1 --port 6008

# RCRL
tensorboard --logdir logs/rcrl_s1_soft --port 6009

# 查看所有算法对比
tensorboard --logdir logs/ --port 6010
```

在浏览器中访问: http://localhost:6006

---

## 问题排查

### 1. CARLA连接失败

```bash
# 检查CARLA是否运行
ps aux | grep Carla

# 测试连接
python examples/test_carla_connection.py

# 杀死卡死的CARLA进程
pkill -9 CarlaUE4
```

### 2. 列出所有可用场景

```bash
python examples/run_c2osr_carla.py --list-scenarios
```

输出示例:
```
Available Scenarios:
============================================================

s1_scenario:
  Description: 环境车逆行场景 - 对向逆行车辆切入本车道
  Map: Town03
  Difficulty: hard

s2_scenario:
  Description: 右侧车辆变道切入场景
  Map: Town03
  Difficulty: medium

...
```

### 3. 修改全局配置

编辑: `src/c2o_drive/config/global_config.py`

```python
@dataclass
class LatticeConfig:
    lateral_offsets: list = field(default_factory=lambda: [-3.0, -2.0, 0.0, 2.0, 3.0])
    speed_variations: list = field(default_factory=lambda: [4.0])
    dt: float = 1.0
    horizon: int = 10
```

### 4. 脚本执行权限问题

```bash
chmod +x examples/run_*.py
```

### 5. Python路径问题

确保从项目根目录运行:
```bash
cd /home/zwt/code/C2O-Drive
python examples/run_c2osr_carla.py --scenario s1 --episodes 5
```

---

## 算法对比表

| 算法 | 类型 | 动作空间 | 特点 | 适用场景 |
|------|------|----------|------|---------|
| **C2OSR** | Planning | 离散(lattice) | Dirichlet分布建模,不确定性量化 | 需要不确定性评估的场景 |
| **PPO** | RL (Policy Gradient) | 离散(lattice) | 稳定训练,clip机制 | 需要稳定学习的场景 |
| **SAC** | RL (Actor-Critic) | 离散(lattice) | 最大熵RL,探索性好 | 需要探索的复杂场景 |
| **Rainbow DQN** | RL (Value-based) | 离散(lattice) | 6种改进组合,分布式RL | 需要高效样本利用 |
| **RCRL** | RL (Safety-aware) | 离散(lattice) | 可达性约束,安全保证 | 需要安全保证的场景 |

---

## 性能指标

测试完成后，可以对比以下指标：

1. **成功率**: 无碰撞完成任务的episodes比例
2. **平均奖励**: 所有episodes的平均累积奖励
3. **平均步数**: 完成任务所需的平均步数
4. **碰撞率**: 发生碰撞的episodes比例
5. **安全违规** (RCRL): 安全约束违规次数
6. **学习效率**: 达到稳定性能所需的episodes数
7. **计算时间**: 每个episode的平均运行时间

---

## 下一步

1. 运行完整对比实验(100+ episodes)
2. 分析TensorBoard日志
3. 生成性能对比图表
4. 测试其他场景(S2, S3, S4)
5. 调整超参数优化性能

---

## 参考

- CARLA文档: https://carla.readthedocs.io
- 项目架构: `docs/ALGORITHM_ARCHITECTURE.md`
- 配置说明: `src/c2o_drive/config/global_config.py`
- 场景定义: `src/c2o_drive/environments/carla/scenarios.py`
