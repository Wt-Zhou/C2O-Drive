# C2OSR 示例脚本

本目录包含使用新架构运行 C2OSR 算法的示例脚本。

## run_c2osr_scenario.py

使用新架构的 C2OSRPlanner 和 ScenarioReplayEnvironment 运行实验。

### 功能特点

- ✅ 使用新架构的 `C2OSRPlanner` (来自 `algorithms/c2osr/`)
- ✅ 使用 `ScenarioReplayEnvironment` (Gym 标准接口)
- ✅ 使用与 `run_sim_cl_simple.py` 相同的 `ScenarioManager` 场景
- ✅ 支持多 episodes 运行
- ✅ 配置预设支持 (fast/default/high-precision)
- ✅ 详细的统计输出
- ✅ 无需 CARLA 依赖

### 快速开始

```bash
# 快速测试（2个episodes，快速配置）
python examples/run_c2osr_scenario.py --episodes 2 --max-steps 10 --config-preset fast

# 标准运行（10个episodes，默认配置）
python examples/run_c2osr_scenario.py --episodes 10 --max-steps 50

# 高精度运行（更多采样，更细网格）
python examples/run_c2osr_scenario.py --episodes 20 --config-preset high-precision --horizon 15
```

### 命令行参数

#### 基本参数
- `--episodes N`: 运行的 episode 数量（默认：10）
- `--max-steps N`: 每个 episode 的最大步数（默认：50）
- `--seed N`: 随机种子（默认：2025）

#### 场景参数
- `--reference-path-mode MODE`: 参考路径模式
  - `straight`: 直线路径（默认）
  - `curve`: 圆弧路径
  - `s_curve`: S 型曲线

#### 配置预设
- `--config-preset PRESET`: 选择配置预设
  - `fast`: 快速测试（少采样，3条轨迹）
  - `default`: 默认配置（中等采样，5条轨迹）
  - `high-precision`: 高精度（多采样，10条轨迹）

#### 规划参数
- `--horizon N`: 规划时域/步数（默认：10）
- `--dt SECONDS`: 时间步长（默认：0.5秒）
- `--grid-size METERS`: 网格范围（默认：50米）

#### 输出参数
- `--output-dir PATH`: 输出目录（默认：`outputs/c2osr_scenario`）
- `--quiet`: 静默模式，减少输出

### 使用示例

#### 1. 测试不同参考路径

```bash
# 直线路径
python examples/run_c2osr_scenario.py --reference-path-mode straight

# 圆弧路径
python examples/run_c2osr_scenario.py --reference-path-mode curve

# S型曲线
python examples/run_c2osr_scenario.py --reference-path-mode s_curve
```

#### 2. 调整规划参数

```bash
# 更长的规划时域
python examples/run_c2osr_scenario.py --horizon 20 --dt 0.3

# 更大的搜索范围
python examples/run_c2osr_scenario.py --grid-size 100.0
```

#### 3. 性能对比

```bash
# 快速配置
python examples/run_c2osr_scenario.py --episodes 20 --config-preset fast

# 默认配置
python examples/run_c2osr_scenario.py --episodes 20 --config-preset default

# 高精度配置
python examples/run_c2osr_scenario.py --episodes 20 --config-preset high-precision
```

### 输出说明

脚本会输出以下统计信息：

```
======================================================================
 实验汇总
======================================================================

Episodes: 10
成功率: 30.0% (3 成功, 5 碰撞, 2 超时)

平均奖励: 25.42 ± 8.33
平均步数: 35.2
总耗时: 125.45s
平均速度: 2.8 steps/s

最佳 Episode: #3
  奖励: 45.23, 步数: 50, 结果: success

最差 Episode: #7
  奖励: 8.12, 步数: 12, 结果: collision
```

### 与原版 run_sim_cl_simple.py 的对比

| 特性 | run_sim_cl_simple.py | run_c2osr_scenario.py |
|------|---------------------|----------------------|
| 架构 | 直接使用原始组件 | 使用新架构适配器 |
| 环境接口 | 自定义轨迹回放 | Gym 标准 (step/reset) |
| 场景来源 | ScenarioManager | ✅ ScenarioManager（相同）|
| 可视化 | ✅ GIF 生成 | ⚠️ 暂未实现 |
| Checkpoint | ✅ 支持 | ⚠️ 暂未实现 |
| 代码行数 | ~539 行 | ~420 行 |
| 易用性 | 复杂 | ✅ 简化 |

### 架构说明

```
run_c2osr_scenario.py
    ↓
C2OSRPlanner (algorithms/c2osr/planner.py)
    ↓ 调用
原始 C2OSR 组件 (agents/c2osr/)
    ├─ LatticePlanner
    ├─ QValueCalculator
    ├─ DirichletBank
    └─ ...

ScenarioReplayEnvironment (environments/scenario_replay_env.py)
    ↓ 使用
ScenarioManager (env/scenario_manager.py)
    ├─ create_scenario()
    └─ generate_reference_path()
```

### 后续增强

可以进一步添加的功能：

- [ ] GIF 可视化生成
- [ ] Checkpoint 保存/恢复
- [ ] Q 值演化图表
- [ ] 碰撞率演化分析
- [ ] 多场景批量测试
- [ ] 与其他算法的对比

### 故障排查

#### Q: 运行时报 ImportError

确保在项目根目录运行，或者 Python 路径包含项目根目录：

```bash
cd /path/to/C2O-Drive
python examples/run_c2osr_scenario.py ...
```

#### Q: 性能较慢

尝试使用快速配置：

```bash
python examples/run_c2osr_scenario.py --config-preset fast --horizon 5
```

#### Q: 碰撞率很高

- 增加规划时域：`--horizon 15`
- 使用高精度配置：`--config-preset high-precision`
- 增加 Q 值采样数（修改配置）
