# C2O-Drive 数据匹配诊断工具

这个目录包含用于诊断历史数据匹配问题的工具，**所有工具都是只读的，不会修改任何原有代码或数据**。

## 问题描述

在运行 `run_c2osr_carla` 时，可能会遇到以下问题：
- 某些timestep无法匹配到历史数据
- Dirichlet分布使用初始化值而非历史数据更新
- 早期timestep匹配率高，后期timestep匹配率低

## 工具列表

### 1. `run_full_diagnostic.py` - 完整诊断流程（推荐使用）

**用途**: 运行完整的诊断流程，生成报告和可视化

**使用方法**:
```bash
cd /home/zwt/code/C2O-Drive/diagnostics
python run_full_diagnostic.py
```

**输出**:
- 终端输出：Buffer统计、匹配成功率、失败原因分析、修复建议
- 图表文件（保存在 `diagnostics/results/` 目录）：
  - `matching_success_rate.png`: 每个timestep的匹配成功率曲线
  - `data_availability.png`: 数据可用性和padding比例
  - `failure_reasons.png`: 失败原因分布

---

### 2. `buffer_inspector.py` - Buffer内容检查

**用途**: 分析trajectory buffer的内容和数据质量

**使用方法**:
```bash
cd /home/zwt/code/C2O-Drive/diagnostics
python buffer_inspector.py
```

**输出信息**:
- 总episode数、平均长度、最长/最短episode
- Agent数量分布
- 每个timestep的数据可用性（有多少episode包含该timestep）
- Padding检测（识别被填充的轨迹）
- Ego action trajectory模式分析

**示例输出**:
```
======================================================================
Buffer 数据统计摘要
======================================================================

[基本信息]
总Episode数: 127
平均Episode长度: 6.3 步
最长Episode: 10 步
最短Episode: 3 步

[每个Timestep的数据可用性]
Timestep   Episodes     Padded     Padding%     状态
----------------------------------------------------------------------
1          127          0          0.0          ✓ 正常
2          125          0          0.0          ✓ 正常
3          118          6          5.1          ✓ 正常
4          95           14         14.7         ✓ 正常
5          71           23         32.4         ⚠️  轻度padding
6          43           25         58.1         ❌ 严重padding
...
```

---

### 3. `analyze_matching_issue.py` - 匹配问题诊断

**用途**: 模拟匹配过程，分析为什么某些timestep匹配失败

**使用方法**:
```bash
cd /home/zwt/code/C2O-Drive/diagnostics
python analyze_matching_issue.py
```

**输出信息**:
- 每个timestep的匹配统计（索引过滤、严格匹配、成功率）
- 失败原因分类（ego距离、agent距离、action距离、数据缺失等）
- 距离统计（平均值、标准差、最大值）
- 按timestep聚合的成功率曲线

**示例输出**:
```
Timestep   总查询    成功    成功率     平均Cells    主要失败原因
--------------------------------------------------------------------------------
1          10        9       ✓ 90.0%   45.2         -
2          10        8       ✓ 80.0%   38.1         -
3          10        7       ✓ 70.0%   25.3         -
4          10        5       ⚠️  50.0%   12.4         action_dist_too_large (3次)
5          10        3       ⚠️  30.0%   5.2          action_dist_too_large (5次)
6          10        1       ❌ 10.0%   0.8          no_spatial_match (7次)
```

---

### 4. `matching_visualizer.py` - 可视化工具

**用途**: 生成图表可视化匹配统计（通常由 `run_full_diagnostic.py` 自动调用）

**功能**:
- `plot_timestep_success_rate()`: 绘制成功率曲线
- `plot_data_availability()`: 绘制数据可用性柱状图
- `plot_failure_reasons()`: 绘制失败原因堆叠图

---

## 使用流程

### 快速诊断（推荐）

```bash
# 1. 运行完整诊断
cd /home/zwt/code/C2O-Drive/diagnostics
python run_full_diagnostic.py

# 2. 查看生成的图表
ls -lh results/

# 3. 根据终端输出的建议进行修复
```

### 详细分析

```bash
# 1. 先查看buffer内容
python buffer_inspector.py

# 2. 再分析匹配问题
python analyze_matching_issue.py

# 3. 根据输出判断问题类型
```

---

## 常见诊断结果和修复方案

### 问题1: 后期timestep严重padding（Padding% > 60%）

**症状**:
- `buffer_inspector.py` 显示后期timestep的padding比例很高
- `analyze_matching_issue.py` 显示 `action_dist_too_large` 失败

**原因**:
- 存储数据时，轨迹用最后位置重复填充
- 查询时，真实轨迹与填充轨迹不匹配

**修复方案**:
```python
# 文件: src/c2o_drive/algorithms/c2osr/trajectory_buffer.py
# 方法: store_episode_trajectories_by_timestep()

# 当前代码（问题）:
traj_idx = min(timestep + action_t, len(ego_trajectory) - 1)  # 重复最后位置

# 修复方案A: 速度外推
if timestep + action_t < len(ego_trajectory):
    ego_action.append(tuple(ego_trajectory[timestep + action_t]))
else:
    # 使用最后两个点的速度外推
    last_pos = ego_trajectory[-1]
    second_last = ego_trajectory[-2]
    velocity = (last_pos[0] - second_last[0], last_pos[1] - second_last[1])
    extrapolated = (last_pos[0] + velocity[0], last_pos[1] + velocity[1])
    ego_action.append(extrapolated)

# 修复方案B: 只存储有效长度
ego_action_valid_length = min(self.horizon, len(ego_trajectory) - timestep)
# 存储时记录实际长度，匹配时只比较有效部分
```

### 问题2: 历史数据不足（某些timestep的Episodes < 10）

**症状**:
- `buffer_inspector.py` 显示后期timestep的episode数很少
- `analyze_matching_issue.py` 显示 `no_spatial_match` 或 `no_data_for_X_agents`

**原因**:
- 大多数episode很短，没到达后期timestep就结束了
- Buffer中缺乏长episode数据

**修复方案**:
```python
# 方案1: 增加episode运行步数
# 文件: examples/run_c2osr_carla.py 或 config
max_episode_steps = 15  # 从10增加到15

# 方案2: 使用自适应阈值（后期放宽）
# 文件: src/c2o_drive/algorithms/c2osr/trajectory_buffer.py
def _get_adaptive_threshold(self, timestep, base_threshold):
    return base_threshold * (1 + 0.1 * timestep)  # 每个timestep增加10%
```

### 问题3: Action距离超阈值

**症状**:
- 距离统计显示 `action_dist_mean` 接近或超过阈值（默认5.0m）

**原因**:
- 轨迹随时间累积误差
- 不同episode的agent行为差异导致ego轨迹分叉

**修复方案**:
```python
# 方案1: 增加action阈值
# 文件: src/c2o_drive/config/global_config.py
ego_action_threshold: float = 7.0  # 从5.0增加到7.0

# 方案2: 使用timestep自适应阈值
# 在匹配时根据timestep动态调整
```

---

## 配置说明

诊断工具会自动读取以下配置:

```python
# src/c2o_drive/config/global_config.py
class GlobalConfig:
    # Buffer路径
    buffer_save_path: str = "data/trajectory_buffer.pkl"

    # 匹配阈值
    ego_state_threshold: float = 5.0  # ego位置阈值（米）
    agents_state_threshold: float = 5.0  # agent位置阈值（米）
    ego_action_threshold: float = 5.0  # ego action阈值（米）

    # Horizon设置
    horizon: int = 10  # 预测时域长度
```

---

## 注意事项

1. **不修改原代码**: 所有诊断工具都是只读的，不会修改任何原有代码或数据
2. **需要历史数据**: 运行前确保已经通过 `run_c2osr_carla.py` 生成了一些历史数据
3. **matplotlib依赖**: 可视化功能需要matplotlib，如果未安装会跳过图表生成
4. **诊断vs修复**: 这些工具只用于诊断问题，实际修复需要修改原代码

---

## 故障排除

### 问题: "Buffer文件不存在"

**解决**:
```bash
# 1. 检查配置文件中的buffer_save_path
# 2. 运行至少一个episode生成数据
python examples/run_c2osr_carla.py --scenario s4 --episodes 5
```

### 问题: "matplotlib未安装"

**解决**:
```bash
pip install matplotlib
```

### 问题: 导入错误

**解决**:
```bash
# 确保在diagnostics目录下运行
cd /home/zwt/code/C2O-Drive/diagnostics
python run_full_diagnostic.py
```

---

## 输出文件

运行诊断后会生成以下文件:

```
diagnostics/
├── results/
│   ├── matching_success_rate.png  # 成功率曲线
│   ├── data_availability.png       # 数据可用性分析
│   └── failure_reasons.png         # 失败原因分布
└── [诊断脚本]
```

---

如有问题，请查看终端输出的详细错误信息和建议。
