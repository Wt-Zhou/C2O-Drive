# 数据匹配诊断工具使用指南

## 问题背景

你报告的问题：
1. 某些episode在中间某些帧会匹配不上历史数据
2. 导致Dirichlet分布使用初始化值（先验）而非历史数据更新
3. 越早的timestep匹配到的数据越多

根据代码分析，我已经识别出三个可能的根本原因：

### 根本原因分析

**原因1: Padding策略导致轨迹失真** (最有可能)
- **位置**: `src/c2o_drive/algorithms/c2osr/trajectory_buffer.py:563-570`
- **问题**: 存储轨迹时，如果episode长度不足horizon，会用最后位置重复填充
- **后果**: 后期timestep的查询轨迹是动态的，但存储的历史轨迹是静态重复的，导致距离超阈值

**原因2: 后期timestep历史数据不足**
- **问题**: 如果大多数episode长度<8步，那么timestep=9的数据就很少
- **后果**: 索引过滤后没有候选数据可匹配

**原因3: 轨迹累积误差**
- **问题**: 不同episode中，agent行为的微小差异会随时间累积
- **后果**: 早期timestep相似，后期diverge导致超过匹配阈值(5m)

---

## 诊断工具使用

### 前提条件

**你需要先生成一些历史数据:**

```bash
# 运行几个episode收集数据
python examples/run_c2osr_carla.py --scenario s4 --episodes 10
```

这会生成buffer数据文件（通常在 `data/` 或 `checkpoints/` 目录）

---

### 步骤1: 运行完整诊断

```bash
cd /home/zwt/code/C2O-Drive
python diagnostics/run_full_diagnostic.py
```

**输出内容**:
1. Buffer基本统计（总episode数、平均长度）
2. 每个timestep的数据可用性（有多少episode、padding比例）
3. 匹配成功率分析（每个timestep的成功率、失败原因）
4. 可视化图表（保存在 `diagnostics/results/` 目录）

**示例输出**:
```
======================================================================
Buffer 数据统计摘要
======================================================================

[基本信息]
总Episode数: 127
平均Episode长度: 6.3 步

[每个Timestep的数据可用性]
Timestep   Episodes     Padded     Padding%     状态
----------------------------------------------------------------------
1          127          0          0.0          ✓ 正常
2          125          0          0.0          ✓ 正常
3          118          6          5.1          ✓ 正常
4          95           14         14.7         ✓ 正常
5          71           23         32.4         ⚠️  轻度padding
6          43           25         58.1         ❌ 严重padding     <-- 问题开始
7          28           21         75.0         ❌ 严重padding
...

[匹配成功率统计]
Timestep   总查询    成功    成功率     平均Cells    主要失败原因
--------------------------------------------------------------------------------
1          10        9       ✓ 90.0%   45.2         -
2          10        8       ✓ 80.0%   38.1         -
3          10        7       ✓ 70.0%   25.3         -
4          10        5       ⚠️  50.0%   12.4         action_dist_too_large (3次)
5          10        3       ⚠️  30.0%   5.2          action_dist_too_large (5次)
6          10        1       ❌ 10.0%   0.8          no_spatial_match (7次)   <-- 匹配崩溃
```

---

### 步骤2: 查看可视化图表

生成的图表会保存在 `diagnostics/results/` 目录:

1. **matching_success_rate.png**:
   - 上图: 每个timestep的匹配成功率曲线
   - 下图: 平均匹配到的cells数量

2. **data_availability.png**:
   - 上图: 每个timestep可用的历史episode数量（柱状图）
   - 下图: 每个timestep的padding比例

3. **failure_reasons.png**:
   - 堆叠柱状图显示每个timestep的失败原因分布

---

## 根据诊断结果修复问题

### 场景A: 严重Padding问题

**诊断特征**:
- `data_availability.png` 显示后期timestep padding% > 60%
- 失败原因主要是 `action_dist_too_large`

**修复方案**: 改进padding策略

```python
# 文件: src/c2o_drive/algorithms/c2osr/trajectory_buffer.py
# 方法: store_episode_trajectories_by_timestep()
# 当前行号: 约563-570

# === 当前代码（问题） ===
for action_t in range(self.horizon):
    traj_idx = min(timestep + action_t, len(ego_trajectory) - 1)  # 重复最后位置
    ego_action.append(tuple(ego_trajectory[traj_idx]))

# === 修复方案：速度外推 ===
for action_t in range(self.horizon):
    idx = timestep + action_t
    if idx < len(ego_trajectory):
        ego_action.append(tuple(ego_trajectory[idx]))
    else:
        # 使用最后两个点的速度外推
        last_pos = ego_trajectory[-1]
        if len(ego_trajectory) >= 2:
            second_last = ego_trajectory[-2]
            vx = last_pos[0] - second_last[0]
            vy = last_pos[1] - second_last[1]
            steps_beyond = idx - (len(ego_trajectory) - 1)
            extrapolated = (last_pos[0] + vx * steps_beyond,
                          last_pos[1] + vy * steps_beyond)
            ego_action.append(extrapolated)
        else:
            ego_action.append(last_pos)  # Fallback
```

---

### 场景B: 历史数据不足

**诊断特征**:
- `data_availability.png` 显示后期timestep的episode数 < 20
- 失败原因主要是 `no_spatial_match` 或 `no_data_for_X_agents`

**修复方案1**: 增加episode长度

```python
# 文件: examples/run_c2osr_carla.py 或相关配置
# 或者: src/c2o_drive/config/global_config.py CarlaConfig类

# 当前
max_episode_steps = 500  # 但实际可能更早终止

# 修改运行参数
# 让更多episode运行到后期timestep
```

**修复方案2**: 自适应匹配阈值

```python
# 文件: src/c2o_drive/algorithms/c2osr/trajectory_buffer.py
# 方法: get_agent_historical_transitions_strict_matching()

# 在匹配时，根据timestep动态调整阈值
def _get_adaptive_threshold(self, base_threshold, timestep):
    """后期timestep使用更宽松的阈值"""
    return base_threshold * (1.0 + 0.1 * min(timestep, 5))  # 最多增加50%

# 然后在匹配代码中使用：
ego_threshold = self._get_adaptive_threshold(config.matching.ego_state_threshold, timestep)
action_threshold = self._get_adaptive_threshold(config.matching.ego_action_threshold, timestep)
```

---

### 场景C: Action距离略超阈值

**诊断特征**:
- 距离统计显示 `action_dist_mean` 接近阈值(5.0m)，比如4.8m
- 只是略微超阈值导致匹配失败

**修复方案**: 增加阈值

```python
# 文件: src/c2o_drive/config/global_config.py
# 类: MatchingConfig

@dataclass
class MatchingConfig:
    ego_state_threshold: float = 5.0        # 保持不变
    agents_state_threshold: float = 5.0     # 保持不变
    ego_action_threshold: float = 7.0       # 从5.0增加到7.0
```

---

## 验证修复效果

修复后，再次运行诊断：

```bash
# 1. 清空旧buffer（可选）
rm -f data/trajectory_buffer.pkl checkpoints/trajectory_buffer.pkl

# 2. 用修复后的代码生成新数据
python examples/run_c2osr_carla.py --scenario s4 --episodes 20

# 3. 运行诊断
python diagnostics/run_full_diagnostic.py

# 4. 对比前后的匹配成功率
```

**期望结果**:
- 后期timestep的匹配成功率从 <20% 提升到 >50%
- Padding比例下降
- 平均匹配cells数增加

---

## 独立工具说明

### buffer_inspector.py - 只查看数据质量

```bash
python diagnostics/buffer_inspector.py
```

**用途**: 快速查看buffer内容和质量，不进行匹配分析

### analyze_matching_issue.py - 只分析匹配问题

```bash
python diagnostics/analyze_matching_issue.py
```

**用途**: 详细分析匹配失败原因，不生成可视化

---

## 当前状态说明

**注意**: 由于当前没有历史数据文件，诊断脚本会提示：
```
❌ 错误: Buffer文件不存在
建议: 先运行 run_c2osr_carla.py 生成历史数据
```

**下一步操作**:
1. 运行CARLA环境收集数据：
   ```bash
   python examples/run_c2osr_carla.py --scenario s4 --episodes 10
   ```

2. 确认buffer文件生成位置（脚本会自动查找以下路径）:
   - `data/trajectory_buffer.pkl`
   - `checkpoints/trajectory_buffer.pkl`
   - `results/trajectory_buffer.pkl`

3. 运行诊断：
   ```bash
   python diagnostics/run_full_diagnostic.py
   ```

4. 根据诊断结果选择修复方案

---

## 总结

根据我的分析，你的问题很可能是由**padding策略**导致的。建议：

1. **优先修复**: 改进padding策略（速度外推代替重复位置）
2. **如果数据量少**: 增加episode数量或长度
3. **微调**: 适当放宽匹配阈值

所有诊断工具都是**只读的**，不会影响原有代码。你可以放心运行诊断，然后根据具体结果决定修复方案。
