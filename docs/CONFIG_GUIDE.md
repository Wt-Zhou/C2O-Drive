# C2O-Drive 配置指南

## 快速指南：改参数去哪里？

### 🎯 99%的情况：修改 `global_config.py`

**文件位置**: `/home/zwt/code/C2O-Drive/carla_c2osr/config/global_config.py`

这是你的**唯一真实参数源**，包括：
- 时间参数（dt, horizon）
- 采样参数（samples数量）
- 网格尺寸（grid size, cell size）
- Dirichlet参数（alpha, learning rate）
- Reward权重（collision, speed, offset等）
- 所有算法超参数

### 📌 1%高级用法：代码中覆盖

仅在需要运行时动态修改时使用：
```python
from carla_c2osr.algorithms.c2osr import C2OSRPlannerConfig

# 从global读取默认值
config = C2OSRPlannerConfig.from_global_config()

# 覆盖特定参数
config.horizon = 20  # 仅本次运行使用20
planner = C2OSRPlanner(config)
```

## 两个配置文件的关系

```
┌────────────────────┐       from_global_config()       ┌─────────────────────┐
│  global_config.py  │ ────────────────────────────────> │  c2osr/config.py    │
│  （默认值源头）    │                                    │  （运行时配置）     │
└────────────────────┘                                    └─────────────────────┘
         ↑                                                           │
         │                                                           ↓
    用户修改这里                                             组件使用这个配置
```

## 常见参数修改示例

### 1. 修改预测horizon

**文件**: `global_config.py`
```python
@dataclass
class TimeConfig:
    default_horizon: int = 15  # 改为15步预测（原来是10）
```

### 2. 修改网格尺寸

**文件**: `global_config.py`
```python
@dataclass
class GridConfig:
    grid_size_m: float = 150.0  # 改为150米（原来是100）
    x_min: float = -75.0        # 相应调整边界
    x_max: float = 75.0
```

### 3. 修改Reward权重

**文件**: `global_config.py`
```python
@dataclass
class RewardConfig:
    collision_penalty: float = -100.0  # 加大碰撞惩罚（原来是-50）
    centerline_offset_penalty_weight: float = 0.2  # 启用中心线惩罚
```

### 4. 修改Dirichlet学习率

**文件**: `global_config.py`
```python
@dataclass
class DirichletConfig:
    learning_rate: float = 0.5  # 降低学习率（原来是1.0）
```

## 参数同步机制

### 自动同步的参数

以下参数会自动从`global_config`同步到算法配置：
- `horizon` → 同步到所有子模块
- `dt` → 时间步长
- `gamma` → 折扣因子
- 所有reward权重
- 所有Dirichlet参数

### 需要手动保持一致的参数

如果你直接创建`C2OSRPlannerConfig()`而不是用`from_global_config()`，需要注意：
- 默认horizon可能不同（global=10, algorithm=10，已统一）
- Lattice参数可能不同

## 配置最佳实践

### ✅ 推荐做法

1. **总是通过global_config修改默认值**
   ```python
   # 修改 global_config.py
   default_horizon: int = 20
   ```

2. **使用from_global_config()创建planner**
   ```python
   config = C2OSRPlannerConfig.from_global_config()
   planner = C2OSRPlanner(config)
   ```

3. **临时覆盖用代码**
   ```python
   config = C2OSRPlannerConfig.from_global_config()
   config.horizon = 25  # 仅本次实验
   ```

### ❌ 避免做法

1. **不要直接修改c2osr/config.py的默认值**
   - 这会导致与global_config不一致

2. **不要混用两种配置**
   ```python
   # 错误示例
   config = C2OSRPlannerConfig()  # 使用algorithm默认值
   # 但内部模块还在读global_config！
   ```

3. **不要忘记同步**
   - 如果必须直接创建config，记得手动同步关键参数

## 调试工具

### 检查配置一致性

运行以下命令检查两个配置是否一致：
```bash
python -c "from carla_c2osr.config.diagnostic import print_config_summary; print_config_summary()"
```

### 查看当前配置

```python
from carla_c2osr.config import get_global_config
gc = get_global_config()
print(f"Horizon: {gc.time.default_horizon}")
print(f"Grid size: {gc.grid.grid_size_m}m")
print(f"Collision penalty: {gc.reward.collision_penalty}")
```

## FAQ

### Q: 为什么有两个配置文件？

A: 历史原因。`global_config`服务老代码，`c2osr/config`适配新架构。我们保留两个以保持向后兼容。

### Q: 我应该改哪个？

A: 99%情况改`global_config.py`。它是默认值的唯一源头。

### Q: 两个配置不一致怎么办？

A: 运行诊断工具检查，然后统一到`global_config`的值。

### Q: 如何确保修改生效？

A: 使用`from_global_config()`创建配置，这会自动读取最新的global值。

## 完整参数列表

请参考`global_config.py`中的注释，每个参数都有详细说明。