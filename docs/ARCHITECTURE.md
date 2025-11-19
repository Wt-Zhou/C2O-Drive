# C2O-Drive 新架构文档

## 概述

C2O-Drive已经重构为一个清晰、模块化的架构，实现了算法、环境和工具之间的完全解耦。

## 新目录结构

```
src/c2o_drive/          # 主包（更Python化的命名）
│
├── core/               # 核心接口和类型
│   ├── types.py       # WorldState, EgoState, AgentState等基础类型
│   └── interfaces.py  # Algorithm, Environment等接口定义
│
├── algorithms/         # 所有算法实现
│   ├── c2osr/        # C2OSR算法（完整实现）
│   │   ├── planner.py         # 主规划器
│   │   ├── q_value.py         # Q值计算（核心算法）
│   │   ├── dirichlet.py       # Dirichlet过程
│   │   ├── trajectory_buffer.py # 轨迹存储
│   │   ├── grid_mapper.py     # 网格映射
│   │   ├── rewards.py         # 奖励函数
│   │   └── config.py          # 配置
│   │
│   ├── dqn/          # DQN实现（待完善）
│   └── sac/          # SAC实现（待完善）
│
├── environments/      # 环境封装
│   ├── carla/        # CARLA仿真器
│   └── virtual/      # 虚拟环境（网格世界等）
│
├── utils/            # 工具函数
│   ├── collision.py  # 碰撞检测
│   ├── geometry.py   # 几何计算
│   └── trajectory.py # 轨迹生成
│
├── config/           # 配置管理
├── visualization/    # 可视化工具
└── runners/          # 执行脚本
```

## 主要改进

### 1. 核心算法组件位置修正

**之前的问题**：
- Q值计算器（651行核心算法）错误地放在了`evaluation/`文件夹
- C2OSR组件分散在`agents/`、`evaluation/`、`algorithms/`三个文件夹

**现在的解决方案**：
- 所有C2OSR核心组件集中在`algorithms/c2osr/`
- Q值计算器正确地作为算法核心放在`algorithms/c2osr/q_value.py`

### 2. 清晰的模块边界

**算法层** (`algorithms/`)
- 实现具体的决策算法
- 不依赖特定环境
- 通过标准接口与环境交互

**环境层** (`environments/`)
- 封装不同的仿真器和环境
- 提供统一的Gym接口
- 隐藏环境特定的实现细节

**工具层** (`utils/`)
- 通用工具函数
- 不依赖具体算法或环境
- 可被任何模块使用

### 3. 消除重复

**之前**：
- 两个env文件夹：`env/`和`environments/`
- 两个tests文件夹
- 重复的rewards.py文件

**现在**：
- 统一的环境文件夹：`environments/`
- 统一的测试文件夹：`tests/`
- 单一的奖励函数位置：`algorithms/c2osr/rewards.py`

## 迁移指南

### 更新导入语句

使用提供的迁移脚本自动更新：

```bash
python scripts/migrate_imports.py your_file.py
```

### 手动更新示例

**旧的导入**：
```python
from carla_c2osr.env.types import WorldState
from c2o_drive.algorithms.c2osr.q_value import QValueCalculator
from carla_c2osr.agents.c2osr.spatial_dirichlet import DirichletParams
```

**新的导入**：
```python
from c2o_drive.core.types import WorldState
from c2o_drive.algorithms.c2osr.q_value import QValueCalculator
from c2o_drive.algorithms.c2osr.dirichlet import DirichletParams
```

## 关键文件映射

| 旧位置 | 新位置 | 说明 |
|--------|--------|------|
| `carla_c2osr/env/types.py` | `c2o_drive/core/types.py` | 核心类型定义 |
| `carla_c2osr/evaluation/q_value_calculator.py` | `c2o_drive/algorithms/c2osr/q_value.py` | Q值计算（核心算法） |
| `carla_c2osr/agents/c2osr/spatial_dirichlet.py` | `c2o_drive/algorithms/c2osr/dirichlet.py` | Dirichlet过程 |
| `carla_c2osr/agents/c2osr/trajectory_buffer.py` | `c2o_drive/algorithms/c2osr/trajectory_buffer.py` | 轨迹缓存 |
| `carla_c2osr/agents/c2osr/grid.py` | `c2o_drive/algorithms/c2osr/grid_mapper.py` | 网格映射 |
| `carla_c2osr/evaluation/rewards.py` | `c2o_drive/algorithms/c2osr/rewards.py` | 奖励函数 |
| `carla_c2osr/evaluation/collision_detector.py` | `c2o_drive/utils/collision.py` | 碰撞检测工具 |

## 使用新架构

### 导入C2OSR算法

```python
from c2o_drive.algorithms.c2osr import (
    C2OSRPlanner,
    C2OSRPlannerConfig,
    QValueCalculator,
    DirichletParams
)
```

### 导入核心类型

```python
from c2o_drive.core import (
    WorldState,
    EgoState,
    AgentState,
    EgoControl
)
```

### 创建算法实例

```python
# 创建配置
config = C2OSRPlannerConfig.from_global_config()

# 创建规划器
planner = C2OSRPlanner(config)

# 使用规划器
action = planner.select_action(world_state)
```

## 下一步计划

1. **完成内部导入更新**：运行迁移脚本更新所有文件中的导入语句
2. **实现DQN/SAC**：完成基线算法的真正实现
3. **统一配置系统**：使用YAML配置替代当前的双配置系统
4. **添加更多测试**：确保新架构的稳定性
5. **性能优化**：利用清晰的架构进行性能优化

## 架构优势

1. **可维护性**：代码位置符合直觉，易于找到和修改
2. **可扩展性**：添加新算法或环境只需在对应文件夹创建新模块
3. **可测试性**：清晰的模块边界使单元测试更容易编写
4. **代码复用**：通用工具和接口可被多个算法/环境共享
5. **团队协作**：不同团队可以独立开发算法和环境模块

## 兼容性说明

在迁移期间，旧的导入路径仍然可以工作（会显示弃用警告）。这通过`carla_c2osr/compatibility.py`模块实现。建议尽快更新到新的导入路径。

---

*更新日期：2024*
*版本：2.0.0*
