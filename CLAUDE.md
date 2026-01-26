# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

C2O-Drive 是一个自动驾驶决策框架，实现并比较多种算法（C2OSR, PPO, SAC, DQN, Rainbow DQN, RCRL）。支持 CARLA 模拟器集成和无需 CARLA 的轻量级场景回放测试。

## 常用命令

### 运行示例

**无需 CARLA（场景回放）：**
```bash
python examples/run_c2osr_scenario.py --episodes 10 --max-steps 50
python examples/run_c2osr_scenario.py --config-preset high-precision --horizon 15
```

**使用 CARLA：**
```bash
python examples/run_c2osr_carla.py --scenario s4_wrong_way --episodes 100
python examples/run_ppo_carla.py --scenario s4_wrong_way --episodes 1000
python examples/run_sac_carla.py --scenario s4 --episodes 1000
```

### 测试
```bash
pytest src/c2o_drive/tests/                # 所有测试
pytest src/c2o_drive/tests/ -m unit        # 仅单元测试
pytest src/c2o_drive/tests/ -m functional  # 功能测试
pytest src/c2o_drive/tests/ -m integration # 集成测试
```

## 架构

### 两种算法模式

**Planner 模式（推荐）：** 统一接口，继承自 `EpisodicAlgorithmPlanner[WorldState, EgoControl]`
- 使用该模式的算法：C2OSR, PPO, Rainbow DQN, RCRL
- 接口方法：`select_action()`, `update()`, `reset()`, `plan_trajectory()`
- 自动处理轨迹生成和 buffer 管理

**Agent 模式（传统 RL）：** 独立 Agent 类，需要手动集成 `LatticePlanner`
- 使用该模式的算法：SAC, DQN

### 核心类型 (`src/c2o_drive/core/types.py`)
- `WorldState`：观测状态，包含自车和周围智能体
- `EgoState` / `AgentState`：车辆状态（位置、航向、速度）
- `EgoControl`：路点控制命令

### 数据流
```
WorldState → Algorithm.select_action() → LatticePlanner 生成候选轨迹
           → Q 值评估 → EgoControl → Environment.step() → Transition
           → Algorithm.update()
```

### 目录结构
- `src/c2o_drive/algorithms/`：算法实现（c2osr, ppo, sac, dqn, rainbow_dqn, rcrl）
- `src/c2o_drive/core/`：接口和类型定义
- `src/c2o_drive/environments/`：CARLA 封装、场景回放、网格世界
- `src/c2o_drive/config/`：配置管理（global_config.py 是参数的唯一真实源）
- `src/c2o_drive/utils/`：Lattice 规划器、碰撞检测、几何工具
- `examples/`：训练和测试脚本

## 配置管理

**修改参数请编辑 `src/c2o_drive/config/global_config.py`** - 这是所有默认值的唯一真实源。

使用工厂方法创建算法配置：
```python
from c2o_drive.algorithms.c2osr import C2OSRPlannerConfig
config = C2OSRPlannerConfig.from_global_config()
config.horizon = 20  # 为特定实验覆盖参数
```

## 添加新算法

推荐使用 Planner 模式：
```python
from c2o_drive.algorithms.base import EpisodicAlgorithmPlanner
from c2o_drive.core.types import WorldState, EgoControl

class MyPlanner(EpisodicAlgorithmPlanner[WorldState, EgoControl]):
    def select_action(self, observation, deterministic=False, **kwargs) -> EgoControl:
        pass
    def update(self, transition: Transition) -> UpdateMetrics:
        pass
    def reset(self) -> None:
        pass
```

## 测试标记

测试使用 pytest 标记：`unit`, `functional`, `integration`, `slow`, `carla`, `gpu`
