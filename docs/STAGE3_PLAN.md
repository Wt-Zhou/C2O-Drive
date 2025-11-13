# 阶段3实施计划: 算法适配器

**目标**: 将现有C2OSR算法包装到统一接口，实现算法与环境的完全解耦

**预计时间**: 2-3周

**状态**: 📝 规划中

---

## 📋 目录

1. [目标与范围](#目标与范围)
2. [现状分析](#现状分析)
3. [设计方案](#设计方案)
4. [实施步骤](#实施步骤)
5. [关键挑战](#关键挑战)
6. [测试策略](#测试策略)
7. [成功标准](#成功标准)

---

## 🎯 目标与范围

### 主要目标

1. **创建algorithms/模块结构**
   - 设计清晰的算法包装架构
   - 支持多种算法并存
   - 保持代码组织清晰

2. **包装C2OSR算法**
   - 实现`BasePlanner`接口
   - 实现`TrajectoryEvaluator`接口
   - 保持核心算法代码不变

3. **验证功能完整性**
   - 确保包装后功能完全一致
   - 验证性能无损
   - 保持向后兼容

### 范围边界

**包含**:
- ✅ C2OSR算法的接口包装
- ✅ Lattice规划器的包装
- ✅ Dirichlet Bank的包装
- ✅ Q值计算器的包装
- ✅ 轨迹缓冲区的包装
- ✅ 网格映射器的包装

**不包含**:
- ❌ 修改C2OSR核心算法逻辑
- ❌ 实现新的RL算法(DQN/SAC在阶段5)
- ❌ 环境相关修改(已在阶段2完成)

---

## 📊 现状分析

### 现有C2OSR代码结构

```
carla_c2osr/agents/c2osr/
├── grid.py                      (893行) - 网格映射与可达集
├── trajectory_buffer.py         (670行) - 轨迹缓冲区
├── spatial_dirichlet.py         (549行) - Dirichlet分布
├── sampling.py                  (81行)  - 采样
├── risk.py                      (81行)  - 风险计算
├── dp_mixture.py                (76行)  - DP混合
└── transition.py                (13行)  - 转移模型
```

### 现有运行脚本结构

```
carla_c2osr/runner/
├── run_sim_cl_simple.py         (538行) - 简化运行脚本
└── refactored/
    ├── episode_context.py       - Episode上下文
    ├── trajectory_evaluator.py  - 轨迹评估
    ├── timestep_executor.py     - 时间步执行
    ├── visualization_manager.py - 可视化
    └── data_manager.py          - 数据管理
```

### 现有工具模块

```
carla_c2osr/
├── utils/
│   └── lattice_planner.py       - Lattice规划器
└── evaluation/
    ├── q_value_calculator.py    - Q值计算
    ├── collision_detector.py    - 碰撞检测
    └── rewards.py               - 奖励计算
```

### 关键依赖关系

```
run_sim_cl_simple.py
    ├── LatticePlanner (utils/)
    ├── QValueCalculator (evaluation/)
    ├── SpatialDirichletBank (agents/c2osr/)
    ├── TrajectoryBuffer (agents/c2osr/)
    ├── GridMapper (agents/c2osr/)
    └── ScenarioManager (env/)
```

---

## 🏗️ 设计方案

### 目标架构

```
carla_c2osr/
├── core/                        [已完成] 核心接口
│   ├── environment.py
│   ├── planner.py
│   ├── evaluator.py
│   └── state_space.py
│
├── algorithms/                  [新增] 算法实现
│   ├── __init__.py
│   ├── base.py                  - 算法基类
│   │
│   └── c2osr/                   - C2OSR算法
│       ├── __init__.py
│       ├── planner.py           [新增] 规划器包装
│       ├── evaluator.py         [新增] 评估器包装
│       ├── config.py            [新增] 配置管理
│       │
│       ├── core/                [移动] 核心算法
│       │   ├── grid.py          [from agents/c2osr/]
│       │   ├── spatial_dirichlet.py
│       │   ├── trajectory_buffer.py
│       │   ├── sampling.py
│       │   ├── risk.py
│       │   ├── dp_mixture.py
│       │   └── transition.py
│       │
│       └── components/          [移动] 组件模块
│           ├── lattice.py       [from utils/lattice_planner.py]
│           └── q_calculator.py  [from evaluation/q_value_calculator.py]
│
├── environments/                [已完成] 环境实现
└── agents/                      [保留] 原有结构(兼容)
```

### 接口设计

#### 1. C2OSRPlanner (实现BasePlanner)

```python
class C2OSRPlanner(EpisodicPlanner[WorldState, Trajectory]):
    """C2OSR算法的统一接口包装

    封装了Lattice规划器、Dirichlet Bank、Q值计算器等组件。
    提供标准的select_action和update接口。
    """

    def __init__(self, env: DrivingEnvironment, config: C2OSRConfig):
        # 初始化所有C2OSR组件
        self.lattice_planner = LatticePlanner(...)
        self.q_calculator = QValueCalculator(...)
        self.bank = SpatialDirichletBank(...)
        self.buffer = TrajectoryBuffer(...)
        self.grid = GridMapper(...)

    def plan_trajectory(self, observation: WorldState, horizon: int) -> List[Trajectory]:
        """规划一条轨迹（C2OSR核心功能）"""
        # 1. 生成候选轨迹
        candidates = self.lattice_planner.generate_trajectories(...)

        # 2. 评估所有候选
        evaluations = [self.evaluate_trajectory(t, observation) for t in candidates]

        # 3. 选择最优
        best_idx = self._select_best(evaluations)
        return candidates[best_idx]

    def select_action(self, observation: WorldState, **kwargs) -> Trajectory:
        """选择动作（第一个时间步）"""
        trajectory = self.plan_trajectory(observation, kwargs.get('horizon', 10))
        return trajectory[0]  # 返回第一个动作

    def update(self, transition: Transition) -> UpdateMetrics:
        """更新Dirichlet后验"""
        # 1. 记录轨迹到buffer
        self.buffer.add_trajectory(...)

        # 2. 更新Dirichlet Bank
        self.bank.update_with_softcount(...)

        return UpdateMetrics(...)
```

#### 2. C2OSREvaluator (实现TrajectoryEvaluator)

```python
class C2OSREvaluator(TrajectoryEvaluator[WorldState, Trajectory]):
    """C2OSR的轨迹评估器

    封装Q值计算逻辑，使用Dirichlet后验和历史数据。
    """

    def __init__(self, config: C2OSRConfig):
        self.q_calculator = QValueCalculator(...)

    def evaluate(self, trajectory: Trajectory, context: EvaluationContext) -> EvaluationResult:
        """评估单条轨迹"""
        # 使用Q值计算器
        q_value, details = self.q_calculator.compute_q_value(
            current_world_state=context.current_state,
            ego_action_trajectory=trajectory,
            trajectory_buffer=context.custom['buffer'],
            grid=context.custom['grid'],
            bank=context.custom['bank'],
            ...
        )

        return EvaluationResult(
            q_value=q_value,
            reward_breakdown=details.get('reward_breakdown'),
            collision_probability=details.get('collision_prob'),
            ...
        )
```

#### 3. C2OSRConfig

```python
@dataclass
class C2OSRConfig:
    """C2OSR算法配置"""
    # Grid配置
    grid_size_m: float = 20.0
    grid_resolution_m: float = 0.5

    # Lattice规划器配置
    lateral_samples: int = 5
    speed_samples: int = 5
    horizon: int = 10
    dt: float = 0.1

    # Dirichlet配置
    alpha_in: float = 1.0
    alpha_out: float = 1.0
    delta: float = 0.1

    # Q值计算配置
    n_samples: int = 100
    percentile: float = 0.9

    # 奖励权重
    safety_weight: float = 10.0
    comfort_weight: float = 1.0
    efficiency_weight: float = 2.0

    # Buffer配置
    max_buffer_size: int = 10000
```

---

## 🔧 实施步骤

### 第1步: 创建基础结构 (第1天)

**任务**:
1. 创建 `algorithms/` 目录结构
2. 创建 `algorithms/base.py` 算法基类
3. 创建 `algorithms/c2osr/` 子模块
4. 创建配置类 `C2OSRConfig`

**验证**:
```python
from carla_c2osr.algorithms.c2osr import C2OSRConfig
config = C2OSRConfig()
print(config)  # 应该正常工作
```

---

### 第2步: 移动核心算法代码 (第2-3天)

**任务**:
1. 创建 `algorithms/c2osr/core/` 目录
2. 复制(不是移动)`agents/c2osr/` 下所有文件到 `algorithms/c2osr/core/`
3. 更新所有import路径
4. 验证模块导入正常

**文件移动清单**:
```bash
# 复制核心算法
cp carla_c2osr/agents/c2osr/*.py carla_c2osr/algorithms/c2osr/core/

# 复制组件
cp carla_c2osr/utils/lattice_planner.py carla_c2osr/algorithms/c2osr/components/lattice.py
cp carla_c2osr/evaluation/q_value_calculator.py carla_c2osr/algorithms/c2osr/components/q_calculator.py
```

**验证**:
```python
from carla_c2osr.algorithms.c2osr.core import GridMapper, SpatialDirichletBank
# 应该正常导入
```

---

### 第3步: 实现C2OSRPlanner (第4-6天)

**任务**:
1. 创建 `algorithms/c2osr/planner.py`
2. 实现 `C2OSRPlanner` 类
3. 实现 `plan_trajectory()` 方法
4. 实现 `select_action()` 方法
5. 实现 `update()` 方法
6. 实现 `reset()` 方法

**关键代码**:
```python
# algorithms/c2osr/planner.py

from carla_c2osr.core import EpisodicPlanner, Transition, UpdateMetrics
from carla_c2osr.env.types import WorldState, Trajectory
from carla_c2osr.algorithms.c2osr.core import (
    GridMapper, SpatialDirichletBank, TrajectoryBuffer
)
from carla_c2osr.algorithms.c2osr.components import (
    LatticePlanner, QValueCalculator
)

class C2OSRPlanner(EpisodicPlanner[WorldState, Trajectory]):
    """C2OSR算法实现"""

    def __init__(self, env, config: C2OSRConfig):
        self.env = env
        self.config = config

        # 初始化组件
        self._init_components()

    def _init_components(self):
        """初始化所有C2OSR组件"""
        # Grid mapper
        self.grid = GridMapper(
            grid_size_m=self.config.grid_size_m,
            resolution_m=self.config.grid_resolution_m
        )

        # Dirichlet bank
        self.bank = SpatialDirichletBank(
            alpha_in=self.config.alpha_in,
            alpha_out=self.config.alpha_out,
            delta=self.config.delta
        )

        # Trajectory buffer
        self.buffer = TrajectoryBuffer(
            max_size=self.config.max_buffer_size
        )

        # Lattice planner
        self.lattice_planner = LatticePlanner(
            lateral_samples=self.config.lateral_samples,
            speed_samples=self.config.speed_samples
        )

        # Q value calculator
        self.q_calculator = QValueCalculator(
            n_samples=self.config.n_samples,
            percentile=self.config.percentile
        )

    def plan_trajectory(self, observation: WorldState, horizon: int, **kwargs):
        """规划轨迹"""
        # 实现详细的规划逻辑
        pass

    def select_action(self, observation: WorldState, **kwargs):
        """选择动作"""
        pass

    def update(self, transition: Transition):
        """更新算法"""
        pass

    def reset(self):
        """重置状态"""
        pass
```

**验证**:
```python
from carla_c2osr.algorithms.c2osr import C2OSRPlanner, C2OSRConfig
from carla_c2osr.environments import SimpleGridEnvironment

env = SimpleGridEnvironment()
config = C2OSRConfig()
planner = C2OSRPlanner(env, config)

state, _ = env.reset()
action = planner.select_action(state)
# 应该返回有效动作
```

---

### 第4步: 实现C2OSREvaluator (第7-8天)

**任务**:
1. 创建 `algorithms/c2osr/evaluator.py`
2. 实现 `C2OSREvaluator` 类
3. 实现 `evaluate()` 方法
4. 实现 `evaluate_batch()` 方法

**关键代码**:
```python
# algorithms/c2osr/evaluator.py

from carla_c2osr.core import TrajectoryEvaluator, EvaluationContext, EvaluationResult

class C2OSREvaluator(TrajectoryEvaluator):
    """C2OSR轨迹评估器"""

    def __init__(self, config: C2OSRConfig):
        self.config = config
        self.q_calculator = QValueCalculator(...)

    def evaluate(self, trajectory, context):
        """评估轨迹"""
        # 调用Q值计算器
        q_value, details = self.q_calculator.compute_q_value(...)

        return EvaluationResult(
            q_value=q_value,
            reward_breakdown=details.reward_breakdown,
            ...
        )
```

---

### 第5步: 创建工厂函数和注册 (第9天)

**任务**:
1. 创建工厂函数 `create_c2osr_planner()`
2. 注册到全局planner registry
3. 实现便捷的创建接口

**关键代码**:
```python
# algorithms/c2osr/__init__.py

from carla_c2osr.core import register_planner
from carla_c2osr.algorithms.c2osr.planner import C2OSRPlanner
from carla_c2osr.algorithms.c2osr.config import C2OSRConfig

# 注册到全局registry
register_planner('c2osr', C2OSRPlanner)

def create_c2osr_planner(env, **kwargs):
    """便捷工厂函数"""
    config = C2OSRConfig(**kwargs)
    return C2OSRPlanner(env, config)

__all__ = [
    'C2OSRPlanner',
    'C2OSREvaluator',
    'C2OSRConfig',
    'create_c2osr_planner',
]
```

**验证**:
```python
from carla_c2osr.core import create_planner
from carla_c2osr.environments import SimpleGridEnvironment

env = SimpleGridEnvironment()
planner = create_planner('c2osr', env=env)
# 应该创建成功
```

---

### 第6步: 集成测试 (第10-12天)

**任务**:
1. 创建 `tests/test_c2osr_planner.py`
2. 测试planner初始化
3. 测试轨迹规划
4. 测试动作选择
5. 测试更新逻辑
6. 测试与环境集成

**测试用例**:
```python
def test_c2osr_planner_initialization():
    """测试C2OSRPlanner初始化"""
    env = SimpleGridEnvironment()
    config = C2OSRConfig()
    planner = C2OSRPlanner(env, config)

    assert planner.grid is not None
    assert planner.bank is not None
    assert planner.buffer is not None
    assert planner.lattice_planner is not None
    assert planner.q_calculator is not None

def test_c2osr_trajectory_planning():
    """测试轨迹规划"""
    env = SimpleGridEnvironment()
    planner = C2OSRPlanner(env, C2OSRConfig())

    state, _ = env.reset()
    trajectory = planner.plan_trajectory(state, horizon=10)

    assert len(trajectory) > 0
    assert trajectory[0] is not None

def test_c2osr_action_selection():
    """测试动作选择"""
    env = SimpleGridEnvironment()
    planner = C2OSRPlanner(env, C2OSRConfig())

    state, _ = env.reset()
    action = planner.select_action(state)

    assert action is not None

def test_c2osr_update():
    """测试更新逻辑"""
    env = SimpleGridEnvironment()
    planner = C2OSRPlanner(env, C2OSRConfig())

    state, _ = env.reset()
    action = planner.select_action(state)
    result = env.step(action)

    transition = Transition(
        state=state,
        action=action,
        reward=result.reward,
        next_state=result.observation,
        terminated=result.terminated
    )

    metrics = planner.update(transition)
    assert metrics is not None

def test_c2osr_full_episode():
    """测试完整episode"""
    env = SimpleGridEnvironment(max_episode_steps=50)
    planner = C2OSRPlanner(env, C2OSRConfig())

    state, _ = env.reset()
    total_reward = 0

    for _ in range(50):
        action = planner.select_action(state)
        result = env.step(action)

        transition = Transition(
            state=state,
            action=action,
            reward=result.reward,
            next_state=result.observation,
            terminated=result.terminated
        )
        planner.update(transition)

        total_reward += result.reward
        state = result.observation

        if result.terminated or result.truncated:
            break

    assert total_reward != 0  # 应该有奖励
```

---

### 第7步: 性能验证 (第13-14天)

**任务**:
1. 创建性能基准测试
2. 对比包装前后性能
3. 确保性能无显著下降(<5%)
4. 记录性能指标

**基准测试**:
```python
def benchmark_c2osr_performance():
    """性能基准测试"""
    import time

    env = SimpleGridEnvironment()
    planner = C2OSRPlanner(env, C2OSRConfig())

    # 测试动作选择时间
    state, _ = env.reset()

    times = []
    for _ in range(100):
        start = time.time()
        action = planner.select_action(state)
        times.append(time.time() - start)

    avg_time = np.mean(times)
    print(f"Average action selection time: {avg_time*1000:.2f} ms")

    # 应该在合理范围内(<100ms)
    assert avg_time < 0.1
```

---

## 🚧 关键挑战

### 挑战1: 复杂的依赖关系

**问题**: C2OSR各组件之间有复杂的依赖关系

**解决方案**:
1. 使用依赖注入模式
2. 在`C2OSRPlanner`中统一管理所有组件
3. 提供清晰的初始化顺序

**代码示例**:
```python
class C2OSRPlanner:
    def __init__(self, env, config):
        # 按依赖顺序初始化
        self.grid = self._init_grid()           # 1. 最底层
        self.bank = self._init_bank()           # 2. 依赖grid
        self.buffer = self._init_buffer()       # 3. 依赖grid
        self.lattice = self._init_lattice()     # 4. 独立
        self.q_calc = self._init_q_calculator() # 5. 依赖所有
```

---

### 挑战2: 状态类型不一致

**问题**: 现有代码使用`WorldState`，但接口要求泛型

**解决方案**:
1. 使用类型变量明确指定
2. 提供类型转换函数
3. 保持接口灵活性

**代码示例**:
```python
class C2OSRPlanner(EpisodicPlanner[WorldState, Trajectory]):
    """明确指定类型参数"""
    pass
```

---

### 挑战3: Q值计算的复杂性

**问题**: Q值计算需要多个组件和历史数据

**解决方案**:
1. 封装所有依赖到`EvaluationContext`
2. 通过`context.custom`传递C2OSR特定数据
3. 保持接口通用性

**代码示例**:
```python
def evaluate(self, trajectory, context: EvaluationContext):
    # 从context获取C2OSR特定组件
    buffer = context.custom['buffer']
    grid = context.custom['grid']
    bank = context.custom['bank']

    # 调用Q值计算器
    q_value = self.q_calculator.compute_q_value(
        ...,
        trajectory_buffer=buffer,
        grid=grid,
        bank=bank
    )
```

---

### 挑战4: 性能优化

**问题**: 包装层可能引入额外开销

**解决方案**:
1. 最小化包装开销
2. 直接调用核心函数，避免多层封装
3. 使用性能分析工具识别瓶颈

**监控点**:
- 动作选择时间
- 轨迹评估时间
- 更新操作时间
- 内存使用

---

## 🧪 测试策略

### 单元测试

**范围**:
- 各组件初始化
- 配置参数设置
- 基础功能调用

**覆盖率目标**: 80%+

---

### 集成测试

**范围**:
- Planner与环境集成
- 完整episode运行
- 多episode训练

**测试场景**:
1. 简单直行场景
2. 避障场景
3. 曲线跟随场景
4. 长episode场景

---

### 性能测试

**指标**:
- 动作选择延迟
- 吞吐量(steps/s)
- 内存使用
- CPU使用率

**基准对比**:
- 与原始实现对比
- 性能下降不超过5%

---

### 兼容性测试

**验证**:
- 与SimpleGridEnvironment兼容
- 与现有ScenarioManager兼容
- 保持WorldState类型兼容

---

## ✅ 成功标准

### 功能完整性

- [x] C2OSRPlanner实现所有BasePlanner接口
- [x] C2OSREvaluator实现所有TrajectoryEvaluator接口
- [x] 支持完整的episode运行
- [x] 支持多episode训练
- [x] 正确的Dirichlet更新逻辑

---

### 性能指标

- [x] 动作选择时间 < 100ms
- [x] 性能下降 < 5%
- [x] 内存使用合理
- [x] 吞吐量 > 1000 steps/s

---

### 代码质量

- [x] 100%类型注解
- [x] 80%+测试覆盖率
- [x] 完整docstring
- [x] 通过所有测试
- [x] 代码风格一致

---

### 文档完整性

- [x] API文档完整
- [x] 使用示例清晰
- [x] 迁移指南准备
- [x] 性能报告生成

---

## 📅 时间表

| 周次 | 任务 | 交付物 | 状态 |
|------|------|--------|------|
| 第1周 | 步骤1-3: 基础结构+代码移动+Planner | 基础架构 | ⏳ 待开始 |
| 第2周 | 步骤4-5: Evaluator+工厂函数 | 完整接口 | ⏳ 待开始 |
| 第3周 | 步骤6-7: 测试+性能验证 | 测试通过 | ⏳ 待开始 |

---

## 🎯 里程碑

### 里程碑1: 基础结构完成 (第1周结束)
- ✅ 目录结构创建
- ✅ 核心代码移动
- ✅ C2OSRPlanner骨架实现

### 里程碑2: 接口实现完成 (第2周结束)
- ✅ C2OSRPlanner完整实现
- ✅ C2OSREvaluator完整实现
- ✅ 工厂函数和注册

### 里程碑3: 测试验证完成 (第3周结束)
- ✅ 所有测试通过
- ✅ 性能验证通过
- ✅ 文档完整

---

## 📋 检查清单

### 开发前检查
- [ ] 阅读并理解现有C2OSR代码
- [ ] 理解各组件之间的依赖关系
- [ ] 准备测试环境
- [ ] 创建git分支

### 开发中检查
- [ ] 遵循代码规范
- [ ] 编写单元测试
- [ ] 更新文档
- [ ] 提交代码时写清晰的commit message

### 开发后检查
- [ ] 所有测试通过
- [ ] 代码review完成
- [ ] 性能验证通过
- [ ] 文档更新完成
- [ ] 创建pull request

---

## 🔗 相关文档

- **阶段2完成报告**: `REFACTORING_PROGRESS.md`
- **核心接口文档**: `carla_c2osr/core/`
- **现有C2OSR代码**: `carla_c2osr/agents/c2osr/`
- **运行脚本**: `carla_c2osr/runner/run_sim_cl_simple.py`

---

**最后更新**: 2025-11-04
**负责人**: 待分配
**状态**: 📝 规划完成，等待开始实施
