# C2O-Drive 算法架构分析

## 目录
- [整体架构概览](#整体架构概览)
- [核心接口设计](#核心接口设计)
- [算法分类](#算法分类)
- [运行流程对比](#运行流程对比)
- [文件组织结构](#文件组织结构)

---

## 整体架构概览

你的代码库采用了**两种设计模式**来组织算法：

### 1. **Planner模式** (标准化接口)
- 继承自 `EpisodicAlgorithmPlanner[WorldState, EgoControl]`
- 遵循统一的规划器接口
- 用于：**C2OSR, PPO, Rainbow DQN, RCRL**

### 2. **Agent模式** (传统RL接口)
- 独立的Agent类（不继承Planner）
- 传统强化学习接口
- 用于：**SAC, DQN**

---

## 核心接口设计

### BasePlanner 接口 (`src/c2o_drive/core/planner.py`)

所有算法都需要实现以下方法：

```python
class BasePlanner(ABC, Generic[ObsType, ActType]):
    @abstractmethod
    def select_action(self, observation: ObsType,
                      deterministic: bool = False,
                      **kwargs) -> ActType:
        """选择动作"""
        pass

    @abstractmethod
    def update(self, transition: Transition[ObsType, ActType]) -> UpdateMetrics:
        """更新学习（从transition中学习）"""
        pass

    @abstractmethod
    def reset(self) -> None:
        """重置内部状态（每个episode开始时调用）"""
        pass

    def save_checkpoint(self, path: str | Path) -> None:
        """保存检查点"""
        pass

    def load_checkpoint(self, path: str | Path) -> None:
        """加载检查点"""
        pass
```

### EpisodicPlanner 扩展接口

轨迹级别规划器需要额外实现：

```python
class EpisodicPlanner(BasePlanner[ObsType, ActType]):
    @abstractmethod
    def plan_trajectory(self, observation: ObsType,
                       horizon: int,
                       **kwargs) -> List[ActType]:
        """规划一条完整轨迹"""
        pass
```

### EpisodicAlgorithmPlanner

融合了 `BaseAlgorithmPlanner` 和 `EpisodicPlanner`：

```python
class EpisodicAlgorithmPlanner(
    BaseAlgorithmPlanner[ObsType, ActType],
    EpisodicPlanner[ObsType, ActType],
    Generic[ObsType, ActType]
):
    """用于轨迹级别规划的基类"""
    pass
```

---

## 算法分类

### 🎯 Planner模式算法

| 算法 | 类名 | 继承关系 | 动作空间 | 特点 |
|------|------|----------|----------|------|
| **C2OSR** | `C2OSRPlanner` | `EpisodicAlgorithmPlanner` | 离散（lattice） | 贝叶斯风险感知 |
| **PPO** | `PPOPlanner` | `EpisodicAlgorithmPlanner` | 离散（lattice） | On-policy，Actor-Critic |
| **Rainbow DQN** | `RainbowDQNPlanner` | `EpisodicAlgorithmPlanner` | 离散（lattice） | DQN改进版 |
| **RCRL** | `RCRLPlanner` | `EpisodicAlgorithmPlanner` | 离散（lattice） | 带约束的RL |

#### 共同特点：
1. ✅ 统一接口：`select_action()`, `update()`, `reset()`
2. ✅ 轨迹级别执行：一次生成完整轨迹，然后逐步执行
3. ✅ 使用 `LatticePlanner` 生成候选轨迹
4. ✅ 输入：`WorldState`，输出：`EgoControl`
5. ✅ 动态动作空间：`action_dim = len(lateral_offsets) × len(speed_variations)`

### 🤖 Agent模式算法

| 算法 | 类名 | 继承关系 | 动作空间 | 特点 |
|------|------|----------|----------|------|
| **SAC** | `SACAgent` | 无（独立类） | 连续（需要rescale） | Off-policy，Actor-Critic |
| **DQN** | `DQNAgent` | 无（独立类） | 离散 | Q-learning |

#### 共同特点：
1. ❌ 不继承 `BasePlanner`
2. ⚙️ 使用传统RL接口：`select_action(state_features)`
3. 🔄 需要**手动**与 `LatticePlanner` 集成
4. 📦 有自己的 `ReplayBuffer` 实现

---

## 运行流程对比

### 🎯 Planner模式算法运行流程（以PPO为例）

```python
# 1. 创建Planner
from c2o_drive.algorithms.ppo import PPOPlanner, PPOConfig

config = PPOConfig(lattice=lattice_config, ...)
planner = PPOPlanner(config)

# 2. Episode循环
for episode in range(num_episodes):
    state, info = env.reset()
    planner.reset()  # 重置planner状态

    reference_path = info.get('reference_path', [])

    # 3. Step循环
    while not done:
        # 选择动作（planner内部生成轨迹并选择waypoint）
        control = planner.select_action(
            state,
            deterministic=False,
            reference_path=reference_path
        )

        # 执行动作
        step_result = env.step(control)

        # 创建Transition
        transition = Transition(
            state=state,
            action=control,
            reward=step_result.reward,
            next_state=step_result.observation,
            terminated=step_result.terminated,
            truncated=step_result.truncated,
            info=step_result.info,
        )

        # 更新planner（内部处理buffer、计算loss、更新网络）
        metrics = planner.update(transition)

        state = step_result.observation
```

#### PPO内部流程：

```
select_action() 第一次调用时：
  ├─ 网络输出action概率分布
  ├─ 采样离散action_idx
  ├─ 使用LatticePlanner生成所有候选轨迹
  ├─ 根据action_idx选择一条轨迹
  └─ 返回该轨迹的第一个waypoint对应的控制

select_action() 后续调用：
  └─ 直接返回当前轨迹的下一个waypoint控制

update()：
  ├─ 存储(state, action, reward, value, log_prob)到buffer
  ├─ 当轨迹结束时：
  │   ├─ 计算GAE advantages
  │   ├─ 执行PPO更新（多个epochs）
  │   └─ 清空buffer
  └─ 返回UpdateMetrics
```

### 🤖 Agent模式算法运行流程（以SAC为例）

```python
# 1. 创建Agent
from c2o_drive.algorithms.sac import SACAgent, SACConfig

config = SACConfig(...)
agent = SACAgent(config)

# 2. 创建LatticePlanner（需要手动创建！）
from c2o_drive.utils.lattice_planner import LatticePlanner

lattice_planner = LatticePlanner(
    lateral_offsets=[-3.0, 0.0, 3.0],
    speed_variations=[4.0, 6.0, 8.0],
    num_trajectories=10,
)

# 3. Episode循环
for episode in range(num_episodes):
    state, info = env.reset()
    reference_path = info.get('reference_path', [])

    # 提取状态特征
    state_features = extract_state_features(state)

    # Agent输出连续动作 [-1, 1]
    action = agent.select_action(state_features, training=True)

    # Rescale到lattice参数范围
    lateral_offset = rescale(action[0], range=[-3.0, 3.0])
    target_speed = rescale(action[1], range=[4.0, 8.0])

    # 使用LatticePlanner生成轨迹（手动调用）
    trajectory = lattice_planner.generate_single_trajectory(
        reference_path, lateral_offset, target_speed
    )

    # 4. Step循环（执行轨迹的每一步）
    for waypoint in trajectory.waypoints:
        control = waypoint_to_control(state, waypoint)
        step_result = env.step(control)

        # 手动存储到replay buffer
        agent.replay_buffer.push(
            state_features, action,
            step_result.reward,
            next_state_features,
            done
        )

        # 手动更新agent
        if agent.replay_buffer.size() >= batch_size:
            loss = agent.update()

        state = step_result.observation
```

#### SAC与Planner模式的关键区别：

| 方面 | Planner模式 (PPO) | Agent模式 (SAC) |
|------|------------------|-----------------|
| **轨迹生成** | ✅ 内部自动处理 | ❌ 需要手动调用 `LatticePlanner` |
| **Buffer管理** | ✅ 内部自动管理 | ❌ 需要手动push/sample |
| **更新时机** | ✅ 自动判断（trajectory结束） | ❌ 需要手动判断buffer大小 |
| **状态特征** | ✅ 内部提取 | ❌ 需要手动提取 |
| **接口统一性** | ✅ 统一 `Transition` | ❌ 手动构造数据 |

### 🎯 C2OSR的特殊流程

C2OSR虽然也是Planner模式，但它的执行流程略有不同：

```python
# C2OSRPlanner内部流程
select_action():
  ├─ 第一次调用时：
  │   ├─ 使用LatticePlanner生成所有候选轨迹
  │   ├─ 对每条轨迹计算Q值（使用Dirichlet先验）
  │   ├─ 选择Q值最高的轨迹
  │   └─ 存储选中轨迹
  └─ 后续调用：返回当前轨迹的下一个waypoint

update():
  ├─ 收集轨迹执行数据
  ├─ 更新Dirichlet后验（贝叶斯更新）
  └─ 更新Trajectory Buffer
```

---

## 文件组织结构

### Planner模式算法结构（以PPO为例）

```
src/c2o_drive/algorithms/ppo/
├── __init__.py              # 导出接口
├── config.py                # PPOConfig配置类
├── network.py               # ActorCriticNetwork网络
├── rollout_buffer.py        # PPO专用buffer
└── planner.py              # PPOPlanner主类（继承EpisodicAlgorithmPlanner）

examples/
└── run_ppo_carla.py        # 训练脚本
```

### Agent模式算法结构（以SAC为例）

```
src/c2o_drive/algorithms/sac/
├── __init__.py              # 导出接口
├── config.py                # SACConfig配置类
├── network.py               # Actor和Critic网络
├── replay_buffer.py         # 经验回放buffer
└── agent.py                # SACAgent主类（独立类，不继承Planner）

examples/
└── run_sac_carla.py        # 训练脚本（需要手动集成LatticePlanner）
```

### C2OSR算法结构（最复杂）

```
src/c2o_drive/algorithms/c2osr/
├── __init__.py              # 导出接口
├── config.py                # 多个Config类（C2OSRPlannerConfig, LatticePlannerConfig等）
├── planner.py              # C2OSRPlanner主类
├── factory.py              # 创建planner的工厂函数
├── dirichlet.py            # Dirichlet后验更新
├── q_value.py              # Q值计算器
├── trajectory_buffer.py    # 轨迹buffer
├── grid_mapper.py          # 网格映射
├── rewards.py              # 奖励函数
└── ... (其他组件)

examples/
└── run_c2osr_carla.py      # 训练脚本
```

---

## 训练脚本的统一模式

所有训练脚本都遵循类似的结构：

```python
# 1. 导入算法和环境
from c2o_drive.algorithms.xxx import XXXPlanner/XXXAgent, XXXConfig
from c2o_drive.environments.carla_env import CarlaEnvironment

# 2. 创建配置
config = XXXConfig(...)

# 3. 创建环境
env = CarlaEnvironment(...)

# 4. 创建Planner/Agent
planner = XXXPlanner(config)  # 或 agent = XXXAgent(config)

# 5. 创建Trainer（封装训练循环）
trainer = XXXTrainer(planner, env, ...)

# 6. 运行训练
trainer.train(num_episodes=1000, max_steps=100)
```

---

## 如何添加新算法？

### 方法1：Planner模式（推荐）

```python
from c2o_drive.algorithms.base import EpisodicAlgorithmPlanner
from c2o_drive.core.types import WorldState, EgoControl

class MyPlanner(EpisodicAlgorithmPlanner[WorldState, EgoControl]):
    def __init__(self, config):
        super().__init__(config)
        # 初始化网络、buffer等
        self.lattice_planner = LatticePlanner(...)

    def select_action(self, observation, deterministic=False, **kwargs):
        # 1. 提取特征
        # 2. 网络输出动作
        # 3. 生成轨迹
        # 4. 返回waypoint控制
        pass

    def update(self, transition):
        # 1. 存储数据
        # 2. 判断是否更新
        # 3. 计算loss
        # 4. 更新网络
        # 5. 返回metrics
        pass

    def reset(self):
        # 重置内部状态
        pass

    def plan_trajectory(self, observation, horizon, **kwargs):
        # 可选：生成完整轨迹
        pass
```

### 方法2：Agent模式（传统RL）

```python
class MyAgent:
    def __init__(self, config):
        self.network = ...
        self.replay_buffer = ...

    def select_action(self, state_features, training=True):
        # 返回动作
        pass

    def update(self):
        # 从buffer采样并更新
        pass
```

然后需要在训练脚本中手动集成LatticePlanner。

---

## 总结

### 🎯 Planner模式的优势

1. ✅ **接口统一**：所有算法都是 `BasePlanner` 子类
2. ✅ **轨迹自动管理**：内部处理轨迹生成和执行
3. ✅ **状态转换封装**：统一使用 `Transition`
4. ✅ **易于替换**：可以无缝切换不同算法
5. ✅ **训练脚本简洁**：主循环代码高度一致

### 🤖 Agent模式的特点

1. 💡 **灵活性高**：不受Planner接口约束
2. ⚙️ **手动控制**：需要手动管理轨迹、buffer、更新
3. 📦 **传统RL风格**：符合经典RL代码习惯
4. 🔧 **集成成本高**：需要更多胶水代码

### 💡 建议

- **新算法优先使用Planner模式**（如PPO、Rainbow DQN、RCRL）
- SAC和DQN使用Agent模式可能是历史遗留，可以考虑重构为Planner模式
- 统一到Planner模式可以简化代码维护和算法对比

---

## 快速参考

### 当前算法清单

| 算法 | 模式 | 文件路径 | 训练脚本 | 状态 |
|------|------|----------|----------|------|
| C2OSR | Planner | `algorithms/c2osr/planner.py` | `run_c2osr_carla.py` | ✅ 完整 |
| PPO | Planner | `algorithms/ppo/planner.py` | `run_ppo_carla.py` | ✅ 完整 |
| Rainbow DQN | Planner | `algorithms/rainbow_dqn/planner.py` | - | ⚠️ 需要训练脚本 |
| RCRL | Planner | `algorithms/rcrl/planner.py` | `test_rcrl.py` | ⚠️ 需要完整训练脚本 |
| SAC | Agent | `algorithms/sac/agent.py` | `run_sac_carla.py` | ✅ 完整 |
| DQN | Agent | `algorithms/dqn/agent.py` | - | ⚠️ 需要训练脚本 |

### 运行示例

```bash
# C2OSR
python examples/run_c2osr_carla.py --scenario s4_wrong_way --episodes 100

# PPO
python examples/run_ppo_carla.py --scenario s4_wrong_way --episodes 1000

# SAC
python examples/run_sac_carla.py --scenario s4_wrong_way --episodes 1000

# RCRL（测试脚本）
python examples/test_rcrl.py
```
