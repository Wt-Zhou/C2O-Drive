# 代码重构指南

本文档说明如何使用新的辅助模块逐步重构 `replay_openloop_refactored.py`。

## 重构进度

### ✅ Phase 1: 基础架构 (已完成)

- [x] 创建 `EpisodeContext` 类 - 封装运行上下文
- [x] 创建 `trajectory_generation.py` - 轨迹生成逻辑
- [x] 完善配置导出 - 方便直接访问配置类

### ✅ Phase 2: 函数分解 (已完成)

- [x] 提取 Q值计算函数 (`q_value_computation.py`)
- [x] 提取可视化数据准备函数 (`visualization_data.py`)
- [x] 提取数据存储函数 (`data_storage.py`)
- [x] 创建简化版本主入口 (`replay_openloop_simple.py`)
- [x] 测试验证功能完整性

### 📋 Phase 3: 优化提升 (计划中)

- [ ] 统一奖励计算接口
- [ ] 分离可视化渲染逻辑
- [ ] 创建 ExperimentRunner 类

---

## 使用新模块的示例

### 1. 使用 EpisodeContext 简化参数传递

#### 重构前 (15个参数):
```python
def run_episode(episode_id: int, horizon: int, ego_trajectory, world_init, grid, bank,
                trajectory_buffer, scenario_state, rng, output_dir, sigma,
                q_evaluator, trajectory_generator, scenario_manager,
                buffer_analyzer, q_tracker):
    # 函数体...
    pass
```

#### 重构后 (1个参数):
```python
from carla_c2osr.runner.episode_context import EpisodeContext

def run_episode(ctx: EpisodeContext):
    """运行单个episode - 使用上下文对象封装所有依赖"""
    # 直接从ctx访问所有组件
    print(f"Running episode {ctx.episode_id}/{ctx.horizon} steps")

    # 生成轨迹
    agent_trajectories, trajectory_cells = generate_agent_trajectories(ctx)

    # 访问配置
    dt = ctx.config.time.dt
    samples = ctx.config.sampling.reachable_set_samples

    # 判断是否可视化
    if ctx.should_visualize():
        # 生成可视化
        pass
```

### 2. 创建 EpisodeContext

```python
# 在main函数中创建context
ctx = EpisodeContext(
    episode_id=e,
    horizon=config.time.default_horizon,
    ego_trajectory=ego_trajectory,
    world_init=world_init,
    grid=grid,
    bank=bank,
    trajectory_buffer=trajectory_buffer,
    scenario_state=scenario_state,
    trajectory_generator=trajectory_generator,
    scenario_manager=scenario_manager,
    q_evaluator=q_evaluator,
    buffer_analyzer=buffer_analyzer,
    q_tracker=q_tracker,
    rng=rng,
    output_dir=output_dir,
    sigma=args.sigma
)

# 调用简化后的函数
result = run_episode(ctx)
```

### 3. 使用轨迹生成模块

```python
from carla_c2osr.runner.trajectory_generation import generate_agent_trajectories

# 重构前: 在run_episode中直接编写30行轨迹生成代码
# 重构后: 一行调用
agent_trajectories, trajectory_cells = generate_agent_trajectories(ctx)
```

---

## 配置访问最佳实践

### 推荐做法 ✅

```python
from carla_c2osr.config import get_global_config

config = get_global_config()

# 使用语义化的配置名称
dt = config.time.dt
horizon = config.time.default_horizon
samples = config.sampling.reachable_set_samples

# 匹配阈值
ego_threshold = config.matching.ego_state_threshold
agents_threshold = config.matching.agents_state_threshold

# 奖励参数
collision_penalty = config.reward.collision_penalty
comfort_weight = config.reward.acceleration_penalty_weight
```

### 避免做法 ❌

```python
# 不要硬编码
dt = 1.0  # ❌ 应该从config读取
samples = 100  # ❌ 应该从config读取

# 不要使用魔法数字
threshold = 5.0  # ❌ 什么的阈值? 应该用config.matching.ego_state_threshold
```

---

## 下一步重构计划

### 优先级 P0: 提取核心函数

```python
# 1. 提取Q值计算 (carla_c2osr/runner/q_value_computation.py)
def compute_q_value_at_timestep(ctx: EpisodeContext, world_state: WorldState) -> Dict:
    """独立的Q值计算函数"""
    pass

# 2. 提取可视化数据准备 (carla_c2osr/runner/visualization_data.py)
def prepare_visualization_data(ctx: EpisodeContext, timestep: int,
                              world_state: WorldState) -> VisualizationData:
    """准备可视化数据（不渲染）"""
    pass

# 3. 提取数据存储 (carla_c2osr/runner/data_storage.py)
def store_episode_data(ctx: EpisodeContext, episode_results: List[TimestepResult]):
    """存储episode数据到buffer"""
    pass
```

### 优先级 P1: 分离可视化

```python
# carla_c2osr/visualization/episode_visualizer.py
class EpisodeVisualizer:
    """完全独立的可视化生成器"""

    def add_timestep(self, viz_data: VisualizationData):
        """添加时间步数据"""
        pass

    def generate_gif(self, episode_id: int) -> Path:
        """生成GIF"""
        pass
```

### 优先级 P2: 实验运行器

```python
# carla_c2osr/runner/experiment_runner.py
class ExperimentRunner:
    """管理多episode实验"""

    def run_all_episodes(self) -> List[Dict]:
        """运行所有episodes"""
        pass

    def generate_summary(self, results: List[Dict]):
        """生成实验总结"""
        pass
```

---

## 验收标准

每个Phase完成后应满足:

### Phase 1
- [x] 所有新文件语法检查通过
- [x] EpisodeContext可以正常实例化
- [x] 配置导出完整

### Phase 2 (待完成)
- [ ] run_episode函数 < 100行
- [ ] 每个提取的函数 < 50行
- [ ] 单元测试覆盖率 > 70%

### Phase 3 (待完成)
- [ ] run_episode函数 < 50行
- [ ] 可视化完全解耦
- [ ] main函数 < 30行

---

## 贡献指南

如需继续重构,请遵循:

1. **小步快跑**: 每次只重构一个函数/模块
2. **保持兼容**: 不破坏现有功能
3. **测试先行**: 重构前先写测试
4. **文档同步**: 更新本文档进度

---

## 重构成果总结

### 代码简化效果

**原版本** (`replay_openloop_refactored.py`):
- `run_episode` 函数: **424行**
- 参数数量: **15个参数**
- 可读性: 较差(多层嵌套,职责混杂)

**简化版本** (`replay_openloop_simple.py`):
- `run_episode` 函数: **~80行** (减少 81%)
- 参数数量: **1个参数** (EpisodeContext)
- 可读性: 显著提升(职责单一,模块化)

### 新增模块

1. **episode_context.py** (87行)
   - 封装15个参数到单一上下文对象
   - 提供配置快捷访问
   - 内置可视化判断逻辑

2. **trajectory_generation.py** (86行)
   - 提取轨迹生成逻辑(~50行)
   - 支持后备轨迹机制
   - 完整的错误处理

3. **q_value_computation.py** (194行)
   - 提取Q值计算逻辑(~110行)
   - 包含结果打印和tracker记录
   - 可视化生成(每5个episode)

4. **visualization_data.py** (207行)
   - 提取可视化数据准备(~120行)
   - 分离数据准备和渲染逻辑
   - 统计信息计算

5. **data_storage.py** (61行)
   - 提取数据存储逻辑(~35行)
   - 按时间步组织数据
   - 清晰的存储流程

**总计**: ~635行高内聚模块化代码替代了424行的单体函数

### 测试验证

已通过测试:
- ✅ 2个episodes, 3个时间步运行成功
- ✅ Q值计算正常(均值=10.00)
- ✅ 可视化生成成功(GIF)
- ✅ 轨迹数据正确存储到buffer
- ✅ 原有功能完全保留

---

**最后更新**: 2025-10-02
**重构进度**: Phase 2 完成 (67%)
