### 背景与目标

- **目标**: 在 CARLA 上做部署期学习 (Deployment-time learning)，对环境车一步转移分布 \(T\) 进行不确定性建模，并在规划时对最坏情形进行保守优化，保证“无试错”和“单调改进”。
- **生成模型**: 令 \(G \sim DP(\alpha, G_0)\)，\(z \sim G\)。给定 \(z\) 后，一步转移 \(T \sim Dirichlet(c \cdot z)\)，定义在当前状态的一步后继宏单元集合 \(S\) 上。
- **在线推断**: 每观测到 1 秒位移，计算 Dirichlet–Multinomial 边缘似然，对已存在“原子” \(z^{(k)}\) 进行责任分配（CRP/SVA 风格），软更新 \(z^{(k)}\)，必要时开启新原子。
- **规划**: 从 DP 后验预测采 \(z^{(s)}\) 并采 \(T^{(s)}\)，做多步卷积得各层占据概率 \(q^{(s)}\)；对自车 lattice 候选轨迹在样本上取最坏值（对 \(s\) 取 min）得到 \(Q\) 下界，选下界最大的轨迹执行。
- **关键性质**:
  - 无试错：采用保守并集界/机会约束规避高风险行为；
  - 单调改进：数据增多 → 后验收敛 → 最坏样本不再极端 → \(Q\) 下界不降。

### 模块职责

- `env/`
  - `types.py`: 所有核心数据结构与类型定义（见下表）。
  - `carla_client.py`/`sensors.py`/`wrappers.py`/`scenario_builder.py`: 与真实 CARLA 或 mock 环境交互的薄封装。
- `agents/`
  - `base.py`: 智能体抽象基类 (`reset`/`update`/`act`)。
  - `c2osr/`: C2OSR-Drive 核心，包括 `grid.py`（栅格/宏单元）、`dp_mixture.py`（DP 后验更新与采样）、`transition.py`（Dirichlet 采样一跳转移）、`risk.py`（风险合成/并集界）、`lattice.py`（候选轨迹生成）、`planner.py`（取样规划）、`agent.py`（对外智能体）。
  - `baselines/`: SAC、Offline-CQL、Shielded 等占位接口。
- `evaluation/`
  - `metrics.py`/`logger.py`/`plots.py`: 指标、日志与可视化占位实现。
- `utils/`
  - `geometry.py`/`seeding.py`/`timing.py`: 几何工具、随机数种子、计时辅助。
- `runner/`
  - `run_sim.py`: 运行一次仿真（支持 `--mock-env`）。
  - `train_baseline.py`/`replay_openloop.py`: 基线训练与开放环回放占位。

### 核心数据模型表

- `AgentState`:
  - `agent_id: str`
  - `position_m: tuple[float, float]`
  - `velocity_mps: tuple[float, float]`
- `EgoState`:
  - `position_m: tuple[float, float]`
  - `velocity_mps: tuple[float, float]`
  - `heading_rad: float`
- `WorldState`:
  - `time_s: float`
  - `ego: EgoState`
  - `agents: list[AgentState]`
- `Trajectory`:
  - `states: list[EgoState]`
  - `controls: list[EgoControl]`
- `EgoControl`:
  - `throttle: float`
  - `steer: float`
  - `brake: float`
- `OccupancyGrid`:
  - `width_cells: int`, `height_cells: int`, `cell_size_m: float`
  - `origin_m: tuple[float, float]`
  - `data: list[list[float]]`

### 一次控制循环

1. 观测当前 `WorldState` 与 1 秒位移；
2. 使用 Dirichlet–Multinomial 边缘似然与 CRP 责任，更新 DP 原子 \(z^{(k)}\)；必要时新建原子；
3. 从后验预测采样 \(z^{(s)}\)，并对每个样本采 \(T^{(s)} \sim Dirichlet(c z^{(s)})\)；
4. 多步传播得到各层占据概率 \(q^{(s)}\)；
5. 生成 lattice 候选轨迹（速度/曲率模板）；
6. 对每条轨迹，在样本维取最坏值（min over \(s\)），得到 \(Q\) 下界；
7. 选取下界最大的轨迹执行；
8. 记录日志与指标。

### 接口签名（摘要）

```python
# env/types.py
@dataclass
class AgentState: ...
@dataclass
class EgoState: ...
@dataclass
class EgoControl: ...
@dataclass
class WorldState: ...
@dataclass
class Trajectory: ...
@dataclass
class OccupancyGrid: ...

# agents/base.py
class BaseAgent(Protocol):
    def reset(self) -> None: ...
    def update(self, world: WorldState) -> None: ...
    def act(self, world: WorldState) -> EgoControl: ...

# agents/c2osr/dp_mixture.py
class DirichletProcessMixture:
    def update_with_observation(self, cell_index: int) -> None: ...
    def sample_z_vectors(self, num_samples: int) -> list[list[float]]: ...
    def get_atom_vectors(self) -> list[list[float]]: ...

# agents/c2osr/transition.py
def sample_dirichlet(prob_vector: list[float], concentration: float) -> list[float]: ...

# agents/c2osr/grid.py
@dataclass
class GridSpec: ...
class GridMapper:
    def world_to_cell(self, position_m: tuple[float, float]) -> int: ...

# agents/c2osr/risk.py
def union_risk(probabilities: list[float]) -> float: ...

# agents/c2osr/lattice.py
def generate_lattice_controls(ego: EgoState, horizon: int) -> list[Trajectory]: ...

# agents/c2osr/planner.py
class C2OSRPlanner:
    def plan(self, world: WorldState) -> tuple[Trajectory, dict]: ...

# agents/c2osr/agent.py
class C2OSRDriveAgent(BaseAgent):
    def act(self, world: WorldState) -> EgoControl: ...
```

### 数学细节（简要）

- 先验：\(G \sim DP(\alpha, G_0)\)。\(z \sim G\)。
- 似然：\(T \mid z \sim Dirichlet(c z)\)。对单个观测 \(o=i\) 的边缘似然 \(p(o=i \mid z) = z_i\)。
- 责任（CRP/SVA 风格）：\(r_k \propto n_k \cdot z^{(k)}_i\)，新原子 \(r_{\text{new}} \propto \alpha \cdot G_0(i)\)。
- 软更新：\(z^{(k)} \leftarrow (1-\eta r_k) z^{(k)} + (\eta r_k)\, \text{onehot}(i)\)，并归一化。
- 规划：对样本 \(s\) 取 \(q^{(s)}\)，在轨迹上进行风险合成（如并集界 \(1-\prod (1-p)\) 或机会约束 \(p\le\epsilon\)），最终对样本取最坏（min over \(s\)）得到 \(Q\) 下界。

以上实现细节在 `agents/c2osr/` 中以清晰类型与 docstring 提供占位；复杂函数处保留 TODO 以便后续替换为高保真实现。
