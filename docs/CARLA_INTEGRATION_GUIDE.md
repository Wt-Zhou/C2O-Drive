# C2OSR + CARLA 集成指南

## 概述

本指南说明如何将C2OSR算法与CARLA仿真环境集成，实现真实的3D自动驾驶场景测试。

## 架构改进总结

### 1. CarlaEnvironment 增强

**文件位置**: `carla_c2osr/environments/carla_env.py`

**改进内容**:
- ✅ **碰撞检测优化**: 使用CARLA碰撞传感器数据，辅以几何距离检测
- ✅ **轨迹记录功能**: 添加`get_episode_trajectory()`方法，记录完整episode数据
- ✅ **增强info字典**: 包含碰撞状态、加速度、急动度等详细信息
- ✅ **可视化功能**: 添加`visualize_trajectory()`方法，生成matplotlib轨迹图

**新增方法**:
```python
# 获取episode轨迹
trajectory = env.get_episode_trajectory()

# 可视化轨迹
env.visualize_trajectory(save_path="trajectory.png")
```

### 2. 全局配置扩展

**文件位置**: `carla_c2osr/config/global_config.py`

**新增配置类**: `CarlaConfig`
```python
@dataclass
class CarlaConfig:
    # 连接配置
    host: str = "localhost"
    port: int = 2000

    # 地图和天气
    town: str = "Town03"
    weather: str = "ClearNoon"

    # 仿真配置
    dt: float = 0.1
    no_rendering: bool = False

    # 场景配置
    num_vehicles: int = 10
    num_pedestrians: int = 5
    autopilot: bool = False

    # 相机视角
    camera_height: float = 60.0
    camera_pitch: float = -90.0

    # Episode配置
    max_episode_steps: int = 500
```

**使用方式**:
```python
from carla_c2osr.config.global_config import GlobalConfig

config = GlobalConfig()
config.carla.town = "Town04"
config.carla.num_vehicles = 20
```

### 3. 场景库

**文件位置**: `carla_c2osr/env/carla_scenarios.py`

**预定义场景**:
- `oncoming_easy` - 对向碰撞（简单）
- `oncoming_medium` - 对向碰撞（中等）
- `oncoming_hard` - 对向碰撞（困难）
- `lane_change_left` - 左变道
- `lane_change_right` - 右变道
- `overtake` - 超车场景
- `intersection` - 路口场景
- `multi_agent` - 多车交互
- `highway` - 高速公路

**使用示例**:
```python
from carla_c2osr.env.carla_scenarios import get_scenario, list_scenarios

# 查看所有场景
scenarios = list_scenarios()

# 获取特定场景
scenario = get_scenario("oncoming_medium")
print(scenario.description)  # 对向车距离适中，需要及时避让
print(scenario.difficulty)    # medium
```

### 4. 主运行脚本

**文件位置**: `examples/run_c2osr_carla.py`

**功能特性**:
- 完整的C2OSR + CARLA集成
- 支持预定义场景库
- 命令行参数配置
- 轨迹数据保存
- 实时性能监控
- 自动统计和报告

## 快速开始

### 前提条件

1. **安装CARLA仿真器**
   ```bash
   # 下载CARLA (例如0.9.13版本)
   wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.13.tar.gz
   tar -xzf CARLA_0.9.13.tar.gz
   cd CARLA_0.9.13
   ```

2. **启动CARLA服务器**
   ```bash
   # 终端1: 启动CARLA
   cd /path/to/CARLA
   ./CarlaUE4.sh

   # 或者无渲染模式（更快）
   ./CarlaUE4.sh -RenderOffScreen
   ```

3. **配置CARLA Python路径**

   确保CARLA Python包可访问（已在carla_scenario_1.py中自动处理）

### 基本使用

1. **列出所有可用场景**
   ```bash
   cd /path/to/C2O-Drive
   python examples/run_c2osr_carla.py --list-scenarios
   ```

2. **运行默认配置**
   ```bash
   python examples/run_c2osr_carla.py
   ```

3. **运行特定场景**
   ```bash
   python examples/run_c2osr_carla.py --scenario oncoming_medium --episodes 5
   ```

4. **自定义配置**
   ```bash
   python examples/run_c2osr_carla.py \
     --town Town04 \
     --num-vehicles 20 \
     --num-pedestrians 10 \
     --horizon 15 \
     --dt 0.3 \
     --episodes 10 \
     --output-dir outputs/my_experiment
   ```

5. **高性能模式（无渲染）**
   ```bash
   # 先启动CARLA无渲染模式
   ./CarlaUE4.sh -RenderOffScreen

   # 运行实验
   python examples/run_c2osr_carla.py \
     --no-rendering \
     --config-preset fast \
     --episodes 20
   ```

### 命令行参数

#### 基本参数
- `--episodes N` - 运行N个episodes（默认5）
- `--max-steps N` - 每个episode最大步数（默认500）
- `--seed N` - 随机种子（默认2025）

#### CARLA配置
- `--host HOST` - CARLA服务器地址（默认localhost）
- `--port PORT` - CARLA端口（默认2000）
- `--town TOWN` - 地图名称（Town01-Town10，默认Town03）
- `--scenario NAME` - 场景名称（见场景库）
- `--num-vehicles N` - 环境车辆数（默认10）
- `--num-pedestrians N` - 行人数（默认5）
- `--no-rendering` - 禁用渲染

#### C2OSR参数
- `--config-preset PRESET` - 配置预设：default/fast/high-precision
- `--horizon N` - 规划时域（默认10）
- `--dt SECONDS` - 时间步长（默认0.5s）
- `--grid-size M` - 网格大小（默认50m）

#### 输出参数
- `--output-dir DIR` - 输出目录
- `--save-trajectory` - 保存轨迹数据
- `--quiet` - 静默模式

## 编程API使用

### 1. 基本使用模式

```python
from carla_c2osr.environments import CarlaEnvironment
from carla_c2osr.algorithms.c2osr import create_c2osr_planner, C2OSRPlannerConfig
from carla_c2osr.env.carla_scenarios import get_scenario

# 创建环境
env = CarlaEnvironment(
    host='localhost',
    port=2000,
    town='Town03',
    dt=0.5,
    max_episode_steps=500,
    num_vehicles=10,
)

# 创建规划器
config = C2OSRPlannerConfig(horizon=10)
planner = create_c2osr_planner(config)

# 运行episode
state, info = env.reset(seed=42)
planner.reset()

for step in range(500):
    # 选择动作
    action = planner.select_action(state)

    # 执行
    result = env.step(action)

    # 更新
    planner.update(Transition(...))

    state = result.observation

    if result.terminated or result.truncated:
        break

# 可视化
env.visualize_trajectory(save_path="trajectory.png")

# 获取轨迹数据
trajectory = env.get_episode_trajectory()

# 清理
env.close()
```

### 2. 使用预定义场景

```python
from carla_c2osr.env.carla_scenarios import get_scenario

# 获取场景
scenario = get_scenario("oncoming_hard")

# 重置环境时应用场景
state, info = env.reset(
    seed=42,
    options={'scenario_config': {'scenario': scenario}}
)
```

### 3. 访问详细信息

```python
# Step返回的info字典包含
result = env.step(action)
print(result.info)
# {
#     'collision': False,
#     'collision_sensor': False,
#     'step': 10,
#     'episode_reward': 45.3,
#     'acceleration': 2.1,
#     'jerk': 0.5,
# }

# Episode轨迹记录包含
trajectory = env.get_episode_trajectory()
for record in trajectory:
    print(record['step'])          # 步数
    print(record['state'])         # WorldState
    print(record['action'])        # EgoControl
    print(record['reward'])        # 奖励
    print(record['acceleration'])  # 加速度
    print(record['jerk'])          # 急动度
```

## 与虚拟环境的对比

| 特性 | SimpleGridEnvironment | ScenarioReplayEnvironment | **CarlaEnvironment** |
|------|----------------------|---------------------------|---------------------|
| 仿真类型 | 2D简化网格 | 2D场景回放 | **3D真实仿真** |
| 物理引擎 | 简化运动学 | 恒速模型 | **CARLA物理引擎** |
| 可视化 | Matplotlib | Matplotlib | **CARLA 3D + Matplotlib** |
| 碰撞检测 | 距离检测 | 距离检测 | **传感器 + 几何检测** |
| 传感器支持 | 无 | 无 | **碰撞（可扩展更多）** |
| 性能 | ⚡ 非常快 | ⚡ 快 | 🐌 **较慢（真实仿真）** |
| 适用场景 | 算法原型测试 | 批量实验 | **最终验证和演示** |

**使用建议**:
- **算法开发**: 使用SimpleGridEnvironment（快速迭代）
- **批量实验**: 使用ScenarioReplayEnvironment（快速收集数据）
- **最终验证**: 使用CarlaEnvironment（真实场景测试）

## 性能优化建议

### 1. 提升仿真速度

```bash
# 无渲染模式
./CarlaUE4.sh -RenderOffScreen

# 使用fast配置预设
python run_c2osr_carla.py --config-preset fast --no-rendering
```

### 2. 减少计算开销

```python
# 使用较少的轨迹候选
config = C2OSRPlannerConfig(
    lattice=LatticePlannerConfig(
        lateral_offsets=[-2.0, 0.0, 2.0],  # 减少到3个
        speed_variations=[4.0],            # 只用1个速度
    ),
    q_value=QValueConfig(
        n_samples=20,  # 减少采样数
    )
)
```

### 3. 批处理实验

```bash
# 使用脚本批量运行
for scenario in oncoming_easy oncoming_medium oncoming_hard; do
    python run_c2osr_carla.py \
        --scenario $scenario \
        --episodes 10 \
        --output-dir outputs/$scenario
done
```

## 故障排查

### 问题1: 无法连接CARLA

**错误**: `✗ 连接CARLA失败: Connection refused`

**解决**:
1. 确保CARLA服务器已启动
   ```bash
   ps aux | grep CarlaUE4
   ```
2. 检查端口是否正确
   ```bash
   netstat -an | grep 2000
   ```
3. 检查防火墙设置

### 问题2: CARLA导入失败

**错误**: `ModuleNotFoundError: No module named 'carla'`

**解决**:
1. 检查CARLA .egg文件路径
2. 手动添加到Python路径
   ```python
   import sys
   sys.path.append('/path/to/carla/PythonAPI/carla/dist/carla-0.9.13-py3.7-linux-x86_64.egg')
   ```

### 问题3: 性能很慢

**解决**:
1. 使用无渲染模式
2. 减少车辆和行人数量
3. 使用`--config-preset fast`
4. 增大时间步长`--dt 1.0`

### 问题4: 碰撞检测不准确

**解决**:
- 现在使用CARLA碰撞传感器，应该非常准确
- 如果仍有问题，检查`collision_threshold`参数
- 查看info字典中的`collision_sensor`字段

## 下一步扩展

### 建议的改进方向

1. **添加更多传感器**
   - RGB相机
   - 深度相机
   - 激光雷达
   - IMU

2. **扩展场景库**
   - 更复杂的路口场景
   - 高速公路合并
   - 停车场景
   - 恶劣天气条件

3. **性能优化**
   - 异步仿真模式
   - 批量轨迹执行
   - GPU加速

4. **数据收集**
   - 自动化批量实验
   - 数据集生成
   - 模型训练支持

## 文件清单

### 修改的文件
- ✅ `carla_c2osr/environments/carla_env.py` - 增强CarlaEnvironment
- ✅ `carla_c2osr/config/global_config.py` - 添加CarlaConfig

### 新增的文件
- ✅ `carla_c2osr/env/carla_scenarios.py` - 场景库
- ✅ `examples/run_c2osr_carla.py` - 主运行脚本
- ✅ `docs/CARLA_INTEGRATION_GUIDE.md` - 本文档

### 未修改的文件（保证兼容性）
- ✅ 所有核心算法文件（`carla_c2osr/algorithms/`）
- ✅ SimpleGridEnvironment
- ✅ ScenarioReplayEnvironment
- ✅ 所有C2OSR核心组件

## 总结

现在你可以：
1. ✅ 在CARLA中运行C2OSR算法
2. ✅ 使用预定义场景测试不同情况
3. ✅ 保存和可视化轨迹数据
4. ✅ 获取详细的性能和碰撞信息
5. ✅ 保持与现有虚拟环境的兼容性

**核心优势**：
- 无需修改核心算法代码
- 标准Gym接口，易于扩展
- 丰富的场景库
- 详细的监控和分析工具

祝实验顺利！🚗💨
