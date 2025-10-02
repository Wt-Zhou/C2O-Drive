# CARLA对向碰撞风险场景使用指南

## 场景设置

### 场景描述
- **自车**：从起点向前行驶（朝向南，-90°）
- **逆行车**：从前方朝向自车驶来（朝向北，90°），带有随机横向偏移
- **风险**：两车对向行驶，存在碰撞风险

### 场景难度
- **Easy**: 逆行车横向偏移小（±1m），速度慢（4m/s）
- **Medium**: 逆行车横向偏移中等（±2m），速度中等（6m/s）
- **Hard**: 逆行车横向偏移大（±3m），速度快（8m/s）

---

## 快速开始

### 1. 启动CARLA服务器
```bash
cd /path/to/CARLA
./CarlaUE4.sh
```

### 2. 运行演示
```bash
# 方式1：简单演示（使用内置规划器）
python carla_c2osr/env/oncoming_scenario_demo.py

# 方式2：测试所有功能
python tests/test_carla_interface.py
```

---

## 与您的模型集成

### 完整示例代码

```python
from carla_c2osr.env.carla_scenario_1 import (
    CarlaSimulator,
    carla_transform_from_position,
    generate_oncoming_trajectory
)

# 1. 创建仿真器
sim = CarlaSimulator(town="Town03", dt=0.1)

# 2. 创建对向碰撞场景
ego_spawn = carla_transform_from_position(x=5.5, y=-90, yaw=-90)
oncoming_spawn = carla_transform_from_position(x=5.5, y=-145, yaw=90)

world_state = sim.create_scenario(ego_spawn, [oncoming_spawn])

# 3. 生成逆行车轨迹（随机偏移）
oncoming_trajectory = generate_oncoming_trajectory(
    start_x=5.5,
    start_y=-145,
    end_y=-100,
    horizon=50,
    lateral_offset_range=(-2.0, 2.0),
    seed=42
)

# 4. 调用您的规划模型生成自车轨迹
# ⭐ 这里是您的模型接口
ego_trajectory = your_planner.plan(
    current_state=world_state,
    horizon=50,
    dt=0.1
)

# 5. 同时执行自车和逆行车轨迹
states = sim.execute_multi_vehicle_trajectories(
    ego_trajectory=ego_trajectory,              # 您的模型输出
    agent_trajectories={0: oncoming_trajectory}, # 逆行车轨迹
    horizon=50,
    ego_velocity=5.0,
    agent_velocities={0: 6.0}
)

# 6. 评估结果
collision_occurred = sim.is_collision_occurred()
print(f"碰撞: {collision_occurred}")

# 计算最小距离
min_distance = float('inf')
for state in states:
    if len(state.agents) > 0:
        ego_pos = state.ego.position_m
        agent_pos = state.agents[0].position_m
        distance = ((ego_pos[0]-agent_pos[0])**2 + (ego_pos[1]-agent_pos[1])**2)**0.5
        min_distance = min(min_distance, distance)

print(f"最小距离: {min_distance:.2f}m")

# 7. 清理
sim.cleanup()
```

---

## 模型接口说明

### 输入：WorldState
```python
world_state = sim.get_world_state()

# WorldState包含：
# - world_state.ego: 自车状态
#   - position_m: (x, y) 位置
#   - velocity_mps: (vx, vy) 速度
#   - yaw_rad: 朝向（弧度）
#
# - world_state.agents: 环境车辆列表
#   - agent.position_m: (x, y)
#   - agent.velocity_mps: (vx, vy)
#   - agent.heading_rad: 朝向
#   - agent.agent_type: 车辆类型
```

### 输出：自车轨迹
```python
# 您的模型应输出：
ego_trajectory = [
    (x0, y0),  # t=0时刻的位置
    (x1, y1),  # t=1时刻的位置
    (x2, y2),  # t=2时刻的位置
    ...
    (xn, yn)   # t=n时刻的位置
]
```

### 执行轨迹
```python
# 只执行自车轨迹
states = sim.execute_trajectory(
    ego_trajectory=ego_trajectory,
    horizon=len(ego_trajectory),
    velocity=5.0,
    smooth=True  # 平滑控制（推荐）
)

# 同时执行自车和环境车轨迹
states = sim.execute_multi_vehicle_trajectories(
    ego_trajectory=ego_trajectory,
    agent_trajectories={
        0: oncoming_trajectory,  # 车辆索引0的轨迹
    },
    horizon=50,
    ego_velocity=5.0,
    agent_velocities={0: 6.0}
)
```

---

## 辅助函数

### 生成逆行车轨迹
```python
from carla_c2osr.env.carla_scenario_1 import generate_oncoming_trajectory

# 带随机横向偏移的逆行轨迹
trajectory = generate_oncoming_trajectory(
    start_x=5.5,           # 起始x坐标
    start_y=-145,          # 起始y坐标（远处）
    end_y=-100,            # 结束y坐标（接近）
    horizon=50,            # 轨迹点数
    lateral_offset_range=(-2.0, 2.0),  # 横向偏移范围
    seed=42                # 随机种子（可选）
)
```

### 生成直线轨迹
```python
from carla_c2osr.env.carla_scenario_1 import generate_straight_trajectory

# 直线轨迹
trajectory = generate_straight_trajectory(
    start_x=5.5,
    start_y=-90,
    direction_yaw=-90,  # 方向角度（度）
    distance=50,        # 总距离（米）
    horizon=50          # 轨迹点数
)
```

---

## 场景可视化

### 相机设置
```python
# 俯视图（默认）
sim.set_camera_view(height=60, pitch=-90)

# 斜视图
sim.set_camera_view(height=80, pitch=-45)

# 相机会自动跟随自车
```

### 调试输出
```python
# 打印每个时刻的状态
for t, state in enumerate(states):
    print(f"t={t}:")
    print(f"  自车: {state.ego.position_m}")
    print(f"  逆行车: {state.agents[0].position_m}")
    print(f"  距离: {calculate_distance(state):.2f}m")
```

---

## 多场景测试

### 批量测试不同难度
```python
difficulties = ["easy", "medium", "hard"]

for difficulty in difficulties:
    sim = CarlaSimulator(town="Town03", dt=0.1)

    # 创建场景
    ego_spawn, agent_spawns, world_state, params = create_oncoming_collision_scenario(
        sim, scenario_difficulty=difficulty
    )

    # 生成轨迹
    # ...

    # 执行
    states = sim.execute_multi_vehicle_trajectories(...)

    # 评估
    collision = sim.is_collision_occurred()
    print(f"{difficulty}: 碰撞={collision}")

    sim.cleanup()
```

### 批量测试不同随机种子
```python
for seed in range(10):
    # 生成不同的逆行车轨迹
    oncoming_trajectory = generate_oncoming_trajectory(
        ...,
        seed=seed
    )

    # 执行并评估
    ...
```

---

## 常见问题

### Q1: 车辆跳跃式移动？
**A**: 确保使用 `smooth=True`（默认）
```python
states = sim.execute_trajectory(..., smooth=True)
```

### Q2: 环境车方向不对？
**A**: 检查生成位置的yaw角度
```python
# 朝向北（逆行）
spawn = carla_transform_from_position(x=5.5, y=-145, yaw=90)
```

### Q3: 如何获取碰撞信息？
**A**: 使用碰撞检测方法
```python
collision_occurred = sim.is_collision_occurred()
```

### Q4: 如何计算最小距离？
**A**: 遍历所有状态
```python
min_distance = min(
    np.linalg.norm(
        np.array(state.ego.position_m) - np.array(state.agents[0].position_m)
    )
    for state in states
)
```

---

## 下一步

1. **集成您的规划模型**：替换 `generate_straight_trajectory()` 为您的模型
2. **批量测试**：运行多个场景，评估模型性能
3. **可视化分析**：记录轨迹并绘制图表
4. **性能优化**：调整 `dt` 和 `horizon` 参数

祝您使用愉快！🚀
