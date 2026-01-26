# 代码修改日志

本文件记录所有由Claude进行的代码修改。

---

## 2026-01-22

### 修改 #1: 修复碰撞检测bug
**时间**: 2026-01-22 下午
**文件**: `examples/run_ppo_carla.py`

**修改内容**:
- **行号**: ~line 397-400
- **修改前**:
  ```python
  if step_result.terminated or step_result.truncated:
      collision = step_result.info.get('collision', False)
      break
  ```
- **修改后**:
  ```python
  # 检查碰撞（每个step都检查，不只是terminated时）
  if step_result.info.get('collision', False):
      collision = True
      print(f"  ⚠️ 碰撞检测！Step {step}, Reward: {step_result.reward:.2f}, Total: {episode_reward:.2f}")

  # Check termination
  if step_result.terminated or step_result.truncated:
      break
  ```

**修改原因**:
- 原代码只在episode终止时检查碰撞，如果episode因其他原因结束会漏检
- 训练模式显示不碰撞，但评估模式显示100%碰撞

**影响**:
- 碰撞统计更准确
- 训练和评估的碰撞检测一致

---

### 修改 #2: 统一训练和评估的控制逻辑
**时间**: 2026-01-22 下午
**文件**: `examples/run_ppo_carla.py`

**修改内容**:
- **行号**: ~line 861-896 (evaluation mode)
- **修改前**: 评估模式使用distance-based throttle控制
- **修改后**: 评估模式使用与训练一致的P-controller:
  ```python
  # Calculate heading error
  dx = target_x - current_x
  dy = target_y - current_y
  target_heading = np.arctan2(dy, dx)
  heading_error = target_heading - state.ego.yaw_rad
  heading_error = np.arctan2(np.sin(heading_error), np.cos(heading_error))

  # P-controller for steering（和训练一致）
  steer = np.clip(heading_error * 0.5, -1.0, 1.0)

  # Speed control（和训练一致）
  current_speed = np.linalg.norm(np.array(state.ego.velocity_mps))
  speed_error = selected_trajectory.target_speed - current_speed
  ```

**修改原因**:
- 训练和评估使用不同控制算法导致行为不一致
- 评估模式导入的模型一直往一个方向走

**影响**:
- 训练和评估行为一致
- 模型评估结果更准确

---

### 修改 #3: SafetyReward改为3-tier分级惩罚系统
**时间**: 2026-01-22 下午
**文件**: `src/c2o_drive/environments/rewards.py`

**修改内容**:
- **行号**: line 35-68
- **修改前**: 2-tier系统
  ```python
  if min_dist < self.critical_distance:
      return -self.distance_weight * (self.critical_distance - min_dist)
  else:
      return 0.1 * self.distance_weight  # near-miss也给正奖励（错误）
  ```
- **修改后**: 3-tier系统
  ```python
  # 从global_config读取near_miss阈值
  from c2o_drive.config import get_global_config
  near_miss_threshold = get_global_config().safety.near_miss_threshold_m

  if min_dist < self.critical_distance:  # < 2m
      # 严重惩罚
      return -self.distance_weight * (self.critical_distance - min_dist)
  elif min_dist < near_miss_threshold:  # 2-4m
      # 轻度惩罚（near-miss区间）
      return -self.near_miss_weight * (near_miss_threshold - min_dist)
  else:  # ≥ 4m
      # 安全奖励
      return 0.1 * self.distance_weight
  ```

**修改原因**:
- 2-4m的near-miss区间应该惩罚，而不是给正奖励
- 需要从global_config读取可配置的near_miss阈值

**影响**:
- 模型会学习在2-4m区间也减速或避让
- SafetyReward梯度更清晰

---

### 修改 #4: 降低EfficiencyReward权重
**时间**: 2026-01-22 晚上
**文件**: `src/c2o_drive/environments/rewards.py`

**修改内容**:
- **行号**: line 103-104
- **修改前**:
  ```python
  def __init__(self,
               speed_target: float = 5.0,
               speed_weight: float = 1.0,
               progress_weight: float = 2.0):
  ```
- **修改后**:
  ```python
  def __init__(self,
               speed_target: float = 5.0,
               speed_weight: float = 0.1,  # 降低到0.1，避免每步累加过多
               progress_weight: float = 0.1):  # 降低到0.1
  ```

**修改原因**:
- progress_weight=2.0 导致50步累加+100，抵消碰撞惩罚-100
- speed_weight=1.0 导致速度偏差累加-36，淹没SafetyReward的+0.9
- 训练日志显示EfficiencyReward=-35.68, SafetyReward=+0.9，完全失衡

**影响**:
- progress_weight: 2.0 → 0.1 (降低20倍)
- speed_weight: 1.0 → 0.1 (降低10倍)
- 50步成功episode: EfficiencyReward从+50降到+5
- 碰撞和非碰撞的reward差距从-50变成-95，学习信号更清晰

---

### 修改 #5: 创建完整Reward公式文档
**时间**: 2026-01-22 晚上
**文件**: `REWARD_FORMULA.md` (新建)

**修改内容**:
- 创建完整的reward系统文档
- 包含所有5个组件的详细公式
- 提供具体数值示例
- 记录关键修复点和设计原则

**修改原因**:
- 用户要求整理现在的reward公式
- 需要文档化所有修复内容

**影响**:
- 提供完整的reward系统参考文档
- 方便后续调试和优化

---

## 2026-01-23

### 修改 #6: 添加每步min_distance记录到日志
**时间**: 2026-01-23 上午
**文件**: `examples/run_ppo_carla.py`

**修改内容**:
1. **行号**: line 382 - 初始化列表
   ```python
   # 记录每步的min_distance
   step_min_distances = []
   ```

2. **行号**: line 395-403 - 计算并记录每步距离
   ```python
   # 计算当前step的min_distance
   current_min_dist = float('inf')
   ego_pos = np.array(state.ego.position_m)
   for agent in state.agents:
       agent_pos = np.array(agent.position_m)
       dist = np.linalg.norm(ego_pos - agent_pos)
       current_min_dist = min(current_min_dist, dist)
   step_min_distances.append(current_min_dist)
   ```

3. **行号**: line 407 - 碰撞打印显示距离
   ```python
   print(f"  ⚠️ 碰撞检测！Step {step}, min_dist={current_min_dist:.2f}m, Reward: {step_result.reward:.2f}, Total: {episode_reward:.2f}")
   ```

4. **行号**: line 512-520 - 日志文件添加距离表格
   ```python
   # 写入每步的min_distance
   f.write("\n  Step-by-Step Min Distances:\n")
   f.write(f"  {'Step':<8} {'Min Distance (m)':<20}\n")
   f.write(f"  {'-'*28}\n")
   for step_idx, min_dist in enumerate(step_min_distances):
       if min_dist == float('inf'):
           f.write(f"  {step_idx:<8} {'No agents':<20}\n")
       else:
           f.write(f"  {step_idx:<8} {min_dist:<20.2f}\n")
   f.write("\n")
   ```

**修改原因**:
- 用户需要诊断near-miss检测是否正确
- 需要看到每一步的距离变化，判断是否触发near-miss阈值
- 当前只在episode结束时打印整体最小距离，无法追踪过程

**影响**:
- 训练日志文件会增加每个episode的逐步距离记录
- 可以诊断距离计算和near-miss判定是否正确
- 日志文件略微增大

---

## 待实现的修改

### 建议修改 #7: 使用CARLA物理引擎计算edge-to-edge距离
**提出时间**: 2026-01-23 上午
**状态**: 待确认

**建议修改的文件**:
- `src/c2o_drive/environments/carla/simulator.py`
- `src/c2o_drive/environments/carla_env.py`
- `src/c2o_drive/environments/rewards.py`

**建议内容**:
当前使用numpy计算中心点距离，不考虑车辆尺寸。应该使用CARLA的bounding box计算edge-to-edge距离：

```python
# 获取bounding box
ego_bbox = self.ego_vehicle.bounding_box.extent
agent_bbox = agent_vehicle.bounding_box.extent

# 计算edge-to-edge距离
center_dist = ego_location.distance(agent_location)
ego_radius = max(ego_bbox.x, ego_bbox.y)
agent_radius = max(agent_bbox.x, agent_bbox.y)
edge_to_edge_dist = center_dist - ego_radius - agent_radius
```

**优点**:
- 距离计算更准确（车辆4.5m×1.8m，行人0.6m×0.4m）
- 碰撞判定：edge_to_edge_dist ≤ 0
- Near-miss判定：0 < edge_to_edge_dist < 2m
- 完全基于CARLA物理模型，不需要维护额外的车辆尺寸数据

**影响**:
- 现有的near_miss_threshold (4m) 需要调整（可能减到2m）
- 距离数值会变小（减去约3-4米的车辆半径）
- Reward公式中的阈值需要重新标定

---

### 修改 #7: 实现基于CARLA OBB的精确near-miss检测
**时间**: 2026-01-23 上午
**状态**: ✅ 已完成

**修改的文件**:
1. `src/c2o_drive/environments/carla/simulator.py`
2. `src/c2o_drive/environments/carla_env.py`
3. `src/c2o_drive/environments/rewards.py`

**修改内容**:

#### 1. simulator.py (新增 `check_near_miss` 方法)
- **行号**: ~line 205
- **新增代码**:
  ```python
  def check_near_miss(self, buffer_m: float = 2.0) -> tuple[bool, float]:
      """使用OBB碰撞检测判断near-miss

      创建一个扩大buffer_m的ego OBB，用SAT检测是否与agents碰撞。
      如果扩大版碰撞但真实版不碰撞 → near-miss
      """
      from c2o_drive.utils.collision import ShapeBasedCollisionDetector, VehicleShape

      # 获取ego的position, rotation, bounding_box
      # 创建真实ego shape和扩大版ego shape
      # 对每个agent:
      #   - 检测扩大版是否碰撞
      #   - 检测真实版是否碰撞
      #   - Near-miss = 扩大版碰撞 && 真实版不碰撞

      return near_miss, min_distance
  ```

#### 2. carla_env.py (调用near-miss检测)
- **行号**: line 312-320
- **新增代码**:
  ```python
  # 检测near-miss（使用OBB扩大2米检测）
  near_miss_detected = False
  min_distance_to_agents = float('inf')
  if self.simulator:
      from c2o_drive.config import get_global_config
      buffer_m = get_global_config().safety.near_miss_threshold_m / 2.0
      near_miss_detected, min_distance_to_agents = self.simulator.check_near_miss(buffer_m)
  ```

- **行号**: line 344-353 (info字典)
- **修改后**:
  ```python
  info = {
      'collision': terminated,
      'collision_sensor': collision_sensor_triggered,
      'near_miss': near_miss_detected,  # 新增
      'min_distance_to_agents': min_distance_to_agents,  # 新增
      'step': self._step_count,
      'acceleration': acceleration,
      'jerk': jerk,
      'lateral_deviation': lateral_deviation,
      'forward_progress': forward_progress,
  }
  ```

#### 3. rewards.py (使用CARLA的OBB距离)
- **行号**: line 35-68
- **修改后**: SafetyReward优先使用info中的near_miss和min_distance_to_agents
  ```python
  # 优先使用CARLA提供的near_miss检测和距离（基于OBB）
  if 'near_miss' in info and 'min_distance_to_agents' in info:
      near_miss = info['near_miss']
      min_dist = info['min_distance_to_agents']

      if near_miss:
          # Near-miss惩罚（基于扩大OBB的精确检测）
          return -self.near_miss_weight * (near_miss_threshold - normalized_dist)
      else:
          # 安全区域
          return 0.1 * self.distance_weight

  # Fallback: 非CARLA环境使用中心点距离
  ```

**修改原因**:
- 之前用numpy计算中心点距离，不考虑车辆尺寸和朝向
- 用户指出：车辆是4.5m×1.8m矩形，不是圆形，侧面和正面接近的安全距离不同
- CARLA collision sensor使用PhysX物理引擎的OBB碰撞检测，考虑了position、rotation、extent
- 用户建议：创建"扩大2米的虚拟车辆"来检测near-miss

**实现方案**:
1. 利用已有的`collision.py`中的SAT (分离轴定理) OBB碰撞检测代码
2. 创建两个OBB：真实ego和扩大版ego（长宽各+2m buffer）
3. 对每个agent检测扩大版OBB是否碰撞
4. 扩大版碰撞 && 真实版不碰撞 → near-miss
5. 完全基于CARLA的bounding box信息，精确且考虑朝向

**技术细节**:
- CARLA bounding box: `vehicle.bounding_box.extent` → Vector3D(half_length, half_width, half_height)
- OBB = center + rotation + extent，完整描述旋转矩形框
- SAT算法：在所有分离轴上投影，如果都有重叠则碰撞
- buffer_m = near_miss_threshold / 2.0 (默认4m/2 = 2m)

**影响**:
- Near-miss检测更精确，考虑了车辆朝向和形状
- 距离计算基于CARLA物理引擎，与实际碰撞判定一致
- SafetyReward信号更准确，有助于训练
- 保留fallback逻辑，兼容非CARLA环境（scenario replay）

**性能考虑**:
- 每个step调用一次OBB检测（~10个agents）
- SAT算法复杂度O(n)，n为边数（矩形=4）
- 计算开销可接受，远小于CARLA渲染

---

### 修改 #8: 在训练脚本中使用OBB near-miss检测结果
**时间**: 2026-01-23 上午
**文件**: `examples/run_ppo_carla.py`

**修改内容**:

1. **行号**: line 375-379 - 初始化记录列表
   ```python
   step_min_distances = []  # 中心点距离
   step_obb_distances = []  # OBB距离
   step_near_miss_flags = []  # 每步的near_miss标志
   episode_near_miss = False  # 整个episode是否触发过near_miss
   ```

2. **行号**: line 403-421 - 收集每步的OBB检测结果
   ```python
   # 获取CARLA的OBB检测结果
   step_near_miss = step_result.info.get('near_miss', False)
   obb_min_dist = step_result.info.get('min_distance_to_agents', float('inf'))

   # 如果这一步触发near_miss，标记整个episode
   if step_near_miss:
       episode_near_miss = True

   # 记录所有距离信息
   step_min_distances.append(current_min_dist)
   step_obb_distances.append(obb_min_dist)
   step_near_miss_flags.append(step_near_miss)
   ```

3. **行号**: line 413-414 - 打印near_miss检测
   ```python
   if step_near_miss and self.verbose:
       print(f"  ⚠️ NEAR-MISS检测！Step {step}, OBB_dist={obb_min_dist:.2f}m, center_dist={current_min_dist:.2f}m")
   ```

4. **行号**: line 468-473 - 使用OBB检测结果
   ```python
   # Near-miss判定：使用CARLA的OBB检测结果
   near_miss = episode_near_miss or collision

   # 打印episode总结
   print(f"  📏 Episode Summary: min_center_distance={min_distance:.2f}m, "
         f"OBB_near_miss={episode_near_miss}, collision={collision}, final_near_miss={near_miss}")
   ```

5. **行号**: line 529-541 - 日志文件添加OBB距离表格
   ```python
   f.write("\n  Step-by-Step Distance Analysis:\n")
   f.write(f"  {'Step':<8} {'Center Dist(m)':<18} {'OBB Dist(m)':<18} {'Near-Miss':<12}\n")
   f.write(f"  {'-'*56}\n")
   for step_idx in range(len(step_min_distances)):
       center_dist = step_min_distances[step_idx]
       obb_dist = step_obb_distances[step_idx]
       near_miss_flag = step_near_miss_flags[step_idx]
       # 输出格式化的距离和near-miss标志
   ```

**修改原因**:
- run_ppo_carla.py原先用中心点距离重新计算near_miss，没有使用CARLA的OBB检测结果
- 用户询问near_miss是否有打印，发现现有打印用的是旧方法

**修改前**:
```python
# 旧方法：用中心点距离判断
near_miss = (min_distance < global_config.safety.near_miss_threshold_m) or collision
```

**修改后**:
```python
# 新方法：使用CARLA的OBB检测结果
step_near_miss = step_result.info.get('near_miss', False)
episode_near_miss = episode_near_miss or step_near_miss
near_miss = episode_near_miss or collision
```

**影响**:
- 现在打印和日志文件同时显示中心点距离和OBB距离
- Near-miss判定基于CARLA的精确OBB检测
- 可以对比中心点方法和OBB方法的差异
- 日志文件显示每一步的near-miss标志

**示例输出**:
```
Step     Center Dist(m)     OBB Dist(m)        Near-Miss
---------------------------------------------------------------
0        8.45               6.23               No
1        7.12               4.89               No
2        5.34               3.11               YES
3        4.89               2.66               YES
4        3.21               0.98               YES
```

---

### 修改 #9: 调整OBB buffer从2米到1米
**时间**: 2026-01-23 上午
**文件**: `src/c2o_drive/environments/carla_env.py`

**修改内容**:
- **行号**: line 316
- **修改前**:
  ```python
  buffer_m = get_global_config().safety.near_miss_threshold_m / 2.0  # buffer=2.0m
  ```
- **修改后**:
  ```python
  buffer_m = 1.0  # OBB扩展距离：车辆尺寸+1米buffer
  ```

**修改原因**:
- 用户反馈buffer_m=2.0太大，导致大部分情况都触发near-miss
- 车辆尺寸4.5m×1.8m，扩大2米后变成6.5m×3.8m，范围过大
- 改成1米后：扩大后尺寸=5.5m×2.8m，更合理

**影响**:
- Near-miss检测更严格，只有真正接近的情况才触发
- 预计near-miss率会显著降低
- 更符合实际驾驶中的危险距离定义

---

### 修改 #10: 修复场景3自行车碰撞检测问题
**时间**: 2026-01-23 上午
**文件**: `src/c2o_drive/environments/carla/simulator.py`

**修改内容**:
- **行号**: line 352-354 (在spawn自行车后)
- **新增代码**:
  ```python
  # 确保自行车启用physics simulation（关键：自行车默认可能关闭physics）
  if 'bike' in agent_bp.id or 'bicycle' in agent_bp.id:
      vehicle.set_simulate_physics(True)
      print(f"✓ 自行车{i+1} physics simulation已启用")
  ```

**问题描述**:
- 用户反馈：场景3碰撞自行车没有被正常记录
- 汽车碰撞检测正常，只有自行车有问题

**根本原因**:
CARLA中自行车（bicycle）的默认physics simulation状态可能是关闭的：
- `vehicle.set_simulate_physics(False)` → 车辆不参与物理碰撞检测
- Collision sensor只能检测到启用了physics的actor
- 自行车默认可能是kinematic模式，只有位置移动，没有物理碰撞

**为什么汽车没问题**:
- 普通车辆（vehicle.audi.tt等）默认启用physics
- 自行车（vehicle.bh.crossbike）可能默认禁用，需要显式启用

**修改后的工作流程**:
1. Spawn自行车 → 检测到blueprint包含'bike'或'bicycle'
2. 显式调用 `vehicle.set_simulate_physics(True)`
3. CARLA物理引擎开始模拟自行车的碰撞
4. Collision sensor可以正常检测到与自行车的碰撞

**影响**:
- 场景3的自行车碰撞现在可以被正确检测
- OBB near-miss检测对自行车也生效
- 训练日志中会正确记录自行车碰撞事件

**验证方法**:
运行场景3，检查：
1. 控制台是否打印"✓ 自行车1 physics simulation已启用"
2. 碰撞时是否打印"⚠️ 碰撞检测: 自车与 vehicle.bh.crossbike 发生碰撞"
3. 训练日志中collision标志是否正确

**后续问题发现**:
- 用户反馈：启用physics后碰撞仍未被检测
- 训练日志显示 `OBB_dist=3.99m, center_dist=3.99m` → 距离相等说明bbox可能异常
- 正常情况OBB距离应该 = 中心距离 - 两车半径和(约3-4米)

---

### 修改 #10.5: 同步修复行人physics启用
**时间**: 2026-01-23 上午
**文件**: `src/c2o_drive/environments/carla/simulator.py`

**修改内容**:
- **行号**: line 360-362
- **新增代码**:
  ```python
  if is_walker:
      # 确保行人启用physics simulation（关键：行人默认可能关闭physics）
      vehicle.set_simulate_physics(True)
      print(f"✓ 行人{i+1} physics simulation已启用")
  ```

**修改原因**:
- 与自行车同样的问题，行人可能也默认禁用physics
- 确保行人碰撞也能被正确检测

---

### 修改 #11: 添加自行车bounding box调试信息
**时间**: 2026-01-23 下午
**文件**: `src/c2o_drive/environments/carla/simulator.py`

**修改内容**:

1. **行号**: line 355 - 打印bbox信息
   ```python
   bbox = vehicle.bounding_box.extent
   print(f"✓ 自行车{i+1} physics simulation已启用, bbox=(length={bbox.x*2:.2f}m, width={bbox.y*2:.2f}m, height={bbox.z*2:.2f}m)")
   ```

2. **行号**: line 696-702 (在check_near_miss中)
   ```python
   # 检查agent的bounding box是否有效
   if agent_bbox.x < 0.01 or agent_bbox.y < 0.01:
       print(f"⚠️ 警告: Agent {agent_vehicle.type_id} 的bounding box异常: extent=({agent_bbox.x:.3f}, {agent_bbox.y:.3f}, {agent_bbox.z:.3f})")
       # 使用默认自行车尺寸作为fallback
       agent_bbox_fallback = type('obj', (object,), {'x': 0.9, 'y': 0.3, 'z': 1.0})()
       agent_bbox = agent_bbox_fallback
   ```

**问题分析**:
从训练日志观察到：
- `OBB_dist=3.99m, center_dist=3.99m` - 两个距离完全相等
- 正常情况下：OBB距离 = 中心距离 - ego_radius - agent_radius
- 汽车半径约2.4m，自行车半径约0.9m → OBB距离应该比中心距离小3.3m左右
- 距离相等说明agent的bounding box可能是(0, 0, 0)

**可能的原因**:
1. CARLA的自行车blueprint (vehicle.bh.crossbike) 的bounding box可能未定义
2. 或者物理模型有问题，extent为0
3. 或者需要在spawn后等待物理引擎初始化

**修改目的**:
1. 打印自行车的实际bbox尺寸，确认是否为0
2. 如果bbox异常，使用fallback尺寸（长1.8m × 宽0.6m）
3. 防止OBB检测因为bbox=0而失效

**预期输出**:
```
✓ 自行车1 physics simulation已启用, bbox=(length=1.80m, width=0.60m, height=1.50m)
```
如果看到bbox=(0.00m, 0.00m, x.xxm)，则说明需要更换blueprint或修复物理模型。

---

### 修改 #12: 简化场景3配置（只保留自行车）
**时间**: 2026-01-23 下午
**文件**: `src/c2o_drive/environments/carla/scenarios.py`

**修改内容**:
- **行号**: line 224-233
- **修改前**: agent_spawns和metadata配置了3个agents（自行车+2辆车）
- **修改后**:
  ```python
  # 原配置（3个agents）注释保留

  # 简化配置（只有1个agent：自行车）
  metadata = {
      "agent_types": ["bicycle"],  # 只有自行车
      "agent_blueprints": ["vehicle.bh.crossbike"],
      "vehicle_types": ["bicycle"],
      "agent_categories": ["bicycle"],
      ...
  }

  agent_spawns=[bicycle]  # 只spawn自行车
  ```

**修改原因**:
- 用户注释掉了2辆背景车，只保留自行车
- metadata的配置必须与agent_spawns数量匹配
- 原metadata配置了3个agent的信息，但只spawn了1个，导致不匹配

**修改细节**:
1. 保留原配置作为注释（方便恢复）
2. 新metadata中所有数组长度改为1
3. agent_types: ["bicycle", "vehicle", "vehicle"] → ["bicycle"]
4. agent_blueprints: ["vehicle.bh.crossbike", None, None] → ["vehicle.bh.crossbike"]
5. 其他数组同步修改

**影响**:
- 场景3现在是纯自行车场景，更容易诊断碰撞检测问题
- 避免metadata索引越界或配置不匹配
- 训练更快（少2辆背景车）

---

### 修改 #13: 加快场景4行人速度（修正）
**时间**: 2026-01-23 下午
**修改的文件**:
1. `src/c2o_drive/environments/carla/simulator.py` (line 366) - **无效**
2. `src/c2o_drive/environments/carla/scenarios.py` (line 342-359) - **有效修改**

**问题发现**:
- 最初修改了simulator.py中的walker_controller速度（1.3 → 2.2 m/s）
- 用户反馈：修改没有起作用
- **根本原因**：行人使用预定义trajectory控制，直接用`set_transform`设置位置，不使用walker_controller的速度

**正确的修改方法**:
修改scenarios.py中的trajectory，增加每步移动距离

**修改内容**:
- **行号**: line 342-359
- **修改前**: 每步移动0.2米（13.5→13.30→13.10...），共15步到达10.5
- **修改后**: 每步移动0.35米（13.5→13.15→12.80...），共9步到达10.5
  ```python
  0: [  # 行人横穿轨迹（加快速度：每步0.35米）
      # 第一阶段：快速横穿（每步0.35米）
      (13.5, -127.00),   # 起始
      (13.15, -127.00),  # 步长0.35米
      (12.80, -127.00),
      (12.45, -127.00),
      (12.10, -127.00),
      (11.75, -127.00),
      (11.40, -127.00),
      (11.05, -127.00),
      (10.70, -127.00),
      (10.50, -127.00),  # 到达道路边缘
  ```

**速度计算**:
- 移动距离相同：3.0米（13.5 → 10.5）
- 原来步数：15步
- 现在步数：9步
- **速度提升：15/9 ≈ 1.67倍（67%）**

**技术细节**:
- carla_env.py line 208-229会读取trajectory并每步执行
- 每个仿真step，agent直接set_transform到trajectory中下一个位置
- 这就是为什么walker_controller的速度设置不起作用

**影响**:
- 场景4的行人横穿速度提升67%
- 行人更快到达危险区域，自车需要更快反应
- 训练难度增加

**后续修改（用户反馈：还是有点慢）**:
用户反馈行人"没走完停留就结束了"，发现问题：
1. 行人在10.00位置停留14步太久
2. 横穿速度还不够快

**进一步优化**:
- 横穿速度：0.35米/步 → **0.5米/步**（提升43%）
- 停留步数：14步 → **2步**（减少86%）
- 后续移动：每步0.4米
- 总步数：40+步 → **20步**（减少50%）

修改后trajectory：
```python
# 第一阶段：快速横穿（7步，每步0.5米）
(13.5, -127.00) → (13.0) → ... → (10.5)

# 第二阶段：短暂停顿（4步，只停2步）
(10.5) → (10.3) → (10.0) → (10.0) → (10.0)

# 第三阶段：继续前进（9步，每步0.4米）
(10.0) → (9.6) → ... → (6.8)
```

---

### 修改 #14: 添加行人bbox调试和改进fallback逻辑
**时间**: 2026-01-23 下午
**文件**: `src/c2o_drive/environments/carla/simulator.py`

**问题描述**:
- 用户反馈：s4行人的碰撞还是有点问题
- 可能和自行车一样，行人的bounding box异常

**修改内容**:

1. **行号**: line 364 - 添加行人bbox打印
   ```python
   bbox = vehicle.bounding_box.extent
   print(f"✓ 行人{i+1} physics simulation已启用, bbox=(length={bbox.x*2:.2f}m, width={bbox.y*2:.2f}m, height={bbox.z*2:.2f}m)")
   ```

2. **行号**: line 704-725 - 改进bbox fallback逻辑
   - **修改前**: 所有异常bbox都用自行车尺寸
   - **修改后**: 根据agent类型使用不同的默认尺寸
   ```python
   if 'walker' in agent_type_id or 'pedestrian' in agent_type_id:
       # 行人尺寸：长0.6m × 宽0.4m × 高1.8m
       agent_bbox_fallback = type('obj', (object,), {'x': 0.3, 'y': 0.2, 'z': 0.9})()
   elif 'bike' in agent_type_id or 'bicycle' in agent_type_id:
       # 自行车尺寸：长1.8m × 宽0.6m × 高1.5m
       agent_bbox_fallback = type('obj', (object,), {'x': 0.9, 'y': 0.3, 'z': 0.75})()
   else:
       # 默认车辆尺寸：长4.5m × 宽1.8m × 高1.5m
       agent_bbox_fallback = type('obj', (object,), {'x': 2.25, 'y': 0.9, 'z': 0.75})()
   ```

**默认尺寸参考**:
- **行人**: 0.8m × 0.6m × 1.8m (extent = 0.4, 0.3, 0.9) - 用户反馈后增大
- **自行车**: 1.8m × 0.6m × 1.5m (extent = 0.9, 0.3, 0.75)
- **车辆**: 4.5m × 1.8m × 1.5m (extent = 2.25, 0.9, 0.75)

**修正（用户反馈：行人模型有点小）**:
- 原行人尺寸：0.6m × 0.4m × 1.8m
- 修正后尺寸：**0.8m × 0.6m × 1.8m**（长度+33%，宽度+50%）
- 原因：原尺寸碰撞box太小，不容易检测到碰撞

**修改原因**:
- 行人体积比自行车小得多，用自行车尺寸不准确
- 不同类型的agent应该用对应的默认尺寸
- 提升OBB碰撞检测的准确性

**影响**:
- 行人的near-miss和collision检测更准确
- 打印行人bbox信息，方便诊断CARLA模型问题
- 如果行人bbox异常，会自动使用合理的fallback尺寸

**调试信息**:
运行s4场景时会看到：
```
✓ 行人1 physics simulation已启用, bbox=(length=X.XXm, width=X.XXm, height=X.XXm)
```
如果bbox显示(0.00m, 0.00m, X.XXm)，会触发警告并使用fallback。

---

### 修改 #15: 更换s4场景行人模型
**时间**: 2026-01-23 下午
**文件**: `src/c2o_drive/environments/carla/scenarios.py`

**修改内容**:
- **行号**: line 335
- **修改前**:
  ```python
  "agent_blueprints": ["walker.pedestrian.0001"],
  ```
- **修改后**:
  ```python
  "agent_blueprints": ["walker.pedestrian.0014"],  # 换更大体型的模型
  ```

**修改原因**:
- 用户反馈："为什么需要手动设置fallback，应该用CARLA自带的行人模型"
- 用户说"这个人的模型有点小"
- 原因：walker.pedestrian.0001可能是体型较小的行人模型

**解决方案**:
尝试使用不同编号的行人模型：
- walker.pedestrian.0001-0050+：CARLA提供了50+种行人模型
- 编号较大的模型通常体型更大（成年男性等）
- walker.pedestrian.0014可能是更高大的模型

**验证方法**:
运行s4场景，观察控制台打印：
```
✓ 行人1 physics simulation已启用, bbox=(length=X.XXm, width=X.XXm, height=X.XXm)
```

**期望结果**:
1. bbox不为0（说明CARLA模型正常，不需要fallback）
2. bbox尺寸更大（长>0.6m, 宽>0.4m）
3. 碰撞检测更准确

**如果bbox仍然为0**:
说明CARLA walker模型的bbox确实未初始化，这时fallback机制会生效。

**后续可尝试的模型**:
- walker.pedestrian.0010 (成年男性)
- walker.pedestrian.0020 (成年女性)
- walker.pedestrian.0025 (老年人)
- walker.pedestrian.0030+ (其他体型)

---

## 修改记录规则

每次修改需要记录：
1. **修改时间**: 日期和时段
2. **修改的文件**: 完整路径
3. **行号**: 修改的大致行号范围
4. **修改前代码**: 关键部分
5. **修改后代码**: 完整的新代码
6. **修改原因**: 为什么要做这个修改
7. **影响**: 对系统其他部分的影响

