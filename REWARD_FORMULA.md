# C2O-Drive Reward公式说明

## 总Reward计算

```
Total Reward = SafetyReward × 1.0
             + ComfortReward × 0.5
             + EfficiencyReward × 1.0
             + CenterlineReward × 0.3
             + TimeReward × 1.0
```

---

## 1. SafetyReward（权重 1.0）

### 碰撞检测
```python
if collision:
    return -100.0  # 固定碰撞惩罚
```

### 距离分级惩罚（未碰撞时）
```python
min_dist = min(距离到所有agents)
critical_distance = 2.0  # 临界距离
near_miss_threshold = 4.0  # near-miss阈值（从global_config读取）

if min_dist < 2.0:
    # 严重惩罚区间
    SafetyReward = -1.0 × (2.0 - min_dist)

elif min_dist < 4.0:
    # Near-miss区间（轻度惩罚）
    SafetyReward = -0.3 × (4.0 - min_dist)

else:
    # 安全区间
    SafetyReward = +0.1
```

### 示例
- 距离0.5m: -1.5 × 1.0 = **-1.5**
- 距离1.5m: -0.5 × 1.0 = **-0.5**
- 距离3.0m: -0.3 × 1.0 = **-0.3**
- 距离5.0m: +0.1 × 1.0 = **+0.1**
- 碰撞: **-100.0**

---

## 2. ComfortReward（权重 0.5）

```python
accel = info['acceleration']  # 加速度
jerk = info['jerk']           # 加加速度

accel_penalty = -1.0 × |accel|
jerk_penalty = -1.0 × |jerk|

ComfortReward = (accel_penalty + jerk_penalty) × 0.5
```

### 典型值（每步）
- 平稳驾驶: (-0.1 - 0.1) × 0.5 = **-0.1**
- 急加速: (-2.0 - 1.0) × 0.5 = **-1.5**

---

## 3. EfficiencyReward（权重 1.0）

### 参数
```python
speed_target = 5.0  # 目标速度（m/s）
speed_weight = 0.1  # 速度惩罚权重（修复后）
progress_weight = 0.1  # 前进奖励权重（修复后）
```

### 计算公式
```python
# Speed惩罚
current_speed = ||velocity||
speed_penalty = -0.1 × |current_speed - 5.0|

# Progress奖励（根据初始yaw方向计算）
forward_progress = info['forward_progress']  # 单步前进距离（有正负）
progress_reward = 0.1 × forward_progress

EfficiencyReward = speed_penalty + progress_reward
```

### 示例（单步）
速度1.0 m/s, 前进1.0m:
- speed: -0.1 × 4.0 = -0.4
- progress: 0.1 × 1.0 = 0.1
- total: **-0.3**

速度5.0 m/s, 前进5.0m:
- speed: -0.1 × 0 = 0
- progress: 0.1 × 5.0 = 0.5
- total: **+0.5**

速度6.0 m/s, 前进6.0m:
- speed: -0.1 × 1.0 = -0.1
- progress: 0.1 × 6.0 = 0.6
- total: **+0.5**

### Episode累计（10步）
速度3.0 m/s:
- speed: -0.1 × 2.0 × 10 = -2.0
- progress: 0.1 × 30 = 3.0
- total: **+1.0**

---

## 4. CenterlineReward（权重 0.3）

```python
lateral_deviation = info['lateral_deviation']  # 横向偏离（米）
max_deviation = 5.0  # 最大允许偏离

if lateral_deviation > max_deviation:
    CenterlineReward = -1.0 × (lateral_deviation - max_deviation)
else:
    CenterlineReward = 0.0
```

### 示例
- 偏离2.0m: 0 × 0.3 = **0**
- 偏离6.0m: -1.0 × 0.3 = **-0.3**

---

## 5. TimeReward（权重 1.0）

```python
TimeReward = -0.1  # 每个step固定惩罚
```

### 作用
鼓励尽快完成任务，避免无限停留。

---

## 完整Episode示例

### 场景：10步成功避障（距离保持5米，速度5 m/s）

```
SafetyReward:     +0.1 × 10 = 1.0
ComfortReward:    -0.1 × 10 × 0.5 = -0.5
EfficiencyReward: (0 + 5.0) = 5.0
CenterlineReward: 0 × 10 × 0.3 = 0
TimeReward:       -0.1 × 10 = -1.0
---------------------------------------
Total Reward = 1.0 - 0.5 + 5.0 + 0 - 1.0 = +4.5
```

### 场景：4步撞车（距离从5米→1米→碰撞）

```
Step 1-3:
SafetyReward:     (+0.1 - 0.5 - 1.0) = -1.4
ComfortReward:    -0.1 × 3 × 0.5 = -0.15
EfficiencyReward: 0.5 × 3 = 1.5
CenterlineReward: 0
TimeReward:       -0.1 × 3 = -0.3

Step 4 (碰撞):
SafetyReward:     -100.0
其他组件:         约 -0.5

---------------------------------------
Total Reward ≈ -1.4 - 0.15 + 1.5 - 0.3 - 100.0 - 0.5 = -100.85
```

---

## 关键修复点（2026-01-23）

### 修复前的问题
1. **progress_weight = 2.0** → 50步累加100，抵消碰撞惩罚
2. **speed_weight = 1.0** → 速度偏差累加过大，淹没安全信号
3. **near-miss区间给正奖励** → 2-4米之间应该惩罚但给了+0.1

### 修复后
1. ✅ progress_weight = 0.1（降低20倍）
2. ✅ speed_weight = 0.1（降低10倍）
3. ✅ near-miss区间 -0.3 × (4.0 - dist)
4. ✅ SafetyReward分级惩罚（critical < 2m, near-miss 2-4m, safe ≥ 4m）

---

## Reward设计原则

1. **碰撞惩罚绝对主导**：-100远大于其他任何组合
2. **安全 > 效率**：SafetyReward × 1.0, EfficiencyReward × 1.0，但Safety优先
3. **避免累加失衡**：每步惩罚/奖励控制在合理范围，避免长episode导致某项过大
4. **梯度清晰**：成功(+5) vs 碰撞(-100)，差距明显，学习信号强

---

## 配置文件位置

- 主配置：`src/c2o_drive/environments/rewards.py` (create_default_reward)
- 全局参数：`src/c2o_drive/config/global_config.py` (SafetyConfig)
- 修改建议：调整权重比例，不要修改单个组件的内部计算逻辑
