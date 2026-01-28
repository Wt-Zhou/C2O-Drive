# Rainbow DQN修复总结

## 修复日期
2026-01-26

## 修复内容概述

将PPO训练中已修复的问题应用到Rainbow DQN，包括：
1. Episode-level transitions（核心算法bug）
2. 使用CARLA OBB检测
3. 详细训练日志
4. 添加_train_step()方法

2026-01-27新增：
1. 修复episode-level训练无法开始的问题（warmup门槛判断）
2. 调整episode-level训练目标为reward-only分布（无bootstrap）
3. 训练时重置NoisyNet噪声
4. 修复训练日志打印None导致崩溃的问题
5. 训练结束保存reward曲线图（与PPO一致）
6. 增加评估模式（deterministic actions，与训练控制一致）
7. 增加Q值Top-K调试输出（观察是否探索左偏动作）
8. 默认使用GlobalConfig的lattice参数（与PPO一致）
9. 修复reward-only投影在边界条件下的索引错误
10. Rainbow状态编码与PPO一致（手工特征）
11. 修复手工特征编码器无参数导致的StopIteration
12. 提高前向进度reward权重（0.3→0.6）
13. Rainbow环境dt与PPO一致（使用lattice.dt）
14. 训练时打印轨迹点数与实际执行步数
15. 修复Env dt打印位置导致的UnboundLocalError
16. 修复CARLA连接前未初始化config导致的报错
17. 增加GlobalConfig/Config的horizon打印用于排查
18. 修复Rainbow使用time.default_horizon导致的horizon不一致
19. 增大NoisyNet探索强度
20. 增加按动作统计reward分布
21. 降低NoisyNet探索强度以偏向高回报选择
22. 进一步降低噪声与学习率以稳定策略
23. Warmup阶段提高NoisyNet噪声以增加探索
24. 自车偏移判定加入场景线约束（-90°与0°方向）
25. 保留原偏移计算并对出线加重惩罚
26. 出线惩罚最小距离阈值=1
27. 降低PER采样偏置并增加前期均匀采样
28. 降低偏离惩罚权重（不影响出线惩罚）

---

## 修复详情

### 1. State Features提取修复

**问题**：代码调用不存在的`_extract_state_features`方法

**原因**：
- PPO使用手动特征提取（`_extract_state_features`）
- Rainbow DQN使用神经网络编码（`WorldStateEncoder`）
- 错误地混用了两种方式

**修复**：
- 修改`run_rainbow_dqn_carla.py:265-269`：删除`_extract_state_features`调用
- 直接使用`self.planner.q_network([state])`（WorldState列表）
- 修改`_compute_q_statistics`方法接收WorldState而非tensor

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 2. Episode-level Transitions存储

**问题**：每个trajectory存储多个step-level transitions

**原因**：
- Rainbow DQN在每步循环中调用`planner.update(transition)`
- 每个50步episode存储50个transitions
- Action语义（选择轨迹）与Reward语义（单步reward）不匹配

**修复**：
- 删除循环内的`transition`创建和`planner.update()`调用
- Episode结束后存储单个transition：
  ```python
  self.planner.replay_buffer.push(
      state=initial_state,
      action=action_idx,
      reward=episode_reward,  # 总reward
      next_state=final_state,
      done=True
  )
  ```
- 添加定期训练逻辑（buffer满时调用`_train_step()`）

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py:281-345`

---

### 3. 使用CARLA OBB检测结果

**问题**：手动计算中心点距离，不准确

**原因**：
- 原代码使用`_compute_min_distance()`手动计算
- 忽略车辆尺寸和朝向
- CARLA已经提供精确的OBB检测结果

**修复**：
- 在执行循环中收集OBB距离：
  ```python
  step_near_miss = step_result.info.get('near_miss', False)
  obb_min_dist = step_result.info.get('min_distance_to_agents', float('inf'))
  ```
- 记录每步的OBB距离和center距离（用于对比）
- 使用OBB检测结果更新episode_data
- 注释掉`_compute_min_distance()`方法

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py:288-350`

---

### 4. 详细训练日志

**问题**：缺少step-by-step距离跟踪和near-miss详细信息

**修复**：
- 实时打印near-miss检测：
  ```python
  if step_near_miss and self.verbose:
      print(f"  ⚠️ NEAR-MISS检测！Step {step}, OBB_dist={obb_min_dist:.2f}m, center_dist={current_min_dist:.2f}m")
  ```
- 实时打印碰撞信息
- Episode结束后打印step-by-step距离分析表格

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py:311-327, 377-385`

---

### 5. 添加_train_step()方法

**问题**：Rainbow DQN的训练逻辑在`update()`中，无法单独调用

**原因**：
- PPO有独立的`_ppo_update()`方法
- Rainbow DQN需要类似的方法用于episode-level训练

**修复**：
- 从`update()`方法中抽取训练逻辑
- 创建`_train_step()`方法：
  - 采样batch
  - 计算Q分布
  - 计算loss（KL divergence）
  - 更新网络
  - 更新target network
- 在`run_episode()`中调用：
  ```python
  if buffer_len >= self.planner.config.training.batch_size:
      metrics = self.planner._train_step()
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/planner.py:293-381`

---

### 6. 修复输出目录结构

**问题**：Rainbow DQN直接使用base目录，多次运行会互相覆盖

**原因**：
- PPO创建带时间戳的子目录：`s4_20260126_143052`
- Rainbow DQN直接使用`outputs/rainbow_dqn_carla`
- 无法区分不同运行的结果

**修复**：
- 添加timestamp和run_name生成：
  ```python
  timestamp = time.strftime("%Y%m%d_%H%M%S")
  run_name = f"{args.scenario}_{timestamp}"
  output_dir = Path(args.output_dir) / run_name
  log_dir = (Path(args.log_dir) / run_name) if TENSORBOARD_AVAILABLE else None
  ```
- 与PPO完全一致的目录结构

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py:693-701`

---

### 7. 修复episode-level训练无法开始

**问题**：episode-level训练永远无法开始（warmup条件永远为True）

**原因**：
- 训练入口使用`_train_step()`而不是`update()`，不会调用`select_action()`
- `_train_step()`用`self._step_count < warmup_steps`进行门槛判断
- `self._step_count`在episode-level训练路径中不会递增

**修复**：
- 将`_train_step()`中的warmup条件改为基于buffer大小：
  ```python
  if len(self.replay_buffer) < self.config.training.warmup_steps:
      return UpdateMetrics(custom={'warmup': True})
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/planner.py`

---

### 8. Episode-level训练：reward-only目标分布 + NoisyNet重置

**问题**：episode-level设定下仍使用C51一步bootstrap目标分布，目标不自洽；训练更新时NoisyNet噪声不重置。

**原因**：
- 单次决策没有后续决策，`next_state`不应参与bootstrap
- `_train_step()`仍用`_project_distribution()`（包含`gamma * next_dist`）
- NoisyNet在训练更新时不重置噪声，探索信号不稳定

**修复**：
- 在`_train_step()`中改用reward-only分布（无bootstrap）：
  ```python
  target_dist = self._project_reward_distribution(rewards)
  ```
- 训练更新前重置噪声：
  ```python
  self.q_network.reset_noise()
  ```
- 新增`_project_reward_distribution()`用于将episode总回报投影到C51 atoms。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/planner.py`

---

### 9. 训练日志None格式化崩溃修复

**问题**：当`UpdateMetrics`中`loss/q_value/td_error`为`None`时，格式化输出触发`TypeError`。

**原因**：
- warmup或buffer不足时返回`UpdateMetrics`不包含数值
- 日志打印中直接使用`{value:.4f}`格式化

**修复**：
- 对`None`做保护，输出`N/A`：
  ```python
  loss_str = f"{loss:.4f}" if loss is not None else "N/A"
  q_value_str = f"{q_value:.4f}" if q_value is not None else "N/A"
  td_error_str = f"{td_error:.4f}" if td_error is not None else "N/A"
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 10. 保存训练曲线图

**问题**：Rainbow DQN训练结束后没有保存`training_curve.png`，即使已有绘图函数。

**原因**：
- `_save_training_curve()`方法定义了但未在训练结束调用。

**修复**：
- 在`train()`结束保存metrics后调用：
  ```python
  self._save_training_curve()
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 11. 增加评估模式（deterministic actions）

**问题**：Rainbow DQN缺少评估模式，无法在不训练的情况下复现实验并保持控制逻辑一致。

**修复**：
- 新增命令行参数：
  ```bash
  --eval --load /path/to/checkpoint.pt
  ```
- 评估时使用Q网络确定性动作（argmax），不进行训练。
- 评估执行控制逻辑与训练一致（相同P控制和速度控制）。

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 12. Q值Top-K调试输出

**问题**：不清楚训练时是否会探索左偏动作（Q值分布不可见）。

**修复**：
- 新增调试参数：
  ```bash
  --debug-q --debug-q-topk 5
  ```
- 每个episode打印Q值Top-K动作及其`lateral_offset`和`target_speed`。

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 13. 默认使用GlobalConfig的lattice参数

**问题**：Rainbow DQN默认lattice参数与PPO不一致，导致“15个轨迹”并非同一组。

**修复**：
- 训练默认使用`RainbowDQNConfig.from_global_config()`。
- 若要使用本地默认参数，需显式加：
  ```bash
  --no-global-config
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 14. reward-only投影索引错误修复

**问题**：`_project_reward_distribution()`在`l==u`时索引维度错误，触发`IndexError`。

**原因**：
- `eq_mask`在(batch,1)上取mask后，`l[eq_mask]`变成1D
- 再调用`.squeeze(1)`导致维度越界

**修复**：
- 先将`eq_mask`压成(batch,)索引，再用同维索引选择`l`：
  ```python
  eq_idx = eq_mask.squeeze(1)
  target_dist[eq_idx, l.squeeze(1)[eq_idx]] = 1.0
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/planner.py`

---

### 15. Rainbow状态编码与PPO一致

**问题**：PPO使用手工特征提取，Rainbow使用注意力编码器，导致状态输入不一致。

**修复**：
- 将`WorldStateEncoder`改为PPO同款手工特征（归一化位置/速度、heading、goal相对位移、最近N个agent相对位移与heading）。
- 保持`state_feature_dim`作为padding长度。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/trajectory_encoder.py`

---

### 16. 手工特征编码器无参数的device获取修复

**问题**：将编码器改成手工特征后无可训练参数，`next(self.parameters())`触发`StopIteration`。

**修复**：
- 直接使用`config.device`确定device：
  ```python
  device = torch.device(self.config.device if hasattr(self.config, "device") else "cpu")
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/trajectory_encoder.py`

---

### 17. 提高前向进度reward权重

**问题**：前向进度的reward权重偏低，信号弱。

**修复**：
- 将 `EfficiencyReward.progress_weight` 从 `0.3` 提高到 `0.6`。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/environments/rewards.py`

---

### 18. Rainbow环境dt与PPO一致

**问题**：Rainbow训练时环境dt固定为1.0，PPO使用`config.lattice.dt`，导致时间尺度和执行步长不一致。

**修复**：
- Rainbow连接环境时使用`config.lattice.dt`：
  ```python
  dt=config.lattice.dt
  ```
- 打印Env dt用于确认。

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 19. 训练时打印轨迹点数与实际执行步数

**问题**：难以对比“轨迹长度”和“执行步数”的截断情况。

**修复**：
- 在训练中打印轨迹点数、原始max_steps与限制后max_steps：
  ```python
  print(f"  📊 轨迹信息: 轨迹点数={num_waypoints}, 原始max_steps={original_max_steps}, 限制后max_steps={max_steps}")
  ```

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 20. 修复Env dt打印位置导致的UnboundLocalError

**问题**：在config创建前打印`config.lattice.dt`导致`UnboundLocalError`。

**修复**：
- 将`Env dt`打印移动到`config`创建之后。

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 21. 修复CARLA连接前未初始化config导致的报错

**问题**：连接CARLA时使用`config.lattice.dt`，但`config`还未初始化，触发`UnboundLocalError`。

**修复**：
- 先创建`RainbowDQNConfig`，再连接CARLA。

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 22. 增加GlobalConfig/Config的horizon打印

**问题**：运行时轨迹点数异常，需要确认GlobalConfig与实际Config的horizon取值。

**修复**：
- 启动时打印GlobalConfig lattice参数
- 打印实际config的lattice.horizon

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 23. 修复Rainbow使用time.default_horizon导致的horizon不一致

**问题**：Rainbow从GlobalConfig读取horizon时使用`time.default_horizon`，与`lattice.horizon`不一致。

**修复**：
- `RainbowDQNConfig.from_global_config()`改为使用`gc.lattice.horizon`与`gc.lattice.dt`。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/config.py`

---

### 24. 增大NoisyNet探索强度

**问题**：前100个episode探索方向单一，NoisyNet噪声偏弱。

**修复**：
- 将`noisy_sigma`默认值从`0.5`提高到`0.8`，增强参数噪声探索。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/config.py`

---

### 25. 增加按动作统计reward分布

**目的**：观察同一action的回报均值/方差，判断策略是否因回报噪声导致频繁切换。

**实现**：
- 训练过程中记录`action_idx -> rewards`列表
- 训练结束打印每个action的均值、标准差与样本量

**文件**：`/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 26. 降低NoisyNet探索强度

**目的**：减少噪声波动，使策略更倾向选择高回报动作。

**修复**：
- 将`noisy_sigma`从`0.8`降低到`0.2`。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/config.py`

---

### 27. 进一步降低噪声与学习率

**目的**：让策略更稳定地选择高回报动作。

**修改**：
- `noisy_sigma` 从 `0.2` 降到 `0.1`
- `--lr` 默认从 `6.25e-5` 降到 `3e-5`

**文件**：
- `/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/config.py`
- `/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 28. Warmup阶段提高NoisyNet噪声

**目的**：避免warmup阶段探索过于一致，提升动作覆盖率。

**修改**：
- 新增`warmup_noisy_sigma`（默认0.5）
- warmup阶段设置更高噪声，训练开始后恢复到`noisy_sigma`

**文件**：
- `/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/config.py`
- `/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/noisy_linear.py`
- `/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/network.py`
- `/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 29. 自车偏移判定加入场景线约束

**需求**：统一所有算法的偏移判定规则：
- yaw≈-90° 时，x < 4.5 或 x > 9.5 视为出线并惩罚
- yaw≈0° 时，y < -136 视为出线并惩罚

**实现**：
- 在 `carla_env.py` 中修改 `lateral_deviation` 计算，使用上述边界条件。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/environments/carla_env.py`

---

### 30. 保留原偏移计算并对出线加重惩罚

**需求**：保持原有`lateral_deviation`计算，同时对出线情况施加更重惩罚。

**实现**：
- 保留原偏移计算
- 新增`out_of_lane`与`out_of_lane_distance`信息
- `CenterlineReward` 在出线时增加额外惩罚

**文件**：
- `/home/dell/Desktop/C2O-Drive/src/c2o_drive/environments/carla_env.py`
- `/home/dell/Desktop/C2O-Drive/src/c2o_drive/environments/rewards.py`

---

### 31. 出线惩罚最小距离阈值

**需求**：只要出线，惩罚至少按距离=1计算。

**实现**：
- `out_of_lane_distance` 取 `max(1.0, 实际超出距离)`。

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/environments/rewards.py`

---

### 32. 降低PER偏置 + 前期均匀采样

**目的**：缓解动作塌缩，提升前期动作覆盖。

**修改**：
- PER alpha 从 0.6 降到 0.4
- 新增 `--random-episodes`：前 N 个 episode 均匀随机选动作

**文件**：
- `/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/config.py`
- `/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`

---

### 33. 降低偏离惩罚权重

**需求**：过线惩罚不变，仅降低偏离惩罚。

**修改**：
- `CenterlineReward.weight` 从 `1.0` 降到 `0.5`

**文件**：`/home/dell/Desktop/C2O-Drive/src/c2o_drive/environments/rewards.py`

---

## 修改的文件列表

1. `/home/dell/Desktop/C2O-Drive/examples/run_rainbow_dqn_carla.py`
   - 修改state features提取
   - 修改episode-level transitions存储
   - 添加OBB距离跟踪
   - 添加详细日志（reward breakdown, episode summary）
   - 添加探索机制（reset_noise, train mode）
   - 修复输出目录结构（添加timestamp子目录，与PPO一致）
   - 注释掉`_compute_min_distance()`
   - 修复训练日志打印None导致崩溃（N/A保护）
   - 训练结束保存training_curve.png
   - 增加评估模式（--eval/--load），控制逻辑与训练一致
   - 增加Q值Top-K调试输出（--debug-q/--debug-q-topk）
   - 默认使用GlobalConfig的lattice参数（新增--no-global-config可关闭）
   - 连接环境时使用config.lattice.dt，并打印Env dt
   - 打印轨迹点数与max_steps截断信息

2. `/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/planner.py`
   - 添加`_train_step()`方法
   - 修复episode-level训练warmup门槛判断（基于buffer大小）
   - episode-level训练改为reward-only分布（无bootstrap）
   - 训练更新前重置NoisyNet噪声
   - 修复reward-only投影在边界条件下的索引错误

3. `/home/dell/Desktop/C2O-Drive/src/c2o_drive/algorithms/rainbow_dqn/trajectory_encoder.py`
   - 使用PPO一致的手工特征提取作为Rainbow状态编码
   - 修复无参数时device获取导致的StopIteration

4. `/home/dell/Desktop/C2O-Drive/src/c2o_drive/environments/rewards.py`
   - 提高前向进度reward权重（0.3→0.6）

---

## 验证方法

运行训练测试：
```bash
python examples/run_rainbow_dqn_carla.py --scenario s4 --episodes 10 --max-steps 50
```

期望输出：
- ✓ 每个episode只存储1个transition
- ✓ Episode reward是累积值（如-5.0），不是单步值（-0.1）
- ✓ Near-miss显示OBB距离信息
- ✓ Step-by-step距离分析表格
- ✓ 训练更新信息（buffer满时）

---

## 记录的疑问

### WorldStateEncoder缺少的特征

**PPO有，Rainbow DQN没有**：
1. **相对距离**：PPO计算ego到agents的相对距离，Rainbow DQN用绝对坐标
2. **Goal信息**：PPO包含到goal的相对距离，Rainbow DQN没有goal信息
3. **归一化**：PPO归一化坐标和速度，Rainbow DQN用原始值

**可能的影响**：
- 位置不变性缺失：相同相对关系在不同位置产生不同特征
- 缺少目标导向信息
- 数值范围大，可能影响训练稳定性

**但Rainbow DQN的优势**：
- 神经网络可以学习这些关系
- Multi-head Attention可以学习agent交互
- 更灵活，处理可变数量agents

**建议**：
- 先测试当前效果
- 如果效果不好，考虑改进WorldStateEncoder添加相对距离和归一化

---

## PPO vs Rainbow DQN对比

| 特性 | PPO | Rainbow DQN |
|-----|-----|-------------|
| **特征提取** | 手动（_extract_state_features） | 神经网络（WorldStateEncoder） |
| **相对距离** | ✓ 手动计算 | ✗ 绝对坐标 |
| **Goal信息** | ✓ 有 | ✗ 无 |
| **归一化** | ✓ 手动归一化 | ✗ 原始值 |
| **Agent数量** | 固定10个 | 可变（Attention） |
| **Reward函数** | create_default_reward() | create_default_reward() |
| **Transition存储** | Episode-level（修复后） | Episode-level（修复后） |
| **OBB检测** | ✓ 使用 | ✓ 使用（修复后） |
| **详细日志** | ✓ 有 | ✓ 有（修复后） |
| **输出目录** | {scenario}_{timestamp} | {scenario}_{timestamp}（修复后） |
| **Exploration** | Entropy bonus | Noisy Nets（修复后） |

---

## 与PPO一致性

✅ **已对齐**：
- Episode-level transitions存储
- 使用CARLA OBB检测
- 详细训练日志（reward breakdown, episode summary）
- Reward函数相同
- 输出目录结构（scenario_timestamp）
- Exploration机制（PPO用entropy, Rainbow DQN用Noisy Nets）

❌ **仍有差异**：
- 特征提取方式（设计理念不同）
- 训练时机（PPO是on-policy，Rainbow DQN是off-policy）

---

## 后续可能的改进

1. **改进WorldStateEncoder**：
   - 添加相对距离计算
   - 添加Goal信息编码
   - 添加特征归一化
   - 需要修改network结构

2. **训练频率调优**：
   - 当前每个episode训练一次（如果buffer满）
   - 可以调整为每N个episode训练一次
   - 或者每次训练多个iterations

3. **ReplayBuffer大小调优**：
   - 当前使用默认值
   - 可以根据实际情况调整capacity

4. **对比测试**：
   - 相同scenario下对比PPO和Rainbow DQN的效果
   - 分析特征提取方式对性能的影响
