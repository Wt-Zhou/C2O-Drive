# Rainbow DQN vs PPO 完整流程对比检查

## 日期
2026-01-26

## 目的
系统性检查Rainbow DQN实现，对比PPO找出所有潜在问题

---

## 1. 环境重置 (Environment Reset)

### PPO
```python
# run_ppo_carla.py:298-309
state, info = self.env.reset(seed=seed, options=reset_options)
reference_path = info.get('reference_path', [])
self.planner.reset()
```

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:238-242
state, info = self.env.reset(seed=seed, options=reset_options)
reference_path = info.get('reference_path', [])
self.planner.reset()
```

**状态**: ✅ 一致

---

## 2. 轨迹生成 (Trajectory Generation)

### PPO
```python
# run_ppo_carla.py:314-333
candidate_trajectories = self.planner.lattice_planner.generate_trajectories(
    reference_path=reference_path,
    horizon=self.planner.config.lattice.horizon,
    dt=self.planner.config.lattice.dt,
    ego_state=ego_state_tuple,
)
```

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:258-263
candidate_trajectories = self.planner.lattice_planner.generate_trajectories(
    reference_path=reference_path,
    horizon=self.planner.config.lattice.horizon,
    dt=self.planner.config.lattice.dt,
    ego_state=ego_state_tuple,
)
```

**状态**: ✅ 一致

---

## 3. 状态特征提取 (State Feature Extraction)

### PPO
```python
# run_ppo_carla.py:345
state_features = self.planner._extract_state_features(state)

# planner.py:391-442
def _extract_state_features(self, world_state: WorldState) -> torch.Tensor:
    features = []

    # Ego: [x/100, y/100, speed/30, cos(yaw), sin(yaw)]
    ego_speed = np.linalg.norm(ego.velocity_mps)
    features.extend([
        ego.position_m[0] / 100.0,
        ego.position_m[1] / 100.0,
        ego_speed / 30.0,
        np.cos(ego.yaw_rad),
        np.sin(ego.yaw_rad),
    ])

    # Goal: [rel_x/100, rel_y/100]
    if hasattr(world_state, 'goal') and world_state.goal is not None:
        rel_x = (goal.position_m[0] - ego.position_m[0]) / 100.0
        rel_y = (goal.position_m[1] - ego.position_m[1]) / 100.0
        features.extend([rel_x, rel_y])
    else:
        features.extend([0.0, 0.0])

    # Agents (max 10): [rel_x/100, rel_y/100, speed/30, cos(heading), sin(heading)]
    for agent in world_state.agents[:10]:
        rel_x = (agent.position_m[0] - ego.position_m[0]) / 100.0
        rel_y = (agent.position_m[1] - ego.position_m[1]) / 100.0
        agent_speed = np.linalg.norm(agent.velocity_mps)
        features.extend([
            rel_x, rel_y,
            agent_speed / 30.0,
            np.cos(agent.heading_rad),
            np.sin(agent.heading_rad),
        ])

    # Pad to state_dim
    return torch.tensor(features[:state_dim])
```

**特征总结**:
- Ego: 5维 (归一化位置、速度、朝向)
- Goal: 2维 (相对距离，归一化)
- Agents: 10个 × 5维 = 50维 (相对位置、速度、朝向，归一化)
- **总维度**: 5 + 2 + 50 = 57维

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:276
initial_state = state  # 直接使用WorldState

# WorldStateEncoder (trajectory_encoder.py:79-148)
def forward(self, world_state_batch: List[WorldState]) -> torch.Tensor:
    # Ego features: [pos_x, pos_y, vel_x, vel_y, yaw]
    ego_feat = torch.tensor([
        ws.ego.position_m[0],      # 绝对坐标，无归一化
        ws.ego.position_m[1],
        ws.ego.velocity_mps[0],
        ws.ego.velocity_mps[1],
        ws.ego.yaw_rad
    ])

    # Agent features: [pos_x, pos_y, vel_x, vel_y, heading, type]
    agent_feat = torch.tensor([
        agent.position_m[0],       # 绝对坐标，无归一化
        agent.position_m[1],
        agent.velocity_mps[0],
        agent.velocity_mps[1],
        agent.heading_rad,
        agent_type_encoding
    ])

    # Self-attention聚合
    agent_aggregated, _ = self.attention(ego_query, agent_features, ...)

    return self.fusion([ego_features, agent_aggregated])
```

**特征总结**:
- Ego: 5维 (绝对坐标、速度向量、朝向)
- Agents: 可变数量 × 6维 (绝对坐标、速度向量、朝向、类型)
- **无Goal信息**
- **无归一化**
- **无相对距离**

**关键差异**:
| 特性 | PPO | Rainbow DQN |
|------|-----|-------------|
| 坐标系 | 相对坐标 (ego-centric) | 绝对坐标 |
| 归一化 | ✓ (/100, /30) | ✗ |
| Goal信息 | ✓ 相对距离 | ✗ 无 |
| Agent数量 | 固定10个 | 可变 (attention) |
| 距离编码 | 手动计算相对距离 | 网络学习 |

**潜在问题**:
1. ❌ **位置不变性缺失**: 相同相对配置在不同绝对位置产生不同特征
2. ❌ **缺少目标导向**: 无法知道目标在哪里
3. ❌ **数值范围大**: 绝对坐标可能[-100, 100]，影响训练稳定性
4. ✅ **优势**: 可变agent数量，更灵活

---

## 4. 动作选择 (Action Selection)

### PPO
```python
# run_ppo_carla.py:345-356
state_features = self.planner._extract_state_features(state)
with torch.no_grad():
    logits, value = self.planner.network(state_features)
    action_probs = F.softmax(logits, dim=-1)
    action_dist = Categorical(probs=action_probs)
    action_idx = action_dist.sample().item()
    log_prob = action_dist.log_prob(torch.tensor(action_idx))

# 关键：设置planner内部变量
self.planner._last_action_idx = action_idx
self.planner._last_log_prob = log_prob
self.planner._last_value = value
```

**探索机制**: Categorical采样 + Entropy bonus

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:278-286
# 保存初始state
initial_state = state

# 探索：重置Noisy Nets噪声
self.planner.q_network.reset_noise()
self.planner.q_network.train()

with torch.no_grad():
    q_dist, q_values = self.planner.q_network([state])
    action_idx = q_values.argmax(dim=1).item()
```

**探索机制**: Noisy Nets (参数噪声)

**状态**: ✅ 都有探索机制，但方式不同
- PPO: 策略网络输出概率分布，采样
- Rainbow DQN: Q-network参数噪声，argmax

**问题检查**:
- ✅ Rainbow DQN调用了`reset_noise()`启用探索
- ✅ 设置了`train()`模式

---

## 5. 轨迹执行 (Trajectory Execution)

### PPO
```python
# run_ppo_carla.py:387-443
for step in range(max_steps):
    control = self._trajectory_to_control(state, selected_trajectory, step)
    step_result = self.env.step(control)

    # 不调用planner.update()

    # 累积reward
    episode_reward += step_result.reward

    # 获取OBB检测结果
    step_near_miss = step_result.info.get('near_miss', False)
    obb_min_dist = step_result.info.get('min_distance_to_agents', float('inf'))

    # 记录距离
    step_min_distances.append(current_min_dist)
    step_obb_distances.append(obb_min_dist)
    step_near_miss_flags.append(step_near_miss)

    # 累积reward breakdown
    if 'reward_breakdown' in step_result.info:
        for comp_name, comp_data in step_result.info['reward_breakdown'].items():
            reward_breakdown_accum[comp_name]['raw'] += comp_data['raw']
            reward_breakdown_accum[comp_name]['weighted'] += comp_data['weighted']

    state = step_result.observation

    if step_result.terminated or step_result.truncated:
        break
```

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:318-376
for step in range(max_steps):
    control = self._trajectory_to_control(state, selected_trajectory, step)
    step_result = self.env.step(control)

    # 不调用planner.update()

    # 获取OBB检测结果
    step_near_miss = step_result.info.get('near_miss', False)
    obb_min_dist = step_result.info.get('min_distance_to_agents', float('inf'))

    # 记录距离
    step_min_distances.append(current_min_dist)
    step_obb_distances.append(obb_min_dist)
    step_near_miss_flags.append(step_near_miss)

    # 累积reward breakdown
    if 'reward_breakdown' in step_result.info:
        for comp_name, comp_data in step_result.info['reward_breakdown'].items():
            reward_breakdown_accum[comp_name]['raw'] += comp_data['raw']
            reward_breakdown_accum[comp_name]['weighted'] += comp_data['weighted']

    # 累积reward
    episode_reward += step_result.reward

    state = step_result.observation

    if step_result.terminated or step_result.truncated:
        break
```

**状态**: ✅ 完全一致
- 都不在循环内调用update()
- 都使用OBB检测
- 都累积reward breakdown

---

## 6. Episode结束存储 (Episode-level Storage)

### PPO
```python
# run_ppo_carla.py:449-457
if self.planner._last_log_prob is not None:
    self.planner.rollout_buffer.push(
        state=state_features,          # 初始状态特征 (tensor)
        action=action_idx,             # 轨迹索引
        reward=episode_reward,         # 总episode reward
        value=self.planner._last_value,
        log_prob=self.planner._last_log_prob,
        done=True,
    )
```

**存储数据**:
- state: 初始状态的特征tensor (57维)
- action: 轨迹索引
- reward: episode总reward
- 附加: value, log_prob (PPO需要)

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:384-390
self.planner.replay_buffer.push(
    state=initial_state,        # 初始WorldState对象
    action=action_idx,          # 轨迹索引
    reward=episode_reward,      # 总episode reward
    next_state=final_state,     # 最终WorldState对象
    done=True
)
```

**存储数据**:
- state: 初始WorldState对象
- action: 轨迹索引
- reward: episode总reward
- next_state: 最终WorldState对象
- done: True

**关键差异**:
| 项目 | PPO | Rainbow DQN |
|------|-----|-------------|
| state类型 | Tensor (特征) | WorldState对象 |
| next_state | 无 (on-policy不需要) | WorldState对象 |
| reward | episode总和 | episode总和 |
| 附加信息 | value, log_prob | 无 |

**状态**: ✅ 符合各自算法要求
- PPO是on-policy，只需当前state的特征
- Rainbow DQN是off-policy，需要(s, a, r, s', done)五元组

**潜在问题检查**:
- ✅ 都存储episode-level transition (每个episode一条记录)
- ✅ 都存储总episode reward
- ✅ Rainbow DQN的WorldState对象可以被encoder处理
- ❓ **疑问**: Rainbow DQN的initial_state和final_state在feature上差异大吗？
  - initial_state: episode开始时的WorldState
  - final_state: 执行50步后的WorldState
  - 这两个状态的相对关系（ego到agents的距离）可能类似，但绝对位置完全不同
  - 如果网络学习的是绝对位置，可能无法泛化

---

## 7. 训练时机 (Training Trigger)

### PPO
```python
# run_ppo_carla.py:459-468
metrics = None
buffer_len = len(self.planner.rollout_buffer)
if buffer_len >= self.planner.config.batch_size:
    print(f"  🔄 PPO更新! buffer={buffer_len}, batch_size={self.planner.config.batch_size}")
    metrics = self.planner._ppo_update()
    if metrics:
        print(f"     policy_loss={metrics.get('policy_loss', 0):.4f}, "
              f"value_loss={metrics.get('value_loss', 0):.4f}, "
              f"entropy={metrics.get('entropy', 0):.4f}")
```

**触发条件**: `buffer_len >= batch_size`
**训练方法**: `_ppo_update()`

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:393-402
metrics = None
buffer_len = len(self.planner.replay_buffer)
if buffer_len >= self.planner.config.training.batch_size:
    if self.verbose:
        print(f"  🔄 Rainbow DQN更新! buffer={buffer_len}")
    if hasattr(self.planner, '_train_step'):
        metrics = self.planner._train_step()
```

**触发条件**: `buffer_len >= batch_size`
**训练方法**: `_train_step()`

**状态**: ✅ 一致

**问题检查**:
- ✅ 有hasattr检查（防御性编程）
- ⚠️ **缺少metrics打印**: PPO打印loss信息，Rainbow DQN没打印

---

## 8. 训练逻辑 (Training Logic)

### PPO (_ppo_update)
```python
# planner.py:241-363
def _ppo_update(self) -> Dict[str, float]:
    # 1. 从buffer获取所有数据
    states, actions, rewards, values, log_probs, advantages = self.rollout_buffer.get()

    # 2. 计算returns (GAE或简单累积)
    returns = advantages + values

    # 3. 归一化advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 4. 多轮训练 (K epochs)
    for epoch in range(self.config.ppo_epochs):
        # Mini-batch训练
        for batch_idx in range(num_mini_batches):
            # 计算新的log_prob和value
            new_logits, new_values = self.network(batch_states)
            new_log_probs = new_dist.log_prob(batch_actions)

            # PPO clip loss
            ratio = (new_log_probs - batch_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1-eps, 1+eps) * batch_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values, batch_returns)

            # Entropy bonus
            entropy = new_dist.entropy().mean()

            # Total loss
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

            # 更新
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()

    # 5. 清空buffer
    self.rollout_buffer.clear()

    return metrics
```

**关键特性**:
- On-policy: 每次更新后清空buffer
- 多轮训练: K个epochs
- Mini-batch: 分批训练
- PPO特有: ratio clipping, entropy bonus

### Rainbow DQN (_train_step)
```python
# planner.py:293-373
def _train_step(self) -> UpdateMetrics:
    # 1. 检查是否准备好训练
    if len(self.replay_buffer) < batch_size:
        return UpdateMetrics(...)

    # 2. 采样batch (prioritized)
    batch, indices, weights = self.replay_buffer.sample(batch_size)

    # 3. 准备数据
    states = [t.state for t in batch]  # List[WorldState]
    actions = torch.LongTensor([t.action for t in batch])
    rewards = torch.FloatTensor([t.reward for t in batch])
    next_states = [t.next_state for t in batch]
    dones = torch.FloatTensor([float(t.done) for t in batch])
    weights_tensor = torch.FloatTensor(weights)

    # 4. 当前Q分布
    self.q_network.train()
    q_dist, _ = self.q_network(states)  # (batch, actions, atoms)
    q_dist = q_dist[range(len(batch)), actions, :]  # 选择实际action的分布

    # 5. 目标Q分布 (Double DQN + C51)
    with torch.no_grad():
        # Double DQN: online network选择action
        _, next_q_values = self.q_network(next_states)
        next_actions = next_q_values.argmax(dim=1)

        # Target network评估
        next_q_dist, _ = self.target_network(next_states)
        next_q_dist = next_q_dist[range(len(batch)), next_actions, :]

        # C51 projection
        target_dist = self._project_distribution(rewards, next_q_dist, dones)

    # 6. 计算loss (KL divergence)
    log_q_dist = q_dist.log()
    loss_elementwise = -(target_dist * log_q_dist).sum(dim=1)
    loss = (weights_tensor * loss_elementwise).mean()

    # 7. 更新priorities
    td_errors = loss_elementwise.detach().cpu().numpy()
    self.replay_buffer.update_priorities(indices, td_errors)

    # 8. 反向传播
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(parameters, gradient_clip)
    optimizer.step()

    # 9. 更新target network (软更新或周期性硬更新)
    if update_count % target_update_freq == 0:
        target_network.load_state_dict(q_network.state_dict())

    return UpdateMetrics(...)
```

**关键特性**:
- Off-policy: buffer不清空，持续积累
- 单次训练: 每次调用只训练一个batch
- Prioritized sampling: 根据TD error采样
- Rainbow特有: distributional RL (C51), Double DQN, target network

**状态**: ✅ 符合各自算法设计

**潜在问题**:
- ⚠️ **训练频率**: Rainbow DQN每次只训练一个batch，可能学习较慢
  - PPO: buffer满时，多轮训练直到清空
  - Rainbow DQN: buffer满时，训练一个batch，buffer继续保留
  - 建议: 考虑每次训练多个batch，或增加训练频率

---

## 9. 日志记录 (Logging)

### PPO
```python
# run_ppo_carla.py:509-563
# Reward breakdown写入文件
with open(self.reward_log_path, 'a') as f:
    f.write(f"Episode {episode_id}\n")
    f.write(f"  Selected Action: {action_idx}\n")
    f.write(f"  Total Reward: {episode_reward:.4f}\n")
    f.write("  Reward Breakdown:\n")
    for comp_name, comp_data in reward_breakdown_accum.items():
        f.write(f"  {comp_name} {comp_data['weight']} {comp_data['raw']} {comp_data['weighted']}\n")

    # Step-by-step距离
    f.write("  Step-by-Step Distance Analysis:\n")
    for step_idx in range(len(step_min_distances)):
        f.write(f"  {step_idx} {center_dist} {obb_dist} {near_miss_flag}\n")

# Episode summary CSV
with open(self.summary_log_path, 'a') as f:
    f.write(f"{episode_id},{episode_reward:.2f},{collision},{near_miss},{steps},{action_idx},{outcome}\n")
```

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:443-474
# Reward breakdown写入文件
with open(self.reward_log_path, 'a') as f:
    f.write(f"Episode {episode_id}\n")
    f.write(f"  Selected Action: {action_idx}\n")
    f.write(f"  Total Reward: {episode_reward:.4f}\n")
    f.write("  Reward Breakdown:\n")
    for comp_name, comp_data in reward_breakdown_accum.items():
        f.write(f"  {comp_name} {comp_data['weight']} {comp_data['raw']} {comp_data['weighted']}\n")

    # Step-by-step距离
    f.write("  Step-by-Step Distance Analysis:\n")
    for step_idx in range(len(step_min_distances)):
        f.write(f"  {step_idx} {center_dist} {obb_dist} {near_miss_flag}\n")

# Episode summary CSV
with open(self.summary_log_path, 'a') as f:
    f.write(f"{episode_id},{episode_reward:.2f},{collision},{near_miss},{steps},{action_idx},{outcome}\n")
```

**状态**: ✅ 完全一致

---

## 10. 输出目录结构

### PPO
```python
# run_ppo_carla.py:830-833
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"{args.scenario}_{timestamp}"
output_dir = Path(args.output_dir) / run_name
log_dir = (Path(args.log_dir) / run_name) if TENSORBOARD_AVAILABLE else None
```

### Rainbow DQN
```python
# run_rainbow_dqn_carla.py:693-701
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"{args.scenario}_{timestamp}"
output_dir = Path(args.output_dir) / run_name
log_dir = (Path(args.log_dir) / run_name) if TENSORBOARD_AVAILABLE else None
```

**状态**: ✅ 完全一致

---

## 总结

### ✅ 已对齐的部分
1. 环境重置流程
2. 轨迹生成逻辑
3. 轨迹执行（使用OBB检测，累积reward breakdown）
4. Episode-level transition存储
5. 训练时机触发
6. 日志记录（reward breakdown, step-by-step距离）
7. 输出目录结构

### ❌ 关键差异（设计理念不同，无需修改）
1. **特征提取方式**:
   - PPO: 手动提取，相对距离，归一化
   - Rainbow DQN: 神经网络编码，绝对坐标

2. **探索机制**:
   - PPO: Categorical采样 + Entropy bonus
   - Rainbow DQN: Noisy Nets参数噪声

3. **训练范式**:
   - PPO: On-policy，buffer满时多轮训练后清空
   - Rainbow DQN: Off-policy，buffer持续积累，每次训练一个batch

### ⚠️ 潜在问题和建议

#### 问题1: WorldStateEncoder缺少关键特征
**现象**: Rainbow DQN使用绝对坐标，无相对距离、Goal信息、归一化

**影响**:
- 位置不变性缺失: (ego at (0,0), agent at (10,0)) 和 (ego at (100,0), agent at (110,0)) 产生不同特征
- 缺少目标导向: 网络不知道goal在哪里
- 数值范围大: 可能影响训练稳定性

**建议**:
- 先测试当前版本效果
- 如果性能不佳，考虑改进WorldStateEncoder:
  - 添加相对距离计算
  - 添加goal信息
  - 添加特征归一化
  - 需要修改network.py和trajectory_encoder.py

#### 问题2: 训练频率可能过低
**现象**: Rainbow DQN每次只训练一个batch，PPO会多轮训练

**影响**:
- 学习速度可能较慢
- Buffer积累大量数据但利用率低

**建议**:
- 考虑每次调用_train_step时训练多个batch:
  ```python
  if buffer_len >= batch_size:
      for _ in range(train_iterations_per_update):
          metrics = self.planner._train_step()
  ```
- 或者增加训练频率（每N个episode训练一次，但每次训练多个iterations）

#### 问题3: 缺少训练metrics打印
**现象**: PPO打印loss信息，Rainbow DQN没有

**建议**: 在run_rainbow_dqn_carla.py中添加:
```python
if metrics:
    print(f"     loss={metrics.get('loss', 0):.4f}, "
          f"q_value={metrics.get('q_value', 0):.4f}, "
          f"td_error={metrics.custom.get('td_error_mean', 0):.4f}")
```

#### 问题4: 无warmup检查
**现象**: _train_step有warmup检查，但run_episode没有相应提示

**建议**: 在训练时添加warmup提示:
```python
if buffer_len >= batch_size:
    if self._step_count < warmup_steps:
        print(f"  ⏳ Warmup阶段: {self._step_count}/{warmup_steps}")
    else:
        print(f"  🔄 Rainbow DQN更新!")
        metrics = self.planner._train_step()
```

---

## 需要立即修复的问题

### 无（已完成所有核心修复）

当前Rainbow DQN实现在算法层面是正确的，与PPO的差异主要来自设计理念不同（on-policy vs off-policy，手动特征 vs 神经网络编码）。

---

## 建议的后续优化（可选）

1. **添加训练metrics打印** - 方便监控训练过程
2. **增加训练频率或每次训练多个batch** - 提高学习效率
3. **测试当前版本** - 在s4等场景测试效果
4. **如果效果不佳，改进WorldStateEncoder** - 添加相对距离、goal、归一化

---

## 验证清单

运行: `python examples/run_rainbow_dqn_carla.py --scenario s4 --episodes 10 --max-steps 50`

检查:
- [ ] 每个episode只存储1个transition
- [ ] Episode reward是累积值（如-5.0）
- [ ] Near-miss使用OBB距离
- [ ] 输出到`outputs/rainbow_dqn_carla/s4_YYYYMMDD_HHMMSS/`
- [ ] 有探索行为（不同episode选不同轨迹）
- [ ] 日志文件包含reward breakdown和距离分析
- [ ] Buffer满时触发训练
- [ ] 训练loss合理下降
