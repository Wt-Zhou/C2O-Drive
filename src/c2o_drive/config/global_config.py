"""
全局配置管理模块

集中管理所有关键参数,确保系统一致性。
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TimeConfig:
    """时间相关配置"""
    dt: float = 1                     # 时间步长（秒）
    default_horizon: int = 8            # 默认预测时间步数（统一为10）
    simulation_fps: int = 10             # 仿真帧率
    
    @property
    def horizon_seconds(self) -> float:
        """计算horizon对应的实际秒数"""
        return self.default_horizon * self.dt
    
    def steps_to_seconds(self, steps: int) -> float:
        """将时间步数转换为秒数"""
        return steps * self.dt
    
    def seconds_to_steps(self, seconds: float) -> int:
        """将秒数转换为时间步数"""
        return int(seconds / self.dt)


@dataclass 
class SamplingConfig:
    """采样相关配置"""
    reachable_set_samples: int = 5000     # 可达集采样数量
    reachable_set_samples_legacy: int = 5000  # 兼容旧版本的采样数量
    q_value_samples: int = 50           # Q值计算采样数量
    trajectory_samples: int = 50        # 轨迹生成采样数量


@dataclass
class GridConfig:
    """网格相关配置"""
    grid_size_m: float = 100.0           # 网格物理尺寸（米）
    grid_resolution: int = 200          # 网格分辨率（N×N）
    cell_size_m: float = 0.5            # 网格单元尺寸（米）

    # 网格边界（基于grid_size_m计算，可显式覆盖）
    x_min: float = -50.0                # X轴最小值
    x_max: float = 50.0                 # X轴最大值
    y_min: float = -50.0                # Y轴最小值
    y_max: float = 50.0                 # Y轴最大值


@dataclass
class DirichletConfig:
    """Dirichlet分布相关配置"""
    alpha_in: float = 1000.0              # 可达集内的先验伪计数
    alpha_out: float = 1e-6             # 可达集外的先验伪计数
    delta: float = 0.05                 # 置信水平参数
    cK: float = 1.0                     # 置信半径校准常数
    learning_rate: float = 1.0          # 学习率


@dataclass
class MatchingConfig:
    """MDP状态-动作匹配相关配置"""
    # 轨迹匹配阈值
    ego_state_threshold: float = 5.0        # 自车状态匹配阈值（米）
    agents_state_threshold: float = 5.0     # 环境智能体状态匹配阈值（米）
    ego_action_threshold: float = 5.0       # 自车动作匹配阈值（米）

    # 轨迹缓冲区索引精度
    spatial_resolution: float = 5.0         # 空间索引分辨率（米）
    ego_action_resolution: float = 5.0      # 动作索引分辨率（米）

    # 数据增强
    trajectory_storage_multiplier: int = 1  # 轨迹存储倍数（1=不重复,10=每次观测存储10次）


@dataclass
class RewardConfig:
    """奖励函数相关配置"""
    # 碰撞相关
    collision_penalty: float = -50.0   # 碰撞惩罚
    collision_threshold: float = 0.01    # 碰撞阈值

    # 碰撞检测优化
    collision_check_cell_radius: int = 6  # Cell剪枝半径（cell数,约为半径米数×2）

    # 权重参数
    comfort_weight: float = 1.0         # 舒适性权重
    efficiency_weight: float = 1.0      # 效率权重
    safety_weight: float = 10.0          # 安全权重

    # 舒适性参数
    max_accel_penalty: float = -1.0     # 最大加速度惩罚
    max_jerk_penalty: float = -2.0      # 最大急动惩罚
    acceleration_penalty_weight: float = 0.1  # 加速度惩罚权重
    jerk_penalty_weight: float = 0.05   # 急动惩罚权重
    max_comfortable_accel: float = 1.0  # 舒适最大加速度（m/s²）

    # 效率参数
    speed_reward_weight: float = 1.0    # 速度奖励权重
    target_speed: float = 5.0           # 目标速度（m/s）
    progress_reward_weight: float = 1.0 # 前进奖励权重

    # 安全距离参数
    safe_distance: float = 3.0          # 安全距离（m）
    distance_penalty_weight: float = 0.0  # 距离惩罚权重

    # 中心线偏移参数
    centerline_offset_penalty_weight: float = 0 # 中心线偏移惩罚权重
    max_deviation: float = 2.0          # 最大可接受偏移（m）

    # 时间相关参数
    time_penalty: float = -0.1          # 每时间步惩罚（鼓励快速完成任务）
    gamma: float = 0.95                 # 时序折扣因子（用于C2OSR碰撞概率计算）


@dataclass
class VisualizationConfig:
    """可视化相关配置"""
    figure_size: tuple = (8, 8)         # 图片尺寸
    dpi: int = 150                      # 图片分辨率
    gif_fps: int = 2                    # GIF帧率
    font_size: int = 12                 # 字体大小

    # 可达集可视化
    timestep_alphas: list = None        # 不同时间步的透明度
    timestep_linewidths: list = None    # 不同时间步的线宽
    agent_colors: list = None           # 智能体颜色

    # 日志输出级别
    verbose_level: int = 1              # 0=minimal, 1=normal, 2=debug

    def __post_init__(self):
        if self.timestep_alphas is None:
            self.timestep_alphas = [1.0, 0.8, 0.6, 0.4, 0.3]
        if self.timestep_linewidths is None:
            self.timestep_linewidths = [2.0, 1.8, 1.5, 1.2, 1.0]
        if self.agent_colors is None:
            self.agent_colors = ['cyan', 'magenta', 'lime', 'orange', 'red', 'blue']


@dataclass
class C2OSRConfig:
    """C2OSR算法相关配置（从sim.yaml迁移）"""
    # Dirichlet Process参数
    dp_alpha: float = 0.5           # DP 浓度
    dp_c: float = 20.0              # Dirichlet(c·z) 总强度
    dp_eta: float = 1.0             # 分数计数学习率
    dp_add_thresh: float = 0.15     # 新原子触发阈值（责任）
    
    # 风险参数
    risk_mode: str = "union"        # union|independent
    risk_lambda: float = 5.0        # 风险权重
    risk_epsilon: float = 0.2       # 机会约束阈值（每条轨迹）
    gamma: float = 0.9             # 折扣因子

    # 采样参数
    samples: int = 32               # 后验采样数 S

    # Q值选择策略
    q_selection_percentile: float = 0.05  # Q值选择百分位数（0.0=最小值, 0.1=10%分位, 0.5=中位数）


@dataclass
class BaselineConfig:
    """基线方法配置（从baselines.yaml迁移）"""
    # SAC配置
    sac_enabled: bool = False
    sac_seed: int = 0

    # SAC网络配置
    sac_state_dim: int = 128              # 状态特征维度
    sac_action_dim: int = 2               # 动作维度 [lateral_offset, target_speed]
    sac_max_action: float = 1.0           # 动作范围 [-1, 1] (会被rescale到实际范围)
    sac_hidden_dims: tuple = (256, 256)   # 隐藏层维度

    # SAC动作空间范围 (对齐lattice planner)
    sac_lateral_offset_range: tuple = (-5.0, 5.0)   # 横向偏移范围（米）
    sac_target_speed_range: tuple = (0.0, 10.0)     # 目标速度范围（m/s）

    # SAC训练超参数
    sac_learning_rate: float = 3e-4       # Actor和Critic学习率
    sac_gamma: float = 0.99               # 折扣因子
    sac_tau: float = 0.005                # 软更新系数
    sac_alpha: float = 0.2                # 初始熵系数
    sac_auto_entropy: bool = True         # 自动调节熵系数

    # SAC经验回放配置
    sac_batch_size: int = 256             # 训练批次大小
    sac_buffer_size: int = 1000000        # 回放缓冲区大小
    sac_updates_per_episode: int = 50     # 每个episode结束后的训练次数

    # SAC训练配置
    sac_num_episodes: int = 1000          # 总训练episodes
    sac_max_steps_per_episode: int = 500  # 每个episode最大步数
    sac_device: str = "cuda"              # 训练设备 (cuda/cpu)

    # SAC保存与评估配置
    sac_save_interval: int = 50           # 模型保存间隔（episodes）
    sac_eval_interval: int = 10           # 评估间隔（episodes）
    sac_eval_episodes: int = 5            # 每次评估的episodes数
    sac_checkpoint_dir: str = "checkpoints/sac_carla"  # 检查点保存目录
    sac_log_dir: str = "logs/sac_carla"   # TensorBoard日志目录

    # PPO配置
    ppo_enabled: bool = False
    ppo_seed: int = 0

    # PPO网络配置
    ppo_state_dim: int = 128              # 状态特征维度
    ppo_hidden_dims: tuple = (256, 256)   # 隐藏层维度

    # PPO训练超参数
    ppo_learning_rate: float = 3e-4       # 学习率
    ppo_gamma: float = 0.99               # 折扣因子
    ppo_gae_lambda: float = 0.95          # GAE lambda参数
    ppo_clip_epsilon: float = 0.2         # PPO clipping参数
    ppo_clip_grad_norm: float = 0.5       # 梯度裁剪范数
    ppo_value_loss_coef: float = 0.5      # Value损失系数
    ppo_entropy_coef: float = 0.01        # 熵正则化系数

    # PPO训练配置
    ppo_n_epochs: int = 10                # PPO更新轮数
    ppo_batch_size: int = 64              # Mini-batch大小
    ppo_buffer_size: int = 2048           # Rollout buffer容量
    ppo_num_episodes: int = 1000          # 总训练episodes
    ppo_max_steps_per_episode: int = 100  # 每个episode最大步数
    ppo_use_gae: bool = True              # 使用GAE
    ppo_normalize_advantage: bool = True  # 归一化advantage
    ppo_device: str = "cpu"               # 训练设备 (cuda/cpu)

    # PPO保存与评估配置
    ppo_save_interval: int = 50           # 模型保存间隔（episodes）
    ppo_eval_interval: int = 10           # 评估间隔（episodes）
    ppo_eval_episodes: int = 5            # 每次评估的episodes数
    ppo_checkpoint_dir: str = "checkpoints/ppo_carla"  # 检查点保存目录
    ppo_log_dir: str = "logs/ppo_carla"   # TensorBoard日志目录

    # Offline CQL配置
    offline_cql_enabled: bool = False
    offline_cql_dataset: str = ""

    # Shielded配置
    shielded_enabled: bool = False
    shielded_epsilon: float = 0.1


@dataclass
class LatticeConfig:
    """Lattice轨迹配置（基于reference path的采样）"""
    lateral_offsets: list = None    # 横向偏移量（米）
    speed_variations: list = None   # 速度变化（m/s）
    dt: float = 1.0                 # 时间步长（秒）
    horizon: int = 10               # 预测时间步数

    def __post_init__(self):
        if self.lateral_offsets is None:
            self.lateral_offsets = [-5.0, -2.5, 0.0, 2.5, 5.0]  # C2OSR标准配置
        if self.speed_variations is None:
            self.speed_variations = [1.0, 3.0,6.0]  # C2OSR标准配置

    @property
    def num_trajectories(self) -> int:
        """动态计算轨迹数量（即action_dim）

        Returns:
            int: len(lateral_offsets) × len(speed_variations)

        Examples:
            - lateral_offsets=[-3,-2,0,2,3], speed_variations=[4.0] → num_trajectories=5
            - lateral_offsets=[-3,-2,0,2,3], speed_variations=[4,6,8] → num_trajectories=15
        """
        return len(self.lateral_offsets) * len(self.speed_variations)


@dataclass
class ScenarioConfig:
    """场景配置（从scenarios.yaml迁移）"""
    town: str = "Town03"
    weather: str = "ClearNoon"
    seed: int = 42


@dataclass
class CarlaConfig:
    """CARLA仿真环境配置"""
    # 连接配置
    host: str = "localhost"
    port: int = 2000
    timeout: float = 10.0

    # 地图和天气
    town: str = "Town03"
    weather: str = "ClearNoon"

    # 仿真配置
    dt: float = 0.1
    no_rendering: bool = False
    synchronous_mode: bool = True

    # 场景配置
    num_vehicles: int = 10
    num_pedestrians: int = 5
    autopilot: bool = False

    # 传感器配置
    enable_collision_sensor: bool = True
    enable_rgb_camera: bool = False
    enable_lidar: bool = False

    # 相机视角配置
    camera_height: float = 60.0
    camera_pitch: float = -90.0  # -90度为俯视
    camera_follow_ego: bool = True

    # Episode配置
    max_episode_steps: int = 500

    # 交通管理器配置
    tm_port: int = 8000
    tm_seed: int = 0


@dataclass
class SafetyConfig:
    """安全和预测评估相关配置"""
    # Near miss detection
    near_miss_threshold_m: float = 3.0       # Near miss距离阈值（米）

    # Confidence set configuration
    confidence_level: float = 0.95           # Confidence set置信水平（95%）
    confidence_set_samples: int = 100        # 基础采样数量（用于叠加计算confidence set）

    # 动态采样配置
    adaptive_sampling: bool = True           # 是否启用动态采样
    min_samples_per_cell: int = 5            # 每个cell最少采样次数
    max_samples: int = 5000                  # 最大采样数上限（防止计算爆炸）


@dataclass
class AgentTrajectoryConfig:
    """智能体轨迹生成配置"""
    mode: str = "stochastic"  # 轨迹生成模式: "stochastic" | "straight" | "stationary"
    random_seed: int = 42  # 轨迹生成随机种子（注：stochastic模式不使用此种子以保证随机性）


@dataclass
class GlobalConfig:
    """全局配置容器"""
    time: TimeConfig = None
    sampling: SamplingConfig = None
    grid: GridConfig = None
    dirichlet: DirichletConfig = None
    matching: MatchingConfig = None
    reward: RewardConfig = None
    visualization: VisualizationConfig = None
    c2osr: C2OSRConfig = None
    baseline: BaselineConfig = None
    lattice: LatticeConfig = None
    scenario: ScenarioConfig = None
    agent_trajectory: AgentTrajectoryConfig = None
    carla: CarlaConfig = None  # CARLA仿真环境配置
    safety: SafetyConfig = None  # 安全和预测评估配置

    # 系统配置
    random_seed: int = 2025
    debug_mode: bool = False
    log_level: str = "INFO"

    def __post_init__(self):
        if self.time is None:
            self.time = TimeConfig()
        if self.sampling is None:
            self.sampling = SamplingConfig()
        if self.grid is None:
            self.grid = GridConfig()
        if self.dirichlet is None:
            self.dirichlet = DirichletConfig()
        if self.matching is None:
            self.matching = MatchingConfig()
        if self.reward is None:
            self.reward = RewardConfig()
        if self.visualization is None:
            self.visualization = VisualizationConfig()
        if self.c2osr is None:
            self.c2osr = C2OSRConfig()
        if self.baseline is None:
            self.baseline = BaselineConfig()
        if self.lattice is None:
            self.lattice = LatticeConfig()
        if self.scenario is None:
            self.scenario = ScenarioConfig()
        if self.agent_trajectory is None:
            self.agent_trajectory = AgentTrajectoryConfig()
        if self.carla is None:
            self.carla = CarlaConfig()
        if self.safety is None:
            self.safety = SafetyConfig()

    def update_dt(self, new_dt: float):
        """更新时间步长并保持其他参数一致性"""
        old_dt = self.time.dt
        self.time.dt = new_dt
        
        # 调整相关参数以保持物理一致性
        scale_factor = new_dt / old_dt
        
        print(f"更新时间步长: {old_dt}s → {new_dt}s (比例: {scale_factor:.2f})")
        print(f"Horizon实际时间: {self.time.horizon_seconds:.2f}s")
    
    def update_horizon(self, new_horizon: int):
        """更新预测时间步数"""
        old_horizon = self.time.default_horizon
        self.time.default_horizon = new_horizon
        
        print(f"更新预测时间步数: {old_horizon} → {new_horizon}")
        print(f"Horizon实际时间: {self.time.horizon_seconds:.2f}s")
    
    def print_summary(self):
        """打印配置摘要"""
        print("=== 全局配置摘要 ===")
        print(f"时间步长: {self.time.dt}s")
        print(f"预测时间步数: {self.time.default_horizon}")
        print(f"预测实际时间: {self.time.horizon_seconds:.2f}s")
        print(f"可达集采样数: {self.sampling.reachable_set_samples}")
        print(f"Q值采样数: {self.sampling.q_value_samples}")
        print(f"网格尺寸: {self.grid.grid_size_m}m × {self.grid.grid_resolution}")
        print(f"随机种子: {self.random_seed}")


# 全局配置实例
_global_config = None

def get_global_config() -> GlobalConfig:
    """获取全局配置实例"""
    global _global_config
    if _global_config is None:
        _global_config = GlobalConfig()
    return _global_config

def set_global_config(config: GlobalConfig):
    """设置全局配置实例"""
    global _global_config
    _global_config = config

def update_dt(new_dt: float):
    """快捷函数：更新全局时间步长"""
    get_global_config().update_dt(new_dt)

def update_horizon(new_horizon: int):
    """快捷函数：更新全局预测时间步数"""
    get_global_config().update_horizon(new_horizon)

def get_dt() -> float:
    """快捷函数：获取当前时间步长"""
    return get_global_config().time.dt

def get_horizon() -> int:
    """快捷函数：获取当前预测时间步数"""
    return get_global_config().time.default_horizon

def get_horizon_seconds() -> float:
    """快捷函数：获取预测时间的秒数"""
    return get_global_config().time.horizon_seconds


# 从环境变量加载配置
def load_config_from_env():
    """从环境变量加载配置"""
    config = get_global_config()
    
    # 时间配置
    if 'C2O_DT' in os.environ:
        config.time.dt = float(os.environ['C2O_DT'])
    if 'C2O_HORIZON' in os.environ:
        config.time.default_horizon = int(os.environ['C2O_HORIZON'])
    
    # 采样配置
    if 'C2O_REACHABLE_SAMPLES' in os.environ:
        config.sampling.reachable_set_samples = int(os.environ['C2O_REACHABLE_SAMPLES'])
    if 'C2O_Q_SAMPLES' in os.environ:
        config.sampling.q_value_samples = int(os.environ['C2O_Q_SAMPLES'])
    
    # 系统配置
    if 'C2O_SEED' in os.environ:
        config.random_seed = int(os.environ['C2O_SEED'])
    if 'C2O_DEBUG' in os.environ:
        config.debug_mode = os.environ['C2O_DEBUG'].lower() == 'true'
    
    return config


# 预设配置模板
class ConfigPresets:
    """预设配置模板"""
    
    @staticmethod
    def fast_testing() -> GlobalConfig:
        """快速测试配置"""
        config = GlobalConfig()
        config.time.dt = 1
        config.time.default_horizon = 8
        config.sampling.reachable_set_samples = 1000
        config.sampling.q_value_samples = 50
        return config
    
    @staticmethod
    def high_precision() -> GlobalConfig:
        """高精度配置"""
        config = GlobalConfig()
        config.time.dt = 0.1
        config.time.default_horizon = 15
        config.sampling.reachable_set_samples = 500
        config.sampling.q_value_samples = 200
        return config
    
    @staticmethod
    def long_horizon() -> GlobalConfig:
        """长期预测配置"""
        config = GlobalConfig()
        config.time.dt = 0.2
        config.time.default_horizon = 25  # 5秒预测
        config.sampling.reachable_set_samples = 300
        config.sampling.q_value_samples = 150
        return config


if __name__ == "__main__":
    # 测试配置系统
    config = get_global_config()
    config.print_summary()
    
    print("\n=== 测试快速配置更新 ===")
    update_dt(0.1)
    update_horizon(20)
    
    print("\n=== 测试预设配置 ===")
    fast_config = ConfigPresets.fast_testing()
    fast_config.print_summary()
