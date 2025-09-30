"""
全局配置管理模块

集中管理所有关键参数，确保系统一致性。
"""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class TimeConfig:
    """时间相关配置"""
    dt: float = 0.2                     # 时间步长（秒）
    default_horizon: int = 8             # 默认预测时间步数
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
    reachable_set_samples: int = 200     # 可达集采样数量
    reachable_set_samples_legacy: int = 50  # 兼容旧版本的采样数量
    q_value_samples: int = 100           # Q值计算采样数量
    trajectory_samples: int = 100        # 轨迹生成采样数量


@dataclass
class GridConfig:
    """网格相关配置"""
    grid_size_m: float = 20.0           # 网格物理尺寸（米）
    grid_resolution: int = 200          # 网格分辨率（N×N）
    cell_size_m: float = 0.1            # 网格单元尺寸（米）


@dataclass
class DirichletConfig:
    """Dirichlet分布相关配置"""
    alpha_in: float = 50.0              # 可达集内的先验伪计数
    alpha_out: float = 1e-6             # 可达集外的先验伪计数
    delta: float = 0.05                 # 置信水平参数
    cK: float = 1.0                     # 置信半径校准常数
    learning_rate: float = 1.0          # 学习率


@dataclass
class RewardConfig:
    """奖励函数相关配置"""
    collision_penalty: float = -100.0   # 碰撞惩罚
    comfort_weight: float = 1.0         # 舒适性权重
    efficiency_weight: float = 1.0      # 效率权重
    safety_weight: float = 1.0          # 安全权重
    
    # 舒适性参数
    max_accel_penalty: float = -1.0     # 最大加速度惩罚
    max_jerk_penalty: float = -2.0      # 最大急动惩罚
    
    # 效率参数
    min_speed_reward: float = 1.0       # 最小速度奖励
    progress_reward: float = 2.0        # 前进奖励


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
    gamma: float = 0.99             # 折扣因子
    
    # 采样参数
    samples: int = 32               # 后验采样数 S


@dataclass
class BaselineConfig:
    """基线方法配置（从baselines.yaml迁移）"""
    # SAC配置
    sac_enabled: bool = False
    sac_seed: int = 0
    
    # Offline CQL配置
    offline_cql_enabled: bool = False
    offline_cql_dataset: str = ""
    
    # Shielded配置
    shielded_enabled: bool = False
    shielded_epsilon: float = 0.1


@dataclass
class LatticeConfig:
    """Lattice轨迹配置（从lattice.yaml迁移）"""
    speeds_mps: list = None         # 速度列表
    steers: list = None             # 转向角度列表
    accels_mps2: list = None        # 加速度列表
    num_trajectories: int = 9       # 生成上限
    
    def __post_init__(self):
        if self.speeds_mps is None:
            self.speeds_mps = [5.0, 7.0]
        if self.steers is None:
            self.steers = [-0.2, -0.1, 0.0, 0.1, 0.2]
        if self.accels_mps2 is None:
            self.accels_mps2 = [-1.0, 0.0, 1.0]


@dataclass
class ScenarioConfig:
    """场景配置（从scenarios.yaml迁移）"""
    town: str = "Town03"
    weather: str = "ClearNoon"
    seed: int = 42


@dataclass
class GlobalConfig:
    """全局配置容器"""
    time: TimeConfig = None
    sampling: SamplingConfig = None
    grid: GridConfig = None
    dirichlet: DirichletConfig = None
    reward: RewardConfig = None
    visualization: VisualizationConfig = None
    c2osr: C2OSRConfig = None
    baseline: BaselineConfig = None
    lattice: LatticeConfig = None
    scenario: ScenarioConfig = None
    
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
