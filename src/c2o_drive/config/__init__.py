"""
配置管理模块

提供全局配置管理和便捷的配置访问接口。
"""

from .global_config import (
    GlobalConfig, TimeConfig, SamplingConfig, GridConfig,
    DirichletConfig, MatchingConfig, RewardConfig, VisualizationConfig,
    C2OSRConfig, BaselineConfig, LatticeConfig, ScenarioConfig,
    get_global_config, set_global_config,
    update_dt, update_horizon, get_dt, get_horizon, get_horizon_seconds,
    load_config_from_env, ConfigPresets
)

__all__ = [
    # 配置类
    'GlobalConfig', 'TimeConfig', 'SamplingConfig', 'GridConfig',
    'DirichletConfig', 'MatchingConfig', 'RewardConfig', 'VisualizationConfig',
    'C2OSRConfig', 'BaselineConfig', 'LatticeConfig', 'ScenarioConfig',
    # 配置访问函数
    'get_global_config', 'set_global_config',
    'update_dt', 'update_horizon', 'get_dt', 'get_horizon', 'get_horizon_seconds',
    'load_config_from_env', 'ConfigPresets'
]
