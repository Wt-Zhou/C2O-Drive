"""
配置管理模块

提供全局配置管理和便捷的配置访问接口。
"""

from .global_config import (
    GlobalConfig, TimeConfig, SamplingConfig, GridConfig, 
    DirichletConfig, RewardConfig, VisualizationConfig,
    get_global_config, set_global_config,
    update_dt, update_horizon, get_dt, get_horizon, get_horizon_seconds,
    load_config_from_env, ConfigPresets
)

__all__ = [
    'GlobalConfig', 'TimeConfig', 'SamplingConfig', 'GridConfig',
    'DirichletConfig', 'RewardConfig', 'VisualizationConfig',
    'get_global_config', 'set_global_config',
    'update_dt', 'update_horizon', 'get_dt', 'get_horizon', 'get_horizon_seconds',
    'load_config_from_env', 'ConfigPresets'
]
