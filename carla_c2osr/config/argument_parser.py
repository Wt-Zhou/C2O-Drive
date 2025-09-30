"""
统一的命令行参数解析器

提供标准化的参数解析，与全局配置系统集成。
"""

import argparse
from typing import Optional, Any
from .global_config import GlobalConfig, ConfigPresets


class ConfigArgumentParser:
    """配置化的命令行参数解析器"""
    
    def __init__(self, description: str = "C2O-Drive 配置化运行器"):
        self.parser = argparse.ArgumentParser(description=description)
        self._setup_arguments()
    
    def _setup_arguments(self):
        """设置所有命令行参数"""
        
        # 基本运行参数
        self.parser.add_argument("--episodes", type=int, default=10, 
                               help="执行episode数")
        self.parser.add_argument("--seed", type=int, default=2025, 
                               help="随机种子")
        self.parser.add_argument("--gif-fps", type=int, default=2, 
                               help="GIF帧率")
        self.parser.add_argument("--ego-mode", 
                               choices=["straight", "fixed-traj"], 
                               default="straight", 
                               help="自车运动模式")
        self.parser.add_argument("--sigma", type=float, default=0.5, 
                               help="软计数核宽度")
        
        # 配置预设（主要配置方式）
        self.parser.add_argument("--config-preset", 
                               choices=["default", "fast", "high-precision", "long-horizon"],
                               default="default", 
                               help="预设配置模板")
        
        # 可选覆盖参数（仅在需要时使用）
        self.parser.add_argument("--dt", type=float, 
                               help="覆盖时间步长（秒）")
        self.parser.add_argument("--horizon", type=int, 
                               help="覆盖预测时间步数")
        self.parser.add_argument("--reachable-samples", type=int, 
                               help="覆盖可达集采样数量")
        self.parser.add_argument("--q-samples", type=int, 
                               help="覆盖Q值采样数量")
        
        # 高级配置参数
        self.parser.add_argument("--grid-size", type=float, 
                               help="覆盖网格尺寸（米）")
        self.parser.add_argument("--alpha-in", type=float, 
                               help="覆盖Dirichlet alpha_in参数")
        self.parser.add_argument("--alpha-out", type=float, 
                               help="覆盖Dirichlet alpha_out参数")
    
    def parse_args(self) -> argparse.Namespace:
        """解析命令行参数"""
        return self.parser.parse_args()
    
    def create_config_from_args(self, args: argparse.Namespace) -> GlobalConfig:
        """从命令行参数创建配置对象"""
        
        # 首先应用预设配置
        if args.config_preset == "fast":
            config = ConfigPresets.fast_testing()
        elif args.config_preset == "high-precision":
            config = ConfigPresets.high_precision()
        elif args.config_preset == "long-horizon":
            config = ConfigPresets.long_horizon()
        else:
            from .global_config import get_global_config
            config = get_global_config()
        
        # 仅在用户明确指定时才覆盖预设配置
        if args.dt is not None:
            config.time.dt = args.dt
        if args.horizon is not None:
            config.time.default_horizon = args.horizon
        if args.reachable_samples is not None:
            config.sampling.reachable_set_samples = args.reachable_samples
        if args.q_samples is not None:
            config.sampling.q_value_samples = args.q_samples
        if args.grid_size is not None:
            config.grid.grid_size_m = args.grid_size
        if args.alpha_in is not None:
            config.dirichlet.alpha_in = args.alpha_in
        if args.alpha_out is not None:
            config.dirichlet.alpha_out = args.alpha_out
        
        # 这些参数总是从命令行获取
        config.random_seed = args.seed
        config.visualization.gif_fps = args.gif_fps
        
        return config
    
    def print_config_summary(self, config: GlobalConfig, args: argparse.Namespace):
        """打印配置摘要"""
        print(f"=== 配置摘要 ===")
        print(f"配置预设: {args.config_preset}")
        print(f"时间步长: {config.time.dt}s, 预测时间: {config.time.horizon_seconds:.1f}s")
        print(f"可达集采样: {config.sampling.reachable_set_samples}, Q值采样: {config.sampling.q_value_samples}")
        print(f"网格尺寸: {config.grid.grid_size_m}m")
        print(f"随机种子: {config.random_seed}")


# 便捷函数
def create_standard_parser(description: str = "C2O-Drive 配置化运行器") -> ConfigArgumentParser:
    """创建标准配置解析器"""
    return ConfigArgumentParser(description)


def parse_config_from_cli(description: str = "C2O-Drive 配置化运行器") -> tuple[GlobalConfig, argparse.Namespace]:
    """一键解析配置和参数"""
    parser = create_standard_parser(description)
    args = parser.parse_args()
    config = parser.create_config_from_args(args)
    return config, args
