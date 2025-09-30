from __future__ import annotations
import argparse
from pathlib import Path
import sys
from typing import Any

# 允许从包目录内直接运行：将仓库根加入 sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.types import AgentState, EgoControl, EgoState, WorldState
from carla_c2osr.agents.c2osr.agent import C2OSRConfig as OldC2OSRConfig, C2OSRDriveAgent
from carla_c2osr.config import get_global_config, set_global_config


# YAML加载函数已废弃，现在使用全局配置系统


def build_agent_from_global_config() -> C2OSRDriveAgent:
    """从全局配置构建agent"""
    config = get_global_config()
    
    # 将全局配置转换为旧的C2OSRConfig格式
    cfg = OldC2OSRConfig(
        grid_size_m=config.grid.grid_size_m,
        grid_cell_m=config.grid.cell_size_m,
        grid_macro=True,  # 使用默认值
        horizon=config.time.default_horizon,
        samples=config.c2osr.samples,
        alpha=config.c2osr.dp_alpha,
        c=config.c2osr.dp_c,
        eta=config.c2osr.dp_eta,
        add_thresh=config.c2osr.dp_add_thresh,
        risk_mode=config.c2osr.risk_mode,
        epsilon=config.c2osr.risk_epsilon,
        gamma=config.c2osr.gamma,
    )
    return C2OSRDriveAgent(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-env", action="store_true")
    parser.add_argument("--dt", type=float, help="时间步长（秒）")
    parser.add_argument("--horizon", type=int, help="预测时间步数")
    parser.add_argument("--samples", type=int, help="采样数量")
    args = parser.parse_args()

    # 使用全局配置系统
    agent = build_agent_from_global_config()
    
    # 应用命令行参数覆盖
    if args.dt:
        config = get_global_config()
        config.time.dt = args.dt
    if args.horizon:
        config = get_global_config()
        config.time.default_horizon = args.horizon
    if args.samples:
        config = get_global_config()
        config.c2osr.samples = args.samples

    ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(0.0, 0.0), yaw_rad=0.0)
    from carla_c2osr.env.types import AgentType
    npc = AgentState(agent_id="npc-0", position_m=(1.0, 0.0), velocity_mps=(0.0, 0.0), heading_rad=0.0, agent_type=AgentType.VEHICLE)

    # 获取horizon配置
    config = get_global_config()
    horizon = config.time.default_horizon
    
    agent.reset()
    for t in range(horizon):
        world = WorldState(time_s=float(t), ego=ego, agents=[npc])
        agent.update(world)
        ctrl: EgoControl = agent.act(world)
        npc = AgentState(agent_id=npc.agent_id, position_m=(npc.position_m[0] + 0.5, npc.position_m[1]), velocity_mps=npc.velocity_mps, heading_rad=npc.heading_rad, agent_type=npc.agent_type)
        print(f"t={t} ctrl={ctrl}")


if __name__ == "__main__":
    main()
