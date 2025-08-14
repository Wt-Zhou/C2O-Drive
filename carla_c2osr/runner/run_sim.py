from __future__ import annotations
import argparse
import yaml
from pathlib import Path
import sys
from typing import Any

# 允许从包目录内直接运行：将仓库根加入 sys.path
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from carla_c2osr.env.types import AgentState, EgoControl, EgoState, WorldState
from carla_c2osr.agents.c2osr.agent import C2OSRConfig, C2OSRDriveAgent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_agent_from_cfg(sim_cfg: dict[str, Any]) -> C2OSRDriveAgent:
    grid = sim_cfg["grid"]
    dp = sim_cfg["dp"]
    risk = sim_cfg["risk"]
    cfg = C2OSRConfig(
        grid_size_m=float(grid["size_m"]),
        grid_cell_m=float(grid["cell_m"]),
        grid_macro=bool(grid["macro"]),
        horizon=int(sim_cfg["horizon"]),
        samples=int(sim_cfg["samples"]),
        alpha=float(dp["alpha"]),
        c=float(dp["c"]),
        eta=float(dp["eta"]),
        add_thresh=float(dp["add_thresh"]),
        risk_mode=str(risk["mode"]),
        epsilon=float(risk["epsilon"]),
        gamma=float(sim_cfg["gamma"]),
    )
    return C2OSRDriveAgent(cfg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-env", action="store_true")
    default_cfg = Path(__file__).resolve().parents[1] / "configs" / "sim.yaml"
    parser.add_argument("--sim-cfg", type=str, default=str(default_cfg))
    args = parser.parse_args()

    sim_cfg = load_yaml(Path(args.sim_cfg))
    agent = build_agent_from_cfg(sim_cfg)

    ego = EgoState(position_m=(0.0, 0.0), velocity_mps=(0.0, 0.0), heading_rad=0.0)
    npc = AgentState(agent_id="npc-0", position_m=(1.0, 0.0), velocity_mps=(0.0, 0.0))

    agent.reset()
    for t in range(sim_cfg["horizon"]):
        world = WorldState(time_s=float(t), ego=ego, agents=[npc])
        agent.update(world)
        ctrl: EgoControl = agent.act(world)
        npc = AgentState(agent_id=npc.agent_id, position_m=(npc.position_m[0] + 0.5, npc.position_m[1]), velocity_mps=npc.velocity_mps)
        print(f"t={t} ctrl={ctrl}")


if __name__ == "__main__":
    main()
