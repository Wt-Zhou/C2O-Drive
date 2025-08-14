from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np

from carla_c2osr.env.types import WorldState, EgoControl
from .grid import GridSpec, GridMapper
from .dp_mixture import DPMixtureConfig, DirichletProcessMixture
from .transition import sample_dirichlet
from .lattice import generate_lattice_controls
from .planner import PlannerConfig, C2OSRPlanner


@dataclass
class C2OSRConfig:
    grid_size_m: float
    grid_cell_m: float
    grid_macro: bool
    horizon: int
    samples: int
    alpha: float
    c: float
    eta: float
    add_thresh: float
    risk_mode: str
    epsilon: float
    gamma: float


class C2OSRDriveAgent:
    """C2OSR-Drive 智能体占位实现。"""

    def __init__(self, cfg: C2OSRConfig) -> None:
        self.grid_spec = GridSpec(cfg.grid_size_m, cfg.grid_cell_m, cfg.grid_macro)
        self.grid_mapper = GridMapper(self.grid_spec)
        self.dp = DirichletProcessMixture(
            DPMixtureConfig(
                alpha=cfg.alpha,
                eta=cfg.eta,
                add_thresh=cfg.add_thresh,
                grid_num_cells=self.grid_spec.num_cells,
            )
        )
        self.planner = C2OSRPlanner(
            PlannerConfig(
                horizon=cfg.horizon,
                samples=cfg.samples,
                risk_mode=cfg.risk_mode,
                epsilon=cfg.epsilon,
                gamma=cfg.gamma,
            )
        )
        self.c = cfg.c
        self.horizon = cfg.horizon
        self.samples = cfg.samples

    def reset(self) -> None:
        self.dp.atoms.clear()
        self.dp.counts.clear()

    def update(self, world: WorldState) -> None:
        if world.agents:
            idx = self.grid_mapper.world_to_cell(world.agents[0].position_m)
            self.dp.update_with_observation(idx)

    def act(self, world: WorldState) -> EgoControl:
        z_samples = self.dp.sample_z_vectors(self.samples)
        samples_q: List[List[List[float]]] = []
        for z in z_samples:
            t = sample_dirichlet(z, self.c)
            q_layers = [t for _ in range(self.horizon)]
            samples_q.append(q_layers)
        trajectories = generate_lattice_controls(world.ego, self.horizon)
        _, info = self.planner.plan(world, trajectories, samples_q)
        return trajectories[1].controls[0]
