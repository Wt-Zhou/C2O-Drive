"""State space discretization and representation interfaces.

This module provides abstractions for representing and discretizing state spaces,
which is crucial for algorithms that require discrete representations
(e.g., tabular methods, grid-based planning, Dirichlet-based learning).
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, TypeVar, Generic, Hashable
from dataclasses import dataclass
import numpy as np


StateType = TypeVar("StateType")
DiscreteStateType = TypeVar("DiscreteStateType", bound=Hashable)


@dataclass
class DiscreteState:
    """A discrete state representation.

    Attributes:
        index: Integer index or tuple of indices
        cell_id: Unique hashable identifier
        continuous_value: Optional continuous value this represents
    """
    index: int | Tuple[int, ...]
    cell_id: Hashable | None = None
    continuous_value: Any | None = None

    def __hash__(self):
        if self.cell_id is not None:
            return hash(self.cell_id)
        return hash(self.index)


class StateSpaceDiscretizer(ABC, Generic[StateType, DiscreteStateType]):
    """Base interface for state space discretization.

    Discretizers convert continuous states to discrete representations,
    which is necessary for:
    - Tabular RL methods
    - Grid-based planning
    - Dirichlet process learning (C2OSR)
    - Spatial hashing
    """

    @abstractmethod
    def discretize(self, state: StateType) -> DiscreteStateType:
        """Convert a continuous state to discrete representation.

        Args:
            state: Continuous state

        Returns:
            discrete_state: Discrete representation
        """
        pass

    @abstractmethod
    def get_dimension(self) -> int:
        """Get the dimensionality of the discrete space.

        Returns:
            dimension: Number of discrete states (cells)
        """
        pass

    def batch_discretize(self, states: List[StateType]) -> List[DiscreteStateType]:
        """Discretize a batch of states.

        Args:
            states: List of continuous states

        Returns:
            discrete_states: List of discrete representations
        """
        return [self.discretize(state) for state in states]


class ReachabilityComputer(ABC, Generic[StateType, DiscreteStateType]):
    """Interface for computing reachable states.

    Computes the set of states reachable from a given state within
    one or more timesteps, considering dynamics constraints.
    """

    @abstractmethod
    def get_successors(self,
                       state: DiscreteStateType,
                       timesteps: int = 1) -> List[DiscreteStateType]:
        """Get reachable successor states.

        Args:
            state: Current discrete state
            timesteps: Number of timesteps to look ahead

        Returns:
            successors: List of reachable states
        """
        pass

    @abstractmethod
    def get_transition_probability(self,
                                   from_state: DiscreteStateType,
                                   to_state: DiscreteStateType) -> float:
        """Get transition probability between states.

        Args:
            from_state: Source state
            to_state: Target state

        Returns:
            probability: P(to_state | from_state) under dynamics
        """
        pass


class GridBasedDiscretizer(StateSpaceDiscretizer[np.ndarray, DiscreteState]):
    """Grid-based spatial discretization.

    Discretizes continuous space using a uniform or non-uniform grid.
    This is the discretization method currently used in C2OSR.
    """

    def __init__(self,
                 bounds: List[Tuple[float, float]],
                 resolution: List[float] | float):
        """
        Args:
            bounds: List of (min, max) for each dimension
            resolution: Grid resolution (cell size) for each dimension
                       Can be a single float for uniform resolution
        """
        self.bounds = np.array(bounds)
        self.ndim = len(bounds)

        if isinstance(resolution, (int, float)):
            self.resolution = np.full(self.ndim, resolution)
        else:
            self.resolution = np.array(resolution)

        # Compute grid size for each dimension
        self.grid_shape = ((self.bounds[:, 1] - self.bounds[:, 0]) / self.resolution).astype(int)
        self.total_cells = int(np.prod(self.grid_shape))

    def discretize(self, state: np.ndarray) -> DiscreteState:
        """Discretize a continuous state to grid cell.

        Args:
            state: Continuous state (numpy array)

        Returns:
            discrete_state: Grid cell representation
        """
        # Clip to bounds
        clipped = np.clip(state, self.bounds[:, 0], self.bounds[:, 1])

        # Compute grid indices
        indices = ((clipped - self.bounds[:, 0]) / self.resolution).astype(int)

        # Clip to valid grid range
        indices = np.clip(indices, 0, self.grid_shape - 1)

        # Convert multi-dimensional index to flat index
        flat_index = int(np.ravel_multi_index(indices, self.grid_shape))

        return DiscreteState(
            index=flat_index,
            cell_id=tuple(indices),
            continuous_value=state
        )

    def get_dimension(self) -> int:
        """Get total number of grid cells."""
        return self.total_cells

    def index_to_position(self, index: int | Tuple[int, ...]) -> np.ndarray:
        """Convert grid index back to continuous position (cell center).

        Args:
            index: Flat index or tuple of indices

        Returns:
            position: Continuous position at cell center
        """
        if isinstance(index, int):
            indices = np.unravel_index(index, self.grid_shape)
        else:
            indices = np.array(index)

        position = self.bounds[:, 0] + (indices + 0.5) * self.resolution
        return position


class FeatureBasedDiscretizer(StateSpaceDiscretizer[StateType, int]):
    """Feature-based discretization using learned representations.

    Uses learned features (e.g., from a neural network encoder) and
    discretizes the feature space using clustering (e.g., K-means).

    This is useful for:
    - High-dimensional state spaces
    - Learned representations from deep RL
    - Adaptive discretization based on data
    """

    def __init__(self,
                 feature_extractor: Any,
                 num_clusters: int,
                 cluster_method: str = "kmeans"):
        """
        Args:
            feature_extractor: Function/model to extract features from states
            num_clusters: Number of discrete clusters
            cluster_method: Clustering method ('kmeans', 'gmm', etc.)
        """
        self.feature_extractor = feature_extractor
        self.num_clusters = num_clusters
        self.cluster_method = cluster_method
        self.clusterer = None  # To be initialized after seeing data

    def discretize(self, state: StateType) -> int:
        """Discretize state using learned features and clustering.

        Args:
            state: Input state

        Returns:
            cluster_id: Discrete cluster ID
        """
        # Extract features
        features = self.feature_extractor(state)

        # Assign to nearest cluster
        if self.clusterer is None:
            raise ValueError("Clusterer not initialized. Call fit() first.")

        cluster_id = self.clusterer.predict(features.reshape(1, -1))[0]
        return int(cluster_id)

    def get_dimension(self) -> int:
        """Get number of clusters."""
        return self.num_clusters

    def fit(self, states: List[StateType]) -> None:
        """Fit the clustering model on a set of states.

        Args:
            states: List of states to fit on
        """
        from sklearn.cluster import KMeans

        # Extract features for all states
        features = np.array([self.feature_extractor(s) for s in states])

        # Fit clusterer
        if self.cluster_method == "kmeans":
            self.clusterer = KMeans(n_clusters=self.num_clusters, random_state=0)
            self.clusterer.fit(features)
        else:
            raise ValueError(f"Unknown cluster method: {self.cluster_method}")


class PathCoordinateDiscretizer(StateSpaceDiscretizer[StateType, DiscreteState]):
    """Discretizes state using Frenet-Serret coordinates along a reference path.

    Represents state as (s, d, s_dot, d_dot) where:
    - s: longitudinal position along path
    - d: lateral deviation from path
    - s_dot: longitudinal velocity
    - d_dot: lateral velocity

    This is useful for path-following tasks and provides a more natural
    representation than Cartesian coordinates.
    """

    def __init__(self,
                 reference_path: Any,
                 s_resolution: float = 1.0,
                 d_resolution: float = 0.5,
                 v_resolution: float = 1.0):
        """
        Args:
            reference_path: Reference path/trajectory
            s_resolution: Resolution along the path
            d_resolution: Resolution perpendicular to path
            v_resolution: Resolution for velocities
        """
        self.reference_path = reference_path
        self.s_resolution = s_resolution
        self.d_resolution = d_resolution
        self.v_resolution = v_resolution

    def discretize(self, state: StateType) -> DiscreteState:
        """Convert state to Frenet coordinates and discretize.

        Args:
            state: State in Cartesian coordinates

        Returns:
            discrete_state: Discretized Frenet state
        """
        # Convert to Frenet coordinates
        frenet = self._cartesian_to_frenet(state)

        # Discretize each component
        s_idx = int(frenet[0] / self.s_resolution)
        d_idx = int(frenet[1] / self.d_resolution)
        s_dot_idx = int(frenet[2] / self.v_resolution)
        d_dot_idx = int(frenet[3] / self.v_resolution)

        indices = (s_idx, d_idx, s_dot_idx, d_dot_idx)

        return DiscreteState(
            index=hash(indices),  # Use hash as flat index
            cell_id=indices,
            continuous_value=frenet
        )

    def get_dimension(self) -> int:
        """Get approximate dimension (depends on path length)."""
        # This is an approximation; actual dimension depends on path
        return 10000  # Placeholder

    def _cartesian_to_frenet(self, state: StateType) -> np.ndarray:
        """Convert Cartesian state to Frenet coordinates.

        Args:
            state: State with (x, y, vx, vy)

        Returns:
            frenet: Frenet coordinates (s, d, s_dot, d_dot)
        """
        # Placeholder implementation
        # Real implementation needs to project onto reference path
        raise NotImplementedError("Frenet conversion not yet implemented")
