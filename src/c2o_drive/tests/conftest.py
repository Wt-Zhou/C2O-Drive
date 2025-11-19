"""Pytest configuration and fixtures for C2O-Drive tests."""

import sys
import os
from pathlib import Path
import pytest
import numpy as np
import torch

# Add src directory to path for imports
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

# Add legacy paths for backward compatibility during migration
project_root = src_path.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "carla_c2osr"))


@pytest.fixture(autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def device():
    """Provide the appropriate device for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for test files."""
    return tmp_path


@pytest.fixture
def mock_world_state():
    """Provide a mock WorldState for testing."""
    from c2o_drive.core.types import WorldState, EgoState, AgentState

    ego = EgoState(
        position_m=np.array([0.0, 0.0]),
        heading_rad=0.0,
        speed_mps=10.0,
    )

    agents = [
        AgentState(
            position_m=np.array([20.0, 0.0]),
            heading_rad=0.0,
            speed_mps=8.0,
        ),
        AgentState(
            position_m=np.array([10.0, 5.0]),
            heading_rad=np.pi/4,
            speed_mps=5.0,
        ),
    ]

    return WorldState(ego=ego, agents=agents)


@pytest.fixture
def mock_config():
    """Provide mock configuration objects."""
    from dataclasses import dataclass

    @dataclass
    class MockConfig:
        horizon: int = 5
        dt: float = 0.1
        grid_size: float = 100.0
        state_dim: int = 128
        action_dim: int = 9
        learning_rate: float = 1e-4
        batch_size: int = 32
        device: str = "cpu"

    return MockConfig()


# Markers for test categorization
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "functional: Functional tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "carla: Tests requiring CARLA")
    config.addinivalue_line("markers", "gpu: Tests requiring GPU")