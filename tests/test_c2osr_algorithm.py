"""
Comprehensive test suite for C2OSR algorithm adapter.

Tests the algorithm interface implementation including:
- Configuration
- Planner functionality
- Evaluator functionality
- Factory functions
- Integration with environments
"""

import sys
from pathlib import Path
import tempfile
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from carla_c2osr.algorithms.c2osr import (
    C2OSRPlannerConfig,
    C2OSREvaluatorConfig,
    C2OSRPlanner,
    C2OSREvaluator,
    create_c2osr_planner,
    create_c2osr_evaluator,
    create_c2osr_planner_evaluator_pair,
)
from carla_c2osr.env.types import EgoState, AgentState, WorldState, EgoControl, AgentType
from carla_c2osr.core.planner import Transition
from carla_c2osr.environments import SimpleGridEnvironment


def create_test_world_state() -> WorldState:
    """Create a test world state."""
    ego = EgoState(
        position_m=(0.0, 0.0),
        velocity_mps=(5.0, 0.0),
        yaw_rad=0.0,
    )

    agents = [
        AgentState(
            agent_id="agent_1",
            position_m=(10.0, 2.0),
            velocity_mps=(3.0, 0.0),
            heading_rad=0.0,
            agent_type=AgentType.VEHICLE,
        ),
        AgentState(
            agent_id="agent_2",
            position_m=(15.0, -2.0),
            velocity_mps=(4.0, 0.0),
            heading_rad=0.0,
            agent_type=AgentType.VEHICLE,
        ),
    ]

    return WorldState(ego=ego, agents=agents, time_s=0.0)


class TestC2OSRConfiguration:
    """Test configuration classes."""

    def test_default_planner_config(self):
        """Test default planner configuration."""
        config = C2OSRPlannerConfig()

        assert config.grid.grid_size_m == 1.0
        assert config.dirichlet.alpha_in == 1.0
        assert config.lattice.horizon == 10
        assert config.q_value.n_samples == 100
        assert config.gamma == 0.99

        print("✓ Default planner config created successfully")

    def test_custom_planner_config(self):
        """Test custom planner configuration."""
        from carla_c2osr.algorithms.c2osr.config import GridConfig, QValueConfig

        config = C2OSRPlannerConfig(
            grid=GridConfig(grid_size_m=2.0),
            q_value=QValueConfig(horizon=15, n_samples=200),
            gamma=0.95,
        )

        assert config.grid.grid_size_m == 2.0
        assert config.q_value.horizon == 15
        assert config.q_value.n_samples == 200
        assert config.gamma == 0.95

        print("✓ Custom planner config created successfully")

    def test_evaluator_config_from_planner(self):
        """Test creating evaluator config from planner config."""
        planner_config = C2OSRPlannerConfig()
        evaluator_config = C2OSREvaluatorConfig.from_planner_config(planner_config)

        assert evaluator_config.grid.grid_size_m == planner_config.grid.grid_size_m
        assert evaluator_config.q_value.horizon == planner_config.q_value.horizon

        print("✓ Evaluator config created from planner config")


class TestC2OSRPlanner:
    """Test C2OSR planner functionality."""

    def test_planner_creation(self):
        """Test basic planner creation."""
        config = C2OSRPlannerConfig()
        planner = C2OSRPlanner(config)

        assert planner.config == config
        assert planner.grid_mapper is not None
        assert planner.trajectory_buffer is not None
        assert planner.dirichlet_bank is not None
        assert planner.lattice_planner is not None

        print("✓ Planner created successfully")

    def test_action_selection(self):
        """Test action selection."""
        config = C2OSRPlannerConfig()
        planner = C2OSRPlanner(config)

        world_state = create_test_world_state()
        reference_path = [(i * 5.0, 0.0) for i in range(20)]

        action = planner.select_action(
            observation=world_state,
            reference_path=reference_path,
        )

        assert isinstance(action, EgoControl)
        assert 0.0 <= action.throttle <= 1.0
        assert -1.0 <= action.steer <= 1.0
        assert 0.0 <= action.brake <= 1.0

        print(f"✓ Action selected: throttle={action.throttle:.2f}, "
              f"steer={action.steer:.2f}, brake={action.brake:.2f}")

    def test_trajectory_planning(self):
        """Test trajectory planning."""
        config = C2OSRPlannerConfig()
        planner = C2OSRPlanner(config)

        world_state = create_test_world_state()
        reference_path = [(i * 5.0, 0.0) for i in range(20)]

        trajectory = planner.plan_trajectory(
            observation=world_state,
            horizon=10,
            reference_path=reference_path,
        )

        assert len(trajectory) == 10
        assert all(isinstance(action, EgoControl) for action in trajectory)

        print(f"✓ Trajectory planned with {len(trajectory)} steps")

    def test_planner_update(self):
        """Test planner update with transition."""
        config = C2OSRPlannerConfig()
        planner = C2OSRPlanner(config)

        world_state = create_test_world_state()
        action = EgoControl(throttle=0.5, steer=0.0, brake=0.0)

        # Create next state
        next_ego = EgoState(
            position_m=(5.0, 0.0),
            velocity_mps=(5.0, 0.0),
            yaw_rad=0.0,
        )
        next_state = WorldState(ego=next_ego, agents=world_state.agents, time_s=0.1)

        transition = Transition(
            state=world_state,
            action=action,
            reward=1.0,
            next_state=next_state,
            terminated=False,
            truncated=False,
            info={},
        )

        metrics = planner.update(transition)

        assert isinstance(metrics.loss, float)
        assert 'buffer_size' in metrics.custom
        assert metrics.custom['buffer_size'] >= 1

        print(f"✓ Planner updated, buffer size: {metrics.custom['buffer_size']}")

    def test_planner_reset(self):
        """Test planner reset."""
        config = C2OSRPlannerConfig()
        planner = C2OSRPlanner(config)

        # Do some actions
        world_state = create_test_world_state()
        planner.select_action(world_state)
        planner.select_action(world_state)

        # Reset
        planner.reset()

        assert planner.episode_step_count == 0
        assert planner.last_selected_trajectory is None

        print("✓ Planner reset successfully")

    def test_planner_save_load(self):
        """Test planner save and load."""
        config = C2OSRPlannerConfig()
        planner = C2OSRPlanner(config)

        # Do some updates
        world_state = create_test_world_state()
        action = planner.select_action(world_state)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            planner.save(tmpdir)

            # Create new planner and load
            new_planner = C2OSRPlanner(config)
            new_planner.load(tmpdir)

            # Check buffer size matches
            assert len(new_planner.trajectory_buffer) == len(planner.trajectory_buffer)

        print("✓ Planner saved and loaded successfully")


class TestC2OSREvaluator:
    """Test C2OSR evaluator functionality."""

    def test_evaluator_creation(self):
        """Test basic evaluator creation."""
        config = C2OSREvaluatorConfig()
        evaluator = C2OSREvaluator(config)

        assert evaluator.config == config
        assert evaluator.grid_mapper is not None
        assert evaluator.q_value_calculator is not None

        print("✓ Evaluator created successfully")

    def test_trajectory_evaluation(self):
        """Test trajectory evaluation."""
        config = C2OSREvaluatorConfig()
        evaluator = C2OSREvaluator(config)

        # Create test trajectory
        trajectory = [(i * 1.0, 0.0) for i in range(10)]

        world_state = create_test_world_state()
        context = {
            'current_state': world_state,
            'dt': 1.0,
        }

        result = evaluator.evaluate(trajectory, context)

        assert 'q_value' in result
        assert 'success' in result
        assert isinstance(result['q_value'], float)

        print(f"✓ Trajectory evaluated, Q-value: {result['q_value']:.3f}")

    def test_detailed_evaluation(self):
        """Test detailed trajectory evaluation."""
        config = C2OSREvaluatorConfig()
        evaluator = C2OSREvaluator(config)

        trajectory = [(i * 1.0, 0.0) for i in range(10)]
        world_state = create_test_world_state()

        context = {
            'current_state': world_state,
            'dt': 1.0,
            'return_details': True,
        }

        result = evaluator.evaluate(trajectory, context)

        assert 'q_value' in result
        assert 'collision_free' in result
        assert 'comfort_score' in result
        assert 'efficiency_score' in result
        assert 'safety_score' in result

        print(f"✓ Detailed evaluation complete:")
        print(f"  Q-value: {result['q_value']:.3f}")
        print(f"  Collision-free: {result['collision_free']}")
        print(f"  Comfort: {result['comfort_score']:.3f}")
        print(f"  Efficiency: {result['efficiency_score']:.3f}")
        print(f"  Safety: {result['safety_score']:.3f}")

    def test_batch_evaluation(self):
        """Test batch trajectory evaluation."""
        config = C2OSREvaluatorConfig()
        evaluator = C2OSREvaluator(config)

        trajectories = [
            [(i * 1.0, 0.0) for i in range(10)],
            [(i * 1.0, 1.0) for i in range(10)],
            [(i * 1.0, -1.0) for i in range(10)],
        ]

        world_state = create_test_world_state()
        context = {'current_state': world_state, 'dt': 1.0}

        results = evaluator.evaluate_batch(trajectories, context)

        assert len(results) == 3
        assert all('q_value' in r for r in results)

        print(f"✓ Batch evaluation complete, evaluated {len(results)} trajectories")


class TestC2OSRFactory:
    """Test factory functions."""

    def test_create_planner(self):
        """Test planner creation via factory."""
        planner = create_c2osr_planner()

        assert isinstance(planner, C2OSRPlanner)
        assert planner.config is not None

        print("✓ Planner created via factory")

    def test_create_evaluator(self):
        """Test evaluator creation via factory."""
        evaluator = create_c2osr_evaluator()

        assert isinstance(evaluator, C2OSREvaluator)
        assert evaluator.config is not None

        print("✓ Evaluator created via factory")

    def test_create_pair(self):
        """Test creating planner-evaluator pair."""
        planner, evaluator = create_c2osr_planner_evaluator_pair()

        assert isinstance(planner, C2OSRPlanner)
        assert isinstance(evaluator, C2OSREvaluator)

        # They should share the same buffer and Dirichlet bank
        assert evaluator.trajectory_buffer is planner.trajectory_buffer
        assert evaluator.dirichlet_bank is planner.dirichlet_bank

        print("✓ Planner-evaluator pair created with shared state")

    def test_create_evaluator_with_planner(self):
        """Test creating evaluator that shares state with planner."""
        planner = create_c2osr_planner()
        evaluator = create_c2osr_evaluator(planner=planner)

        assert evaluator.trajectory_buffer is planner.trajectory_buffer
        assert evaluator.dirichlet_bank is planner.dirichlet_bank

        print("✓ Evaluator created sharing state with planner")


class TestC2OSRIntegration:
    """Test integration with environments."""

    def test_env_integration(self):
        """Test planner integration with SimpleGridEnvironment."""
        # Create environment
        env = SimpleGridEnvironment(dt=1.0, max_episode_steps=20)

        # Create planner
        planner = create_c2osr_planner()

        # Run episode
        state, info = env.reset(seed=42)
        total_reward = 0.0
        steps = 0

        for _ in range(10):
            # Get reference path from environment
            reference_path = [(i * 5.0, 0.0) for i in range(20)]

            # Select action
            action = planner.select_action(state, reference_path=reference_path)

            # Step environment
            step_result = env.step(action)

            # Update planner
            transition = Transition(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.observation,
                terminated=step_result.terminated,
                truncated=step_result.truncated,
                info=step_result.info,
            )
            planner.update(transition)

            total_reward += step_result.reward
            state = step_result.observation
            steps += 1

            if step_result.terminated or step_result.truncated:
                break

        env.close()

        print(f"✓ Environment integration test complete:")
        print(f"  Steps: {steps}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Buffer size: {len(planner.trajectory_buffer)}")

    def test_multiple_episodes(self):
        """Test planner across multiple episodes."""
        env = SimpleGridEnvironment(dt=1.0, max_episode_steps=15)
        planner = create_c2osr_planner()

        episode_rewards = []

        for episode in range(3):
            state, _ = env.reset(seed=42 + episode)
            planner.reset()
            episode_reward = 0.0

            for _ in range(15):
                reference_path = [(i * 5.0, 0.0) for i in range(20)]
                action = planner.select_action(state, reference_path=reference_path)
                step_result = env.step(action)

                transition = Transition(
                    state=state,
                    action=action,
                    reward=step_result.reward,
                    next_state=step_result.observation,
                    terminated=step_result.terminated,
                    truncated=step_result.truncated,
                    info=step_result.info,
                )
                planner.update(transition)

                episode_reward += step_result.reward
                state = step_result.observation

                if step_result.terminated or step_result.truncated:
                    break

            episode_rewards.append(episode_reward)

        env.close()

        print(f"✓ Multiple episodes test complete:")
        for i, reward in enumerate(episode_rewards):
            print(f"  Episode {i+1}: {reward:.2f}")
        print(f"  Final buffer size: {len(planner.trajectory_buffer)}")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 70)
    print(" C2OSR ALGORITHM ADAPTER TEST SUITE")
    print("=" * 70)

    # Configuration tests
    print("\n--- Configuration Tests ---")
    config_tests = TestC2OSRConfiguration()
    config_tests.test_default_planner_config()
    config_tests.test_custom_planner_config()
    config_tests.test_evaluator_config_from_planner()

    # Planner tests
    print("\n--- Planner Tests ---")
    planner_tests = TestC2OSRPlanner()
    planner_tests.test_planner_creation()
    planner_tests.test_action_selection()
    planner_tests.test_trajectory_planning()
    planner_tests.test_planner_update()
    planner_tests.test_planner_reset()
    planner_tests.test_planner_save_load()

    # Evaluator tests
    print("\n--- Evaluator Tests ---")
    evaluator_tests = TestC2OSREvaluator()
    evaluator_tests.test_evaluator_creation()
    evaluator_tests.test_trajectory_evaluation()
    evaluator_tests.test_detailed_evaluation()
    evaluator_tests.test_batch_evaluation()

    # Factory tests
    print("\n--- Factory Tests ---")
    factory_tests = TestC2OSRFactory()
    factory_tests.test_create_planner()
    factory_tests.test_create_evaluator()
    factory_tests.test_create_pair()
    factory_tests.test_create_evaluator_with_planner()

    # Integration tests
    print("\n--- Integration Tests ---")
    integration_tests = TestC2OSRIntegration()
    integration_tests.test_env_integration()
    integration_tests.test_multiple_episodes()

    print("\n" + "=" * 70)
    print(" ✓ ALL TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    try:
        run_all_tests()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
