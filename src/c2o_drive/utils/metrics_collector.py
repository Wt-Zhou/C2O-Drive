"""
Metrics collection and analysis for baseline algorithms.

This module provides a lightweight episode-level data recorder for tracking
training metrics across different RL algorithms (PPO, SAC, Rainbow DQN, RCRL, C2OSR).
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np


def _convert_to_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_json_serializable(item) for item in obj]
    else:
        return obj


class MetricsCollector:
    """
    Lightweight episode-level metrics collector for baseline algorithm comparison.

    Records:
    - Safety metrics: collision rate, near-miss rate, minimum distance
    - Efficiency metrics: success rate, episode length, timeout rate
    - Learning metrics: episode reward, Q-values, training loss
    - Decision quality: action selection, exploration rate

    Usage:
        collector = MetricsCollector("PPO", "s1", output_dir="outputs/ppo")

        # During training
        for episode in range(num_episodes):
            episode_data = {
                'episode_id': episode,
                'steps': 12,
                'total_reward': 45.3,
                'collision': False,
                'near_miss': True,
                'q_value_mean': 50.2,
                ...
            }
            collector.add_episode(episode_data)

        # After training
        collector.save("outputs/ppo/metrics.json")
    """

    def __init__(
        self,
        algorithm_name: str,
        scenario_name: str,
        output_dir: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize metrics collector.

        Args:
            algorithm_name: Name of the algorithm (e.g., "PPO", "SAC")
            scenario_name: Name of the scenario (e.g., "s1", "s2")
            output_dir: Directory to save metrics (optional)
            config: Algorithm configuration dict (optional)
        """
        self.algorithm_name = algorithm_name
        self.scenario_name = scenario_name
        self.output_dir = Path(output_dir) if output_dir else None
        self.config = config or {}

        self.episodes: List[Dict[str, Any]] = []
        self.start_time = time.time()

    def add_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Add episode data to the collector.

        Args:
            episode_data: Dictionary containing episode metrics
                Required fields:
                    - episode_id: int
                    - steps: int
                    - total_reward: float
                    - outcome: str ('success'/'collision'/'timeout'/'planning_failed')
                    - collision: bool
                    - near_miss: bool (v2.0: inclusive definition - collision cases ARE counted as near-miss)
                    - min_distance_to_agents: float
                Optional fields:
                    - q_value_mean, q_value_std, q_value_min, q_value_max: float
                    - selected_action_idx: int
                    - policy_loss, value_loss, policy_entropy: float
                    - epsilon: float (for DQN/RCRL)
                    - safety_violations: int (for RCRL)

        Note:
            Near-miss definition (v2.0): min_distance < GlobalConfig.safety.near_miss_threshold_m (3.0m)
            This is an INCLUSIVE definition where collision ⊆ near-miss, ensuring near_miss_rate >= collision_rate.
            Previous versions used mutual exclusion (near_miss AND NOT collision), which was logically inconsistent.
        """
        # Validate required fields
        required_fields = [
            'episode_id', 'steps', 'total_reward', 'outcome',
            'collision', 'near_miss', 'min_distance_to_agents'
        ]
        for field in required_fields:
            if field not in episode_data:
                raise ValueError(f"Missing required field: {field}")

        # Add timestamp if not present
        if 'timestamp' not in episode_data:
            episode_data['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

        # Add episode time if not present
        if 'episode_time' not in episode_data:
            episode_data['episode_time'] = 0.0

        # Convert numpy types to JSON-serializable Python types
        episode_data_clean = _convert_to_json_serializable(episode_data)

        self.episodes.append(episode_data_clean)

    def compute_summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all episodes.

        Returns:
            Dictionary containing aggregate statistics
        """
        if not self.episodes:
            return {}

        n_episodes = len(self.episodes)

        # Extract episode data
        rewards = [ep['total_reward'] for ep in self.episodes]
        steps = [ep['steps'] for ep in self.episodes]
        times = [ep.get('episode_time', 0.0) for ep in self.episodes]

        # Count outcomes
        collisions = sum(1 for ep in self.episodes if ep['collision'])
        near_misses = sum(1 for ep in self.episodes if ep['near_miss'])
        successes = sum(1 for ep in self.episodes if ep['outcome'] == 'success')
        timeouts = sum(1 for ep in self.episodes if ep['outcome'] == 'timeout')

        # Distance statistics
        min_distances = [ep['min_distance_to_agents'] for ep in self.episodes]

        # Q-value statistics (if available)
        q_values = [ep.get('q_value_mean') for ep in self.episodes
                   if ep.get('q_value_mean') is not None]

        # Compute summary
        summary = {
            'algorithm': self.algorithm_name,
            'scenario': self.scenario_name,
            'total_episodes': n_episodes,
            'total_steps': sum(steps),
            'total_time': sum(times),

            # Safety metrics
            'collision_rate': collisions / n_episodes,
            'near_miss_rate': near_misses / n_episodes,
            'avg_min_distance': float(np.mean(min_distances)),
            'min_distance_std': float(np.std(min_distances)),

            # Efficiency metrics
            'success_rate': successes / n_episodes,
            'timeout_rate': timeouts / n_episodes,
            'avg_episode_length': float(np.mean(steps)),
            'avg_episode_time': float(np.mean(times)),

            # Learning metrics
            'avg_reward': float(np.mean(rewards)),
            'reward_std': float(np.std(rewards)),
            'min_reward': float(np.min(rewards)),
            'max_reward': float(np.max(rewards)),
            'final_100_avg_reward': float(np.mean(rewards[-100:])) if len(rewards) >= 100 else float(np.mean(rewards)),

            # Decision quality (if available)
            'avg_q_value': float(np.mean(q_values)) if q_values else None,
            'q_value_std': float(np.std(q_values)) if q_values else None,

            # Sample efficiency metrics (episodes to reach X% of final performance)
            'episodes_to_80_percent': self.get_episodes_to_threshold(0.8),
            'episodes_to_90_percent': self.get_episodes_to_threshold(0.9),
        }

        # Algorithm-specific metrics
        if self.algorithm_name == 'RCRL':
            safety_violations = [ep.get('safety_violations', 0) for ep in self.episodes]
            summary['avg_safety_violations'] = float(np.mean(safety_violations))
            summary['total_safety_violations'] = sum(safety_violations)

        return summary

    def get_moving_average(self, metric: str = 'total_reward', window: int = 100) -> List[float]:
        """
        Compute moving average of a metric.

        Args:
            metric: Name of the metric to average (default: 'total_reward')
            window: Window size for moving average (default: 100)

        Returns:
            List of moving average values
        """
        if not self.episodes:
            return []

        values = [ep.get(metric, 0.0) for ep in self.episodes]

        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i+1]
            moving_avg.append(float(np.mean(window_values)))

        return moving_avg

    def get_episodes_to_threshold(self, threshold: float = 0.9, window: int = 100) -> Optional[int]:
        """
        Calculate the episode number when moving average first reaches
        (threshold) percent of the improvement from initial to final performance.

        This is a sample efficiency metric measuring how quickly the algorithm
        learns to achieve good performance. Works correctly for both positive
        and negative rewards.

        Args:
            threshold: Performance threshold as fraction (default 0.9 for 90%)
            window: Moving average window size (default 100)

        Returns:
            Episode number (1-indexed) when threshold is first reached,
            or None if never reached or insufficient data

        Example:
            - episodes_to_80_percent = get_episodes_to_threshold(0.8)
            - episodes_to_90_percent = get_episodes_to_threshold(0.9)

        Note:
            Target is calculated as: initial_avg + threshold * (final_avg - initial_avg)
            This works for both positive and negative rewards.
        """
        if not self.episodes or len(self.episodes) < window:
            return None

        rewards = [ep['total_reward'] for ep in self.episodes]

        # Calculate initial and final performance
        initial_avg = np.mean(rewards[:window])
        final_avg = np.mean(rewards[-window:])

        # Target = initial + threshold * (final - initial)
        # This represents reaching 'threshold' percent of the improvement
        improvement = final_avg - initial_avg
        target = initial_avg + threshold * improvement

        # Calculate moving average and find first crossing
        for i in range(len(rewards)):
            start_idx = max(0, i - window + 1)
            window_values = rewards[start_idx:i+1]
            avg = np.mean(window_values)

            # Check if we've reached the target
            # For positive improvement (final > initial): avg >= target
            # For negative improvement (final < initial): avg <= target
            if improvement >= 0:
                if avg >= target:
                    return i + 1
            else:
                if avg <= target:
                    return i + 1

        return None  # Never reached target

    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save metrics to JSON file.

        Args:
            filepath: Path to save JSON file (optional)
                     If not provided, saves to output_dir/metrics.json

        Returns:
            Path to saved file
        """
        # Determine save path
        if filepath:
            save_path = Path(filepath)
        elif self.output_dir:
            save_path = self.output_dir / "metrics.json"
        else:
            save_path = Path(f"metrics_{self.algorithm_name}_{self.scenario_name}.json")

        # Create parent directory if needed
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data structure
        data = {
            'algorithm': self.algorithm_name,
            'scenario': self.scenario_name,
            'config': self.config,
            'training_time': time.time() - self.start_time,
            'episodes': self.episodes,
            'summary': self.compute_summary()
        }

        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)

        return str(save_path)

    @staticmethod
    def load(filepath: str) -> 'MetricsCollector':
        """
        Load metrics from JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            MetricsCollector instance with loaded data
        """
        with open(filepath, 'r') as f:
            data = json.load(f)

        collector = MetricsCollector(
            algorithm_name=data['algorithm'],
            scenario_name=data['scenario'],
            config=data.get('config', {})
        )

        collector.episodes = data.get('episodes', [])
        collector.start_time = time.time() - data.get('training_time', 0)

        return collector

    def get_episode_data(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """
        Get data for a specific episode.

        Args:
            episode_id: Episode ID to retrieve

        Returns:
            Episode data dictionary, or None if not found
        """
        for ep in self.episodes:
            if ep['episode_id'] == episode_id:
                return ep
        return None

    def get_metric_series(self, metric: str) -> List[Any]:
        """
        Get time series of a specific metric across all episodes.

        Args:
            metric: Name of the metric (e.g., 'total_reward', 'steps')

        Returns:
            List of metric values
        """
        return [ep.get(metric) for ep in self.episodes]

    def print_summary(self) -> None:
        """Print formatted summary statistics."""
        summary = self.compute_summary()

        print(f"\n{'='*70}")
        print(f" {self.algorithm_name} Training Summary - Scenario: {self.scenario_name}")
        print(f"{'='*70}")
        print(f"  Episodes:           {summary['total_episodes']}")
        print(f"  Total Steps:        {summary['total_steps']}")
        print(f"  Training Time:      {summary['total_time']:.1f}s")
        print(f"\n  Safety Metrics:")
        print(f"    Collision Rate:   {summary['collision_rate']:.2%}")
        print(f"    Near-miss Rate:   {summary['near_miss_rate']:.2%}")
        print(f"    Avg Min Distance: {summary['avg_min_distance']:.2f}m")
        print(f"\n  Efficiency Metrics:")
        print(f"    Success Rate:     {summary['success_rate']:.2%}")
        print(f"    Timeout Rate:     {summary['timeout_rate']:.2%}")
        print(f"    Avg Episode Len:  {summary['avg_episode_length']:.1f} steps")
        print(f"\n  Learning Metrics:")
        print(f"    Avg Reward:       {summary['avg_reward']:.2f}")
        print(f"    Reward Std:       {summary['reward_std']:.2f}")
        print(f"    Final 100 Avg:    {summary['final_100_avg_reward']:.2f}")

        if summary.get('avg_q_value') is not None:
            print(f"\n  Decision Quality:")
            print(f"    Avg Q-value:      {summary['avg_q_value']:.2f}")
            print(f"    Q-value Std:      {summary['q_value_std']:.2f}")

        if self.algorithm_name == 'RCRL':
            print(f"\n  RCRL Safety:")
            print(f"    Total Violations: {summary.get('total_safety_violations', 0)}")
            print(f"    Avg Violations:   {summary.get('avg_safety_violations', 0):.2f}")

        print(f"{'='*70}\n")
