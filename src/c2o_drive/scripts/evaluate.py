#!/usr/bin/env python3
"""Evaluation script for trained C2O-Drive agents."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from c2o_drive.scripts.train import create_agent, create_environment, setup_logging


def evaluate_agent(
    agent,
    env,
    num_episodes: int = 100,
    render: bool = False,
    save_trajectories: bool = False,
) -> Dict[str, Any]:
    """Evaluate agent performance with detailed metrics."""
    metrics = {
        "rewards": [],
        "lengths": [],
        "collisions": [],
        "success_rate": [],
        "trajectories": [],
    }

    for episode in tqdm(range(num_episodes), desc="Evaluating"):
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        trajectory = []
        done = False

        while not done:
            # Select action (deterministic for evaluation)
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, training=False)
            else:
                action = agent.act(state)

            # Step environment
            next_state, reward, done, info = env.step(action)

            # Record trajectory if needed
            if save_trajectories:
                trajectory.append({
                    "state": state.copy() if isinstance(state, np.ndarray) else state,
                    "action": action,
                    "reward": reward,
                })

            # Render if requested
            if render:
                env.render()

            # Update metrics
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Store episode metrics
        metrics["rewards"].append(episode_reward)
        metrics["lengths"].append(episode_length)

        # Check for collision/success in info
        if "collision" in info:
            metrics["collisions"].append(info["collision"])
        if "success" in info:
            metrics["success_rate"].append(info["success"])

        if save_trajectories:
            metrics["trajectories"].append(trajectory)

    # Calculate statistics
    results = {
        "mean_reward": np.mean(metrics["rewards"]),
        "std_reward": np.std(metrics["rewards"]),
        "min_reward": np.min(metrics["rewards"]),
        "max_reward": np.max(metrics["rewards"]),
        "mean_length": np.mean(metrics["lengths"]),
        "std_length": np.std(metrics["lengths"]),
    }

    if metrics["collisions"]:
        results["collision_rate"] = np.mean(metrics["collisions"])

    if metrics["success_rate"]:
        results["success_rate"] = np.mean(metrics["success_rate"])

    if save_trajectories:
        results["trajectories"] = metrics["trajectories"]

    results["raw_rewards"] = metrics["rewards"]
    results["raw_lengths"] = metrics["lengths"]

    return results


def plot_results(results: Dict[str, Any], output_dir: Path):
    """Create visualization plots for evaluation results."""
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Reward distribution
    ax = axes[0, 0]
    ax.hist(results["raw_rewards"], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results["mean_reward"], color='red', linestyle='--',
               label=f'Mean: {results["mean_reward"]:.2f}')
    ax.set_xlabel("Episode Reward")
    ax.set_ylabel("Frequency")
    ax.set_title("Reward Distribution")
    ax.legend()

    # Episode length distribution
    ax = axes[0, 1]
    ax.hist(results["raw_lengths"], bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(results["mean_length"], color='red', linestyle='--',
               label=f'Mean: {results["mean_length"]:.2f}')
    ax.set_xlabel("Episode Length")
    ax.set_ylabel("Frequency")
    ax.set_title("Episode Length Distribution")
    ax.legend()

    # Reward over episodes
    ax = axes[1, 0]
    episodes = range(len(results["raw_rewards"]))
    ax.plot(episodes, results["raw_rewards"], alpha=0.7)
    # Add rolling average
    window = min(20, len(results["raw_rewards"]) // 5)
    if window > 1:
        rolling_avg = np.convolve(results["raw_rewards"],
                                 np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(results["raw_rewards"])),
               rolling_avg, color='red', linewidth=2,
               label=f'Rolling Avg (window={window})')
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Reward Over Episodes")
    ax.legend()

    # Summary statistics
    ax = axes[1, 1]
    ax.axis('off')
    stats_text = f"""Evaluation Summary:

Episodes: {len(results['raw_rewards'])}
Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}
Min/Max Reward: {results['min_reward']:.2f} / {results['max_reward']:.2f}
Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}"""

    if "collision_rate" in results:
        stats_text += f"\nCollision Rate: {results['collision_rate']:.2%}"
    if "success_rate" in results:
        stats_text += f"\nSuccess Rate: {results['success_rate']:.2%}"

    ax.text(0.1, 0.5, stats_text, fontsize=12, family='monospace',
            verticalalignment='center')

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_results.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained C2O-Drive agents")

    # Model selection
    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="Path to trained model",
    )
    parser.add_argument(
        "--algorithm", "-a",
        choices=["dqn", "sac", "c2osr"],
        required=True,
        help="Algorithm type",
    )

    # Environment selection
    parser.add_argument(
        "--env", "-e",
        choices=["virtual", "carla", "simplegrid"],
        default="virtual",
        help="Environment to evaluate in",
    )

    # Evaluation parameters
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during evaluation",
    )
    parser.add_argument(
        "--save-trajectories",
        action="store_true",
        help="Save episode trajectories",
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="evaluation",
        help="Output directory",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir, args.verbose)
    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Environment: {args.env}")

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # Load configuration
    config = {}
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)

    # Override device in config
    if "agent" not in config:
        config["agent"] = {}
    config["agent"]["device"] = args.device

    # Create agent and environment
    agent = create_agent(args.algorithm, config.get("agent", {}))
    env = create_environment(args.env, config.get("env", {}))

    # Load trained model
    logger.info(f"Loading model from: {args.model}")
    agent.load(args.model)

    # Run evaluation
    logger.info(f"Running {args.episodes} evaluation episodes...")
    results = evaluate_agent(
        agent,
        env,
        num_episodes=args.episodes,
        render=args.render,
        save_trajectories=args.save_trajectories,
    )

    # Log results
    logger.info(f"Evaluation complete!")
    logger.info(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    logger.info(f"Mean Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")

    if "collision_rate" in results:
        logger.info(f"Collision Rate: {results['collision_rate']:.2%}")
    if "success_rate" in results:
        logger.info(f"Success Rate: {results['success_rate']:.2%}")

    # Save results
    results_file = output_dir / "evaluation_results.json"
    with open(results_file, 'w') as f:
        # Remove trajectories from JSON (too large)
        save_results = {k: v for k, v in results.items()
                       if k != "trajectories" and k != "raw_rewards" and k != "raw_lengths"}
        json.dump(save_results, f, indent=2)

    # Save raw data
    np.savez(
        output_dir / "evaluation_data.npz",
        rewards=results["raw_rewards"],
        lengths=results["raw_lengths"],
    )

    # Save trajectories if requested
    if args.save_trajectories and "trajectories" in results:
        np.save(output_dir / "trajectories.npy", results["trajectories"])
        logger.info(f"Trajectories saved to: {output_dir / 'trajectories.npy'}")

    # Create plots
    plot_results(results, output_dir)
    logger.info(f"Plots saved to: {output_dir / 'evaluation_results.png'}")

    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    main()