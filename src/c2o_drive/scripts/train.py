#!/usr/bin/env python3
"""Unified training script for C2O-Drive algorithms."""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from c2o_drive.core.types import WorldState, EgoState, AgentState, EgoControl


def setup_logging(log_dir: Path, verbose: bool = False):
    """Set up logging configuration."""
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = log_dir / f"training_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)


def create_agent(algorithm: str, config: Dict[str, Any]):
    """Create agent based on algorithm type."""
    if algorithm == "dqn":
        from c2o_drive.algorithms.dqn import DQNAgent, DQNConfig
        agent_config = DQNConfig(**config)
        return DQNAgent(agent_config)

    elif algorithm == "sac":
        from c2o_drive.algorithms.sac import SACAgent, SACConfig
        agent_config = SACConfig(**config)
        return SACAgent(agent_config)

    elif algorithm == "c2osr":
        from c2o_drive.algorithms.c2osr import C2OSRAgent, C2OSRConfig
        agent_config = C2OSRConfig(**config)
        return C2OSRAgent(agent_config)

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def create_environment(env_type: str, config: Dict[str, Any]):
    """Create environment based on type."""
    if env_type == "virtual":
        from c2o_drive.environments.virtual import VirtualEnvironment
        return VirtualEnvironment(**config)

    elif env_type == "carla":
        from c2o_drive.environments.carla import CarlaEnvironment
        return CarlaEnvironment(**config)

    elif env_type == "simplegrid":
        from c2o_drive.environments.simplegrid import SimpleGridEnvironment
        return SimpleGridEnvironment(**config)

    else:
        raise ValueError(f"Unknown environment: {env_type}")


def train_episode(agent, env, training: bool = True) -> Dict[str, float]:
    """Run one training episode."""
    state = env.reset()
    total_reward = 0.0
    episode_length = 0
    done = False

    while not done:
        # Select action
        if hasattr(agent, 'select_action'):
            action = agent.select_action(state, training=training)
        else:
            action = agent.act(state)

        # Step environment
        next_state, reward, done, info = env.step(action)

        # Store transition if training
        if training and hasattr(agent, 'store_transition'):
            agent.store_transition(state, action, reward, next_state, done)

        # Update state
        state = next_state
        total_reward += reward
        episode_length += 1

        # Train agent
        if training and hasattr(agent, 'train_step'):
            train_info = agent.train_step()

    return {
        "reward": total_reward,
        "length": episode_length,
    }


def evaluate(agent, env, num_episodes: int = 10) -> Dict[str, float]:
    """Evaluate agent performance."""
    rewards = []
    lengths = []

    for _ in range(num_episodes):
        metrics = train_episode(agent, env, training=False)
        rewards.append(metrics["reward"])
        lengths.append(metrics["length"])

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "std_length": np.std(lengths),
    }


def main():
    """Main training loop."""
    parser = argparse.ArgumentParser(description="Train C2O-Drive agents")

    # Algorithm selection
    parser.add_argument(
        "--algorithm", "-a",
        choices=["dqn", "sac", "c2osr"],
        default="dqn",
        help="Algorithm to train",
    )

    # Environment selection
    parser.add_argument(
        "--env", "-e",
        choices=["virtual", "carla", "simplegrid"],
        default="virtual",
        help="Environment to train in",
    )

    # Training parameters
    parser.add_argument(
        "--episodes", "-n",
        type=int,
        default=1000,
        help="Number of training episodes",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=50,
        help="Evaluation frequency (episodes)",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=100,
        help="Model save frequency (episodes)",
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
        help="Device to use for training",
    )

    # Output
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="output",
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

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint to resume from",
    )

    args = parser.parse_args()

    # Set up output directory
    output_dir = Path(args.output) / args.algorithm / f"{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(output_dir, args.verbose)
    logger.info(f"Starting training with algorithm: {args.algorithm}")
    logger.info(f"Environment: {args.env}")
    logger.info(f"Output directory: {output_dir}")

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

    # Load checkpoint if resuming
    if args.resume:
        logger.info(f"Loading checkpoint from: {args.resume}")
        agent.load(args.resume)

    # Training loop
    best_reward = float('-inf')
    training_history = []

    pbar = tqdm(range(args.episodes), desc="Training")
    for episode in pbar:
        # Training episode
        train_metrics = train_episode(agent, env, training=True)
        training_history.append(train_metrics)

        # Update progress bar
        pbar.set_postfix(
            reward=f"{train_metrics['reward']:.2f}",
            length=train_metrics['length']
        )

        # Evaluation
        if (episode + 1) % args.eval_freq == 0:
            eval_metrics = evaluate(agent, env, args.eval_episodes)
            logger.info(f"Episode {episode + 1} - Evaluation: "
                       f"Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}, "
                       f"Length: {eval_metrics['mean_length']:.1f} ± {eval_metrics['std_length']:.1f}")

            # Save best model
            if eval_metrics['mean_reward'] > best_reward:
                best_reward = eval_metrics['mean_reward']
                agent.save(output_dir / "best_model.pth")
                logger.info(f"New best model saved with reward: {best_reward:.2f}")

        # Regular checkpoint
        if (episode + 1) % args.save_freq == 0:
            agent.save(output_dir / f"checkpoint_{episode + 1}.pth")
            logger.info(f"Checkpoint saved at episode {episode + 1}")

    # Save final model
    agent.save(output_dir / "final_model.pth")

    # Save training history
    np.save(output_dir / "training_history.npy", training_history)

    logger.info("Training complete!")
    logger.info(f"Best reward: {best_reward:.2f}")
    logger.info(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()