#!/usr/bin/env python3
"""
Visualize PPO Training Metrics

This script loads training metrics from PPO runs and creates visualizations
showing episode rewards, collision rates, and training losses over time.

Usage:
    python scripts/visualize_ppo_training.py --metrics outputs/ppo_carla/metrics.json
    python scripts/visualize_ppo_training.py --metrics outputs/ppo_carla/metrics.json --output figures/ppo_training.png
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load metrics from JSON file."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def compute_moving_average(data: List[float], window: int = 10) -> List[float]:
    """Compute moving average with given window size."""
    if len(data) < window:
        window = max(1, len(data) // 2)

    moving_avg = []
    for i in range(len(data)):
        start_idx = max(0, i - window + 1)
        moving_avg.append(np.mean(data[start_idx:i+1]))
    return moving_avg


def plot_training_metrics(metrics: Dict[str, Any], output_path: str = None, window: int = 10):
    """
    Create comprehensive training metrics visualization.

    Args:
        metrics: Loaded metrics dictionary
        output_path: Path to save the figure (optional)
        window: Window size for moving average
    """
    episodes = metrics['episodes']

    # Extract data
    episode_ids = [ep['episode_id'] for ep in episodes]
    rewards = [ep['total_reward'] for ep in episodes]
    steps = [ep['steps'] for ep in episodes]
    collisions = [ep['collision'] for ep in episodes]
    near_misses = [ep['near_miss'] for ep in episodes]

    # Extract training metrics (may be None for some episodes)
    policy_losses = [ep.get('policy_loss') for ep in episodes]
    value_losses = [ep.get('value_loss') for ep in episodes]
    entropies = [ep.get('policy_entropy') for ep in episodes]

    # Filter out None values for loss plots
    valid_policy_loss_episodes = [(i, loss) for i, loss in enumerate(policy_losses) if loss is not None]
    valid_value_loss_episodes = [(i, loss) for i, loss in enumerate(value_losses) if loss is not None]
    valid_entropy_episodes = [(i, ent) for i, ent in enumerate(entropies) if ent is not None]

    # Compute moving averages
    rewards_ma = compute_moving_average(rewards, window)

    # Compute collision and near-miss rates
    collision_rate = []
    near_miss_rate = []
    rate_window = 20
    for i in range(len(episodes)):
        start_idx = max(0, i - rate_window + 1)
        collision_rate.append(np.mean(collisions[start_idx:i+1]) * 100)
        near_miss_rate.append(np.mean(near_misses[start_idx:i+1]) * 100)

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Episode Rewards
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(episode_ids, rewards, alpha=0.3, color='blue', label='Episode Reward')
    ax1.plot(episode_ids, rewards_ma, linewidth=2, color='darkblue',
             label=f'Moving Average (window={window})')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('PPO Training: Episode Rewards', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Add summary statistics
    summary_text = (f'Mean: {np.mean(rewards):.2f} | '
                   f'Std: {np.std(rewards):.2f} | '
                   f'Min: {np.min(rewards):.2f} | '
                   f'Max: {np.max(rewards):.2f}')
    ax1.text(0.02, 0.98, summary_text, transform=ax1.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 2. Episode Steps
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(episode_ids, steps, color='green', alpha=0.6)
    ax2.plot(episode_ids, compute_moving_average(steps, window),
             linewidth=2, color='darkgreen', label=f'MA (window={window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Length', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # 3. Collision & Near-Miss Rates
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(episode_ids, collision_rate, color='red', linewidth=2, label='Collision Rate')
    ax3.plot(episode_ids, near_miss_rate, color='orange', linewidth=2, label='Near-Miss Rate')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Rate (%)')
    ax3.set_title(f'Safety Metrics (window={rate_window})', fontsize=12, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 105])

    # 4. Policy Loss
    ax4 = fig.add_subplot(gs[2, 0])
    if valid_policy_loss_episodes:
        ep_indices = [episode_ids[i] for i, _ in valid_policy_loss_episodes]
        losses = [loss for _, loss in valid_policy_loss_episodes]
        ax4.plot(ep_indices, losses, color='purple', alpha=0.6, label='Policy Loss')
        if len(losses) >= window:
            losses_ma = compute_moving_average(losses, window)
            ax4.plot(ep_indices, losses_ma, linewidth=2, color='darkviolet',
                    label=f'MA (window={window})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Loss')
        ax4.set_title('Policy Loss', fontsize=12, fontweight='bold')
        ax4.legend(loc='best')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No Policy Loss Data', ha='center', va='center',
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Policy Loss', fontsize=12, fontweight='bold')

    # 5. Value Loss & Entropy
    ax5 = fig.add_subplot(gs[2, 1])
    if valid_value_loss_episodes and valid_entropy_episodes:
        # Value loss on left y-axis
        ep_indices_val = [episode_ids[i] for i, _ in valid_value_loss_episodes]
        val_losses = [loss for _, loss in valid_value_loss_episodes]
        color_val = 'brown'
        ax5.plot(ep_indices_val, val_losses, color=color_val, alpha=0.6, label='Value Loss')
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Value Loss', color=color_val)
        ax5.tick_params(axis='y', labelcolor=color_val)
        ax5.grid(True, alpha=0.3)

        # Entropy on right y-axis
        ax5_twin = ax5.twinx()
        ep_indices_ent = [episode_ids[i] for i, _ in valid_entropy_episodes]
        ent_values = [ent for _, ent in valid_entropy_episodes]
        color_ent = 'teal'
        ax5_twin.plot(ep_indices_ent, ent_values, color=color_ent, alpha=0.6, label='Policy Entropy')
        ax5_twin.set_ylabel('Entropy', color=color_ent)
        ax5_twin.tick_params(axis='y', labelcolor=color_ent)

        ax5.set_title('Value Loss & Policy Entropy', fontsize=12, fontweight='bold')

        # Combine legends
        lines1, labels1 = ax5.get_legend_handles_labels()
        lines2, labels2 = ax5_twin.get_legend_handles_labels()
        ax5.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax5.text(0.5, 0.5, 'No Value Loss / Entropy Data', ha='center', va='center',
                transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Value Loss & Policy Entropy', fontsize=12, fontweight='bold')

    # Overall title
    algorithm_name = metrics['metadata'].get('algorithm', 'PPO')
    scenario_name = metrics['metadata'].get('scenario', 'Unknown')
    total_episodes = len(episodes)
    fig.suptitle(f'{algorithm_name} Training Metrics - {scenario_name} Scenario ({total_episodes} episodes)',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")
    else:
        plt.tight_layout()
        plt.show()

    plt.close()


def plot_simple_reward_curve(metrics: Dict[str, Any], output_path: str = None, window: int = 10):
    """
    Create a simple, clean reward curve plot.

    Args:
        metrics: Loaded metrics dictionary
        output_path: Path to save the figure (optional)
        window: Window size for moving average
    """
    episodes = metrics['episodes']

    # Extract data
    episode_ids = [ep['episode_id'] for ep in episodes]
    rewards = [ep['total_reward'] for ep in episodes]

    # Compute moving average
    rewards_ma = compute_moving_average(rewards, window)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot
    ax.plot(episode_ids, rewards, alpha=0.3, color='steelblue', linewidth=1, label='Episode Reward')
    ax.plot(episode_ids, rewards_ma, linewidth=2.5, color='darkblue',
            label=f'Moving Average (window={window})')
    ax.fill_between(episode_ids, rewards, rewards_ma, alpha=0.1, color='blue')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Formatting
    ax.set_xlabel('Episode', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('PPO Training: Episode Reward Progression', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Add summary statistics
    summary_text = (f'Episodes: {len(rewards)} | '
                   f'Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f} | '
                   f'Best: {np.max(rewards):.2f}')
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    plt.tight_layout()

    # Save or show
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Figure saved to: {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize PPO training metrics')
    parser.add_argument('--metrics', type=str, required=True,
                       help='Path to metrics.json file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for the figure (optional)')
    parser.add_argument('--simple', action='store_true',
                       help='Create simple reward curve only (instead of full dashboard)')
    parser.add_argument('--window', type=int, default=10,
                       help='Moving average window size (default: 10)')

    args = parser.parse_args()

    # Load metrics
    print(f"Loading metrics from: {args.metrics}")
    metrics = load_metrics(args.metrics)

    num_episodes = len(metrics['episodes'])
    print(f"✓ Loaded {num_episodes} episodes")

    # Create visualization
    if args.simple:
        print("Creating simple reward curve...")
        plot_simple_reward_curve(metrics, args.output, args.window)
    else:
        print("Creating comprehensive training dashboard...")
        plot_training_metrics(metrics, args.output, args.window)

    print("✓ Done!")


if __name__ == "__main__":
    main()
