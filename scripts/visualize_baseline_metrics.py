#!/usr/bin/env python3
"""
Visualization script for baseline algorithm comparison.

Generates paper-quality comparison charts from metrics.json files.

Usage:
    # Single scenario comparison
    python scripts/visualize_baseline_metrics.py \\
        --ppo outputs/s1/ppo/metrics.json \\
        --sac outputs/s1/sac/metrics.json \\
        --rainbow outputs/s1/rainbow_dqn/metrics.json \\
        --rcrl outputs/s1/rcrl/metrics.json \\
        --c2osr outputs/s1/c2osr/metrics.json \\
        --output paper_figures/s1_comparison

    # Multiple scenarios aggregation
    python scripts/visualize_baseline_metrics.py \\
        --scenarios s1 s2 s3 s4 s5 \\
        --output paper_figures/all_scenarios
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
plt.style.use('seaborn-v0_8-darkgrid')

# Set publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Algorithm colors for consistency
ALGO_COLORS = {
    'PPO': '#1f77b4',      # Blue
    'SAC': '#ff7f0e',      # Orange
    'RainbowDQN': '#2ca02c',  # Green
    'RCRL': '#d62728',     # Red
    'C2OSR': '#9467bd',    # Purple
}


def load_metrics(filepath: str) -> Dict:
    """Load metrics from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def smooth_curve(values: List[float], window: int = 10) -> np.ndarray:
    """Smooth curve using moving average."""
    if len(values) < window:
        return np.array(values)

    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')

    # Pad to match original length
    pad_size = len(values) - len(smoothed)
    return np.pad(smoothed, (pad_size, 0), mode='edge')


def plot_learning_curves(metrics_dict: Dict[str, Dict], output_dir: Path):
    """Plot learning curves (reward, Q-value, etc.)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Learning Curves Comparison', fontsize=16, fontweight='bold')

    # Episode Reward
    ax = axes[0, 0]
    for algo_name, metrics in metrics_dict.items():
        episodes = [ep['episode_id'] for ep in metrics['episodes']]
        rewards = [ep['total_reward'] for ep in metrics['episodes']]
        ax.plot(episodes, rewards, label=algo_name, color=ALGO_COLORS.get(algo_name, 'gray'), alpha=0.6)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Episode Reward')
    ax.set_title('Episode Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Moving Average Reward (window=100)
    ax = axes[0, 1]
    for algo_name, metrics in metrics_dict.items():
        episodes = [ep['episode_id'] for ep in metrics['episodes']]
        rewards = [ep['total_reward'] for ep in metrics['episodes']]
        smoothed = smooth_curve(rewards, window=min(100, len(rewards) // 5))
        ax.plot(episodes, smoothed, label=algo_name, color=ALGO_COLORS.get(algo_name, 'gray'), linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Reward (100-episode window)')
    ax.set_title('Moving Average Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Q-value Evolution
    ax = axes[1, 0]
    for algo_name, metrics in metrics_dict.items():
        episodes = [ep['episode_id'] for ep in metrics['episodes'] if ep.get('q_value_mean') is not None]
        q_values = [ep['q_value_mean'] for ep in metrics['episodes'] if ep.get('q_value_mean') is not None]
        if q_values:
            ax.plot(episodes, q_values, label=algo_name, color=ALGO_COLORS.get(algo_name, 'gray'), alpha=0.7)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Q-value (Mean)')
    ax.set_title('Q-value Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Collision Rate (Cumulative)
    ax = axes[1, 1]
    for algo_name, metrics in metrics_dict.items():
        episodes = [ep['episode_id'] for ep in metrics['episodes']]
        collisions = [ep['collision'] for ep in metrics['episodes']]
        cumulative_collisions = np.cumsum(collisions)
        cumulative_rate = cumulative_collisions / np.arange(1, len(collisions) + 1)
        ax.plot(episodes, cumulative_rate, label=algo_name, color=ALGO_COLORS.get(algo_name, 'gray'), linewidth=2)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Collision Rate (Cumulative)')
    ax.set_title('Cumulative Collision Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_safety_comparison(metrics_dict: Dict[str, Dict], output_dir: Path):
    """Plot safety metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Safety Metrics Comparison', fontsize=16, fontweight='bold')

    algo_names = list(metrics_dict.keys())

    # Collision Rate
    ax = axes[0, 0]
    collision_rates = [metrics_dict[algo]['summary']['collision_rate'] for algo in algo_names]
    bars = ax.bar(algo_names, collision_rates, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Collision Rate')
    ax.set_title('Collision Rate Comparison')
    ax.set_ylim(0, max(collision_rates) * 1.2 if max(collision_rates) > 0 else 1.0)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{collision_rates[i]:.2%}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Near-miss Rate
    ax = axes[0, 1]
    near_miss_rates = [metrics_dict[algo]['summary']['near_miss_rate'] for algo in algo_names]
    bars = ax.bar(algo_names, near_miss_rates, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Near-miss Rate')
    ax.set_title('Near-miss Rate Comparison')
    ax.set_ylim(0, max(near_miss_rates) * 1.2 if max(near_miss_rates) > 0 else 1.0)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{near_miss_rates[i]:.2%}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Average Minimum Distance
    ax = axes[1, 0]
    min_distances = [metrics_dict[algo]['summary']['avg_min_distance'] for algo in algo_names]
    bars = ax.bar(algo_names, min_distances, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Average Min Distance (m)')
    ax.set_title('Average Minimum Distance to Agents')
    ax.axhline(y=3.0, color='red', linestyle='--', linewidth=2, label='Near-miss threshold (3.0m)')
    ax.legend()
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{min_distances[i]:.2f}m',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Safety Score (composite)
    ax = axes[1, 1]
    # Safety score = (1 - collision_rate) * (1 - near_miss_rate) * min(avg_min_distance / 10.0, 1.0)
    safety_scores = [
        (1 - metrics_dict[algo]['summary']['collision_rate']) *
        (1 - metrics_dict[algo]['summary']['near_miss_rate']) *
        min(metrics_dict[algo]['summary']['avg_min_distance'] / 10.0, 1.0)
        for algo in algo_names
    ]
    bars = ax.bar(algo_names, safety_scores, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Safety Score')
    ax.set_title('Composite Safety Score')
    ax.set_ylim(0, 1.0)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{safety_scores[i]:.3f}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'safety_comparison.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_efficiency_comparison(metrics_dict: Dict[str, Dict], output_dir: Path):
    """Plot efficiency metrics comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Efficiency Metrics Comparison', fontsize=16, fontweight='bold')

    algo_names = list(metrics_dict.keys())

    # Success Rate
    ax = axes[0, 0]
    success_rates = [metrics_dict[algo]['summary']['success_rate'] for algo in algo_names]
    bars = ax.bar(algo_names, success_rates, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Success Rate')
    ax.set_title('Task Success Rate')
    ax.set_ylim(0, 1.0)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{success_rates[i]:.2%}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Average Episode Length
    ax = axes[0, 1]
    avg_lengths = [metrics_dict[algo]['summary']['avg_episode_length'] for algo in algo_names]
    bars = ax.bar(algo_names, avg_lengths, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Average Episode Length (steps)')
    ax.set_title('Average Episode Length')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{avg_lengths[i]:.1f}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Timeout Rate
    ax = axes[1, 0]
    timeout_rates = [metrics_dict[algo]['summary']['timeout_rate'] for algo in algo_names]
    bars = ax.bar(algo_names, timeout_rates, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Timeout Rate')
    ax.set_title('Episode Timeout Rate')
    ax.set_ylim(0, max(timeout_rates) * 1.2 if max(timeout_rates) > 0 else 1.0)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{timeout_rates[i]:.2%}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Sample Efficiency (reward per episode)
    ax = axes[1, 1]
    # Calculate episodes to reach 80% of best avg reward
    sample_efficiency = []
    for algo in algo_names:
        rewards = [ep['total_reward'] for ep in metrics_dict[algo]['episodes']]
        if len(rewards) >= 100:
            target = metrics_dict[algo]['summary']['final_100_avg_reward'] * 0.8
            # Find first episode where moving avg exceeds target
            moving_avg = []
            for i in range(len(rewards)):
                window_start = max(0, i - 99)
                window = rewards[window_start:i+1]
                moving_avg.append(np.mean(window))

            episodes_to_target = len(rewards)  # Default: didn't reach
            for i, avg in enumerate(moving_avg):
                if avg >= target:
                    episodes_to_target = i + 1
                    break
            sample_efficiency.append(episodes_to_target)
        else:
            sample_efficiency.append(len(rewards))

    bars = ax.bar(algo_names, sample_efficiency, color=[ALGO_COLORS.get(algo, 'gray') for algo in algo_names])
    ax.set_ylabel('Episodes to 80% Final Performance')
    ax.set_title('Sample Efficiency')
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(sample_efficiency[i])}',
                ha='center', va='bottom', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = output_dir / 'efficiency_comparison.png'
    plt.savefig(output_path)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def plot_q_value_distribution(metrics_dict: Dict[str, Dict], output_dir: Path):
    """Plot Q-value distribution boxplot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Q-value Distribution Comparison', fontsize=16, fontweight='bold')

    data_to_plot = []
    labels = []
    colors = []

    for algo_name, metrics in metrics_dict.items():
        q_means = [ep['q_value_mean'] for ep in metrics['episodes'] if ep.get('q_value_mean') is not None]
        if q_means:
            data_to_plot.append(q_means)
            labels.append(algo_name)
            colors.append(ALGO_COLORS.get(algo_name, 'gray'))

    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Q-value (Mean)')
        ax.set_title('Q-value Distribution Across Episodes')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        output_path = output_dir / 'q_value_distribution.png'
        plt.savefig(output_path)
        plt.close()
        print(f"  ✓ Saved: {output_path}")
    else:
        plt.close()
        print(f"  ⚠ No Q-value data available for boxplot")


def plot_summary_table(metrics_dict: Dict[str, Dict], output_dir: Path):
    """Generate a summary table as an image."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    algo_names = list(metrics_dict.keys())
    table_data = []

    headers = ['Algorithm', 'Success Rate', 'Collision Rate', 'Near-miss Rate',
               'Avg Reward', 'Avg Episode Length', 'Avg Min Distance']

    for algo in algo_names:
        summary = metrics_dict[algo]['summary']
        row = [
            algo,
            f"{summary['success_rate']:.2%}",
            f"{summary['collision_rate']:.2%}",
            f"{summary['near_miss_rate']:.2%}",
            f"{summary['avg_reward']:.2f}",
            f"{summary['avg_episode_length']:.1f}",
            f"{summary['avg_min_distance']:.2f}m",
        ]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    colWidths=[0.15, 0.13, 0.13, 0.13, 0.13, 0.17, 0.16])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color rows by algorithm
    for i, algo in enumerate(algo_names):
        color = ALGO_COLORS.get(algo, 'lightgray')
        table[(i+1, 0)].set_facecolor(color)
        table[(i+1, 0)].set_text_props(weight='bold', color='white')

    plt.title('Performance Summary Table', fontsize=16, fontweight='bold', pad=20)

    output_path = output_dir / 'summary_table.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  ✓ Saved: {output_path}")


def visualize_single_scenario(metrics_paths: Dict[str, str], output_dir: Path):
    """Visualize comparison for a single scenario."""
    print(f"\nGenerating visualizations for single scenario...")
    print(f"Output directory: {output_dir}")

    # Load all metrics
    metrics_dict = {}
    for algo_name, path in metrics_paths.items():
        if path and Path(path).exists():
            metrics_dict[algo_name] = load_metrics(path)
            print(f"  ✓ Loaded {algo_name}: {path}")
        else:
            print(f"  ⚠ Skipping {algo_name}: file not found")

    if not metrics_dict:
        print("  ✗ No valid metrics files found!")
        return

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print(f"\nGenerating plots...")
    plot_learning_curves(metrics_dict, output_dir)
    plot_safety_comparison(metrics_dict, output_dir)
    plot_efficiency_comparison(metrics_dict, output_dir)
    plot_q_value_distribution(metrics_dict, output_dir)
    plot_summary_table(metrics_dict, output_dir)

    print(f"\n✓ All visualizations saved to: {output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Visualize baseline algorithm comparison metrics",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Single scenario mode
    parser.add_argument("--ppo", type=str, help="Path to PPO metrics.json")
    parser.add_argument("--sac", type=str, help="Path to SAC metrics.json")
    parser.add_argument("--rainbow", type=str, help="Path to Rainbow DQN metrics.json")
    parser.add_argument("--rcrl", type=str, help="Path to RCRL metrics.json")
    parser.add_argument("--c2osr", type=str, help="Path to C2OSR metrics.json")

    # Output settings
    parser.add_argument("--output", type=str, default="paper_figures",
                       help="Output directory for figures")

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()

    # Prepare metrics paths
    metrics_paths = {
        'PPO': args.ppo,
        'SAC': args.sac,
        'RainbowDQN': args.rainbow,
        'RCRL': args.rcrl,
        'C2OSR': args.c2osr,
    }

    # Remove None entries
    metrics_paths = {k: v for k, v in metrics_paths.items() if v is not None}

    if not metrics_paths:
        print("Error: No metrics files specified!")
        print("Please specify at least one metrics file using --ppo, --sac, --rainbow, --rcrl, or --c2osr")
        return

    # Visualize
    output_dir = Path(args.output)
    visualize_single_scenario(metrics_paths, output_dir)


if __name__ == "__main__":
    main()
