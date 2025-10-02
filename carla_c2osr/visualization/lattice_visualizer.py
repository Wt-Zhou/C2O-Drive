"""
Lattice Trajectory Selection Visualization Module

Visualize candidate trajectories and their Q-value evaluation results
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection


def visualize_lattice_selection(
    trajectory_q_values: List[Dict[str, Any]],
    selected_trajectory_info: Dict[str, Any],
    current_world_state,
    grid,
    episode_idx: int,
    output_dir: Path,
    show_top_n: int = None
) -> None:
    """Visualize lattice trajectory selection process

    Args:
        trajectory_q_values: Q-value evaluation results for all candidate trajectories
        selected_trajectory_info: Selected trajectory information
        current_world_state: Current world state
        grid: Grid mapper
        episode_idx: Episode index
        output_dir: Output directory
        show_top_n: Show only top N trajectories by Q value (None=show all)
    """

    if not trajectory_q_values:
        print(f"  ⚠️ No trajectory data available for visualization")
        return

    # 创建图形
    fig, (ax_traj, ax_q) = plt.subplots(1, 2, figsize=(16, 7))

    # ===== Left: Trajectory Visualization =====

    # Sort by min_q (low to high)
    sorted_trajectories = sorted(trajectory_q_values, key=lambda x: x['min_q'])

    # Show top N if specified
    if show_top_n is not None and show_top_n < len(sorted_trajectories):
        display_trajectories = sorted_trajectories[-show_top_n:]  # Highest N
    else:
        display_trajectories = sorted_trajectories

    # Find Q value range for color mapping
    min_q_values = [traj['min_q'] for traj in display_trajectories]
    q_min, q_max = min(min_q_values), max(min_q_values)
    q_range = q_max - q_min if q_max > q_min else 1.0

    # Plot all candidate trajectories
    for traj_info in display_trajectories:
        trajectory = traj_info['trajectory']
        min_q = traj_info['min_q']
        is_selected = (traj_info['trajectory_id'] == selected_trajectory_info['trajectory_id'])

        # Color based on Q value (red=low, green=high)
        normalized_q = (min_q - q_min) / q_range
        color = plt.cm.RdYlGn(normalized_q)

        # Extract x, y coordinates
        xs = [pos[0] for pos in trajectory]
        ys = [pos[1] for pos in trajectory]

        if is_selected:
            # Selected trajectory: thick line + highlight
            ax_traj.plot(xs, ys, 'o-', linewidth=4, markersize=8,
                        color=color, label=f'Selected: Traj{traj_info["trajectory_id"]} (min_Q={min_q:.1f})',
                        zorder=10, alpha=0.9)
            # Add start marker
            ax_traj.plot(xs[0], ys[0], 'g*', markersize=20, zorder=11,
                        markeredgecolor='black', markeredgewidth=1.5)
        else:
            # Other trajectories: thin lines
            ax_traj.plot(xs, ys, 'o-', linewidth=1.5, markersize=4,
                        color=color, alpha=0.5, zorder=5)

    # Plot ego and agents
    ego_pos = current_world_state.ego.position_m
    ax_traj.plot(ego_pos[0], ego_pos[1], 'bs', markersize=15,
                label='Ego Vehicle', zorder=12, markeredgecolor='black', markeredgewidth=1.5)

    for i, agent in enumerate(current_world_state.agents):
        agent_pos = agent.position_m
        ax_traj.plot(agent_pos[0], agent_pos[1], 'r^', markersize=12,
                    label=f'Agent {i+1}' if i == 0 else '',
                    zorder=12, markeredgecolor='black', markeredgewidth=1)

    # Set left plot properties
    ax_traj.set_xlabel('X (m)', fontsize=12)
    ax_traj.set_ylabel('Y (m)', fontsize=12)
    ax_traj.set_title(f'Episode {episode_idx}: Lattice Candidates ({len(display_trajectories)}/{len(trajectory_q_values)} trajs)',
                     fontsize=14, fontweight='bold')
    ax_traj.grid(True, alpha=0.3)
    ax_traj.legend(loc='best', fontsize=10)
    ax_traj.axis('equal')

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn,
                               norm=plt.Normalize(vmin=q_min, vmax=q_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax_traj)
    cbar.set_label('Min Q Value', fontsize=11)

    # ===== Right: Q Value Bar Chart =====

    # Sorted trajectory IDs and Q values
    sorted_ids = [traj['trajectory_id'] for traj in sorted_trajectories]
    sorted_min_qs = [traj['min_q'] for traj in sorted_trajectories]
    sorted_mean_qs = [traj['mean_q'] for traj in sorted_trajectories]
    sorted_collision_rates = [traj['collision_rate'] for traj in sorted_trajectories]

    x_pos = np.arange(len(sorted_ids))

    # Plot min_Q bar chart
    colors_bar = [plt.cm.RdYlGn((q - q_min) / q_range) for q in sorted_min_qs]
    bars = ax_q.bar(x_pos, sorted_min_qs, color=colors_bar, alpha=0.8,
                    edgecolor='black', linewidth=1.5)

    # Highlight selected trajectory
    selected_idx = sorted_ids.index(selected_trajectory_info['trajectory_id'])
    bars[selected_idx].set_edgecolor('lime')
    bars[selected_idx].set_linewidth(4)

    # Annotate collision rates on bars
    for i, (bar, collision_rate) in enumerate(zip(bars, sorted_collision_rates)):
        height = bar.get_height()
        ax_q.text(bar.get_x() + bar.get_width()/2., height,
                 f'{collision_rate:.2f}',
                 ha='center', va='bottom', fontsize=8, rotation=90)

    # Set right plot properties
    ax_q.set_xlabel('Trajectory ID', fontsize=12)
    ax_q.set_ylabel('Min Q Value', fontsize=12)
    ax_q.set_title(f'Q Value Comparison (Green=Selected)', fontsize=14, fontweight='bold')
    ax_q.set_xticks(x_pos)
    ax_q.set_xticklabels([f'{tid}' for tid in sorted_ids], rotation=45)
    ax_q.grid(True, alpha=0.3, axis='y')
    ax_q.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    # Add selected trajectory info text
    info_text = (
        f"Selected: Traj{selected_trajectory_info['trajectory_id']}\n"
        f"Offset: {selected_trajectory_info['lateral_offset']:.1f}m\n"
        f"Speed: {selected_trajectory_info['target_speed']:.1f}m/s\n"
        f"Min Q: {selected_trajectory_info['min_q']:.2f}\n"
        f"Mean Q: {selected_trajectory_info['mean_q']:.2f}\n"
        f"Collision: {selected_trajectory_info['collision_rate']:.3f}"
    )
    ax_q.text(0.02, 0.98, info_text,
             transform=ax_q.transAxes,
             fontsize=11,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"lattice_selection_ep{episode_idx:02d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Lattice selection visualization saved: {output_path.name}")


def visualize_lattice_trajectories_detailed(
    trajectory_q_values: List[Dict[str, Any]],
    selected_trajectory_info: Dict[str, Any],
    current_world_state,
    grid,
    episode_idx: int,
    output_dir: Path
) -> None:
    """Detailed visualization: Q value distribution for each trajectory

    Shows complete Q value distribution (boxplot) for each candidate
    """

    if not trajectory_q_values:
        return

    # Create figure
    n_trajectories = len(trajectory_q_values)
    fig, ax = plt.subplots(figsize=(max(12, n_trajectories * 0.8), 6))

    # Collect Q value distributions for each trajectory
    q_distributions = []
    labels = []
    colors = []

    for traj_info in sorted(trajectory_q_values, key=lambda x: x['trajectory_id']):
        q_distributions.append(traj_info['q_values'])
        labels.append(f"T{traj_info['trajectory_id']}\n"
                     f"({traj_info['lateral_offset']:+.0f}m,"
                     f"{traj_info['target_speed']:.0f}m/s)")

        # Green for selected trajectory
        is_selected = (traj_info['trajectory_id'] == selected_trajectory_info['trajectory_id'])
        colors.append('lightgreen' if is_selected else 'lightblue')

    # Plot boxplot
    bp = ax.boxplot(q_distributions, labels=labels, patch_artist=True,
                    showmeans=True, meanline=True)

    # Set colors
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Highlight selected trajectory
    selected_idx = [i for i, traj in enumerate(sorted(trajectory_q_values, key=lambda x: x['trajectory_id']))
                   if traj['trajectory_id'] == selected_trajectory_info['trajectory_id']][0]
    bp['boxes'][selected_idx].set_edgecolor('green')
    bp['boxes'][selected_idx].set_linewidth(3)

    ax.set_xlabel('Trajectory (offset, speed)', fontsize=12)
    ax.set_ylabel('Q Value Distribution', fontsize=12)
    ax.set_title(f'Episode {episode_idx}: Q Value Distributions (Green=Selected)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save figure
    output_path = output_dir / f"lattice_q_distributions_ep{episode_idx:02d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Q distribution visualization saved: {output_path.name}")
