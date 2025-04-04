"""Module for plotting memory and node connectivity data from HNSW index experiments."""
import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scripts.plots.plot_utils import (load_csv_data, setup_plot_style, save_plot, set_axis_limits)

def plot_memory_usage(df: pd.DataFrame,
                     title: str,
                     save_dir: str,
                     filename: str,
                     experiment_paths: Dict[str, List[str]]):
    """Plot memory usage metrics over iterations."""
    fig, ax = plt.subplots()
    setup_plot_style()

    # Plot each memory metric
    metrics = ['index_size', 'index_capacity', 'index_mem_usage']
    colors = ['blue', 'green', 'red']

    for metric, color in zip(metrics, colors):
        ax.plot(df.index, df[metric], label=metric, color=color)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Memory (bytes)')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()

    # Set axis limits without extra space
    set_axis_limits(ax, df.index)

    save_plot(fig, save_dir, filename, experiment_paths)

def plot_slot_distribution(df: pd.DataFrame,
                          title: str,
                          save_dir: str,
                          filename: str,
                          experiment_paths: Dict[str, List[str]]):
    """Plot slot distribution showing the distribution of slot types over time."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Stacked bar chart showing slot distribution
    x = range(len(df))
    bottom = np.zeros(len(df))

    # Plot deleted slots at the bottom
    ax.bar(x, df['slot_lookup_deleted_slots'], label='Deleted Slots', bottom=bottom)
    bottom += df['slot_lookup_deleted_slots']

    # Plot populated slots in the middle
    ax.bar(x, df['slot_lookup_populated_slots'], label='Populated Slots', bottom=bottom)
    bottom += df['slot_lookup_populated_slots']

    # Plot empty slots at the top
    empty_slots = df['slot_lookup_total_slots'] - df['slot_lookup_populated_slots'] - df['slot_lookup_deleted_slots']
    ax.bar(x, empty_slots, label='Empty Slots', bottom=bottom)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Slots')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits without extra space
    set_axis_limits(ax, pd.Series(x))

    save_plot(fig, save_dir, f"{filename}_slot_distribution", experiment_paths)
    plt.close()

def plot_memory_usage_over_time(df: pd.DataFrame,
                            title: str,
                            save_dir: str,
                            filename: str,
                            experiment_paths: Dict[str, List[str]]):
    """Plot memory usage over time."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Line plot showing memory usage in megabytes
    ax.plot(range(len(df)), df['index_mem_usage'] / 1e6, label='Memory Usage', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Memory Usage (megabytes)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(range(len(df))), y_padding=0.4)

    save_plot(fig, save_dir, f"{filename}_memory_usage_over_time", experiment_paths)
    plt.close()

def plot_node_connectivity(df: pd.DataFrame,
                         dataset_name: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]]):
    """Plot non-level-specific node connectivity metrics over time."""
    setup_plot_style()

    # Define non-level specific metrics and their colors
    metrics = {
        'nodes_count': '#1f77b4',  # blue
        'unreachable_count': '#ff7f0e',  # orange
        'avg_connections': '#d62728'  # red
    }

    # Create iteration sequence
    iterations = range(len(df))

    # 1. Individual plots for each metric
    for metric, color in metrics.items():
        if metric in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(iterations, df[metric], color=color, linewidth=2)

            ax.set_title(f'{metric.replace("_", " ").title()} Over Iterations - {filename.split("_")[0].title()} ({dataset_name})')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Count')
            ax.grid(True, alpha=0.3)

            # Set axis limits with extra padding for better visualization
            # Add more padding for unreachable_count and avg_connections
            if metric == 'avg_connections':
                max_val = df[metric].max()
                # Add 20% padding at the top for better visualization
                padding = max_val * 0.2
                ax.set_ylim(bottom=0, top=max_val + padding)
            else:
                y_padding = 0.4 if metric == 'unreachable_count' else 0.2
                set_axis_limits(ax, pd.Series(iterations), y_padding=y_padding)
                ax.set_ylim(bottom=0)

            save_plot(fig, save_dir, f"{filename}_{metric}", experiment_paths)
            plt.close()

def plot_level_connectivity(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]]):
    """Plot average connectivity by level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for connectivity
    level_cols = [col for col in df.columns if col.startswith('avg_conn_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # distinct colors for each level

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]  # Extract level number
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Node Connectivity')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    save_plot(fig, save_dir, f"{filename}_level_connectivity", experiment_paths)
    plt.close()

def plot_level_unreachable(df: pd.DataFrame,
                         title: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]]):
    """Plot unreachable points by level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for unreachable points
    level_cols = [col for col in df.columns if col.startswith('unreachable_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Unreachable Points')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    save_plot(fig, save_dir, f"{filename}_level_unreachable", experiment_paths)
    plt.close()

def plot_level_nodes(df: pd.DataFrame,
                   title: str,
                   save_dir: str,
                   filename: str,
                   experiment_paths: Dict[str, List[str]]):
    """Plot number of nodes by level over iterations."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create iteration sequence
    iterations = range(len(df))

    # Get level columns for node counts
    level_cols = [col for col in df.columns if col.startswith('nodes_l')]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Plot line for each level
    for col, color in zip(sorted(level_cols), colors):
        level = col.split('_l')[1]
        ax.plot(iterations, df[col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Number of Nodes')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    save_plot(fig, save_dir, f"{filename}_level_nodes", experiment_paths)
    plt.close()

def plot_level_distances(df: pd.DataFrame,
                         dataset_name: str,
                         title: str,
                         save_dir: str,
                         filename: str,
                         experiment_paths: Dict[str, List[str]]):
    """Plot average, minimum, and maximum distances by level over iterations."""
    setup_plot_style()
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Get level numbers
    level_nums = set()
    for col in df.columns:
        if col.startswith('avg_dist_l'):
            level_nums.add(col.split('_l')[1])
    level_nums = sorted(level_nums)

    # Create iteration sequence
    iterations = range(len(df))

    # 1. Average distances
    fig, ax = plt.subplots(figsize=(10, 6))
    for level, color in zip(level_nums, colors):
        avg_col = f'avg_dist_l{level}'
        ax.plot(iterations, df[avg_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Euclidean Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    save_plot(fig, save_dir, f"{filename}_level_avg_distances", experiment_paths)
    plt.close()

    # 2. Minimum distances
    fig, ax = plt.subplots(figsize=(10, 6))
    for level, color in zip(level_nums, colors):
        min_col = f'min_dist_l{level}'
        ax.plot(iterations, df[min_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(f'Minimum Distances between Neighbor Nodes by Level - {filename.title()} ({dataset_name})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Euclidean Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    save_plot(fig, save_dir, f"{filename}_level_min_distances", experiment_paths)
    plt.close()

    # 3. Maximum distances
    fig, ax = plt.subplots(figsize=(10, 6))
    for level, color in zip(level_nums, colors):
        max_col = f'max_dist_l{level}'
        ax.plot(iterations, df[max_col], label=f'Level {level}', color=color, linewidth=2)

    ax.set_title(f'Maximum Distances between Neighbor Nodes by Level - {filename.title()} ({dataset_name})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Euclidean Distance')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Set axis limits with extra padding for better visualization
    set_axis_limits(ax, pd.Series(iterations), y_padding=0.2)

    save_plot(fig, save_dir, f"{filename}_level_max_distances", experiment_paths)
    plt.close()

def plot_connectivity_scatter(df: pd.DataFrame,
                          search_df: pd.DataFrame,
                          title: str,
                          save_dir: str,
                          filename: str,
                          experiment_paths: Dict[str, List[str]]):
    """Plot scatter plot of connectivity vs search performance."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color gradient based on recall
    scatter = ax.scatter(df['avg_connections'],
                        search_df['mean_computed_distances'],
                        c=search_df['mean_recall'],
                        cmap='viridis',
                        alpha=0.6)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Recall', rotation=270, labelpad=15)

    ax.set_xlabel('Mean Node Connectivity')
    ax.set_ylabel('Number of Distances Computed at Search')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # Set axis limits without forcing x-axis to start at 0
    set_axis_limits(ax, df['avg_connections'], force_x_zero=False, y_padding=0.2)

    save_plot(fig, save_dir, f"{filename}_node_connectivity_vs_distances_computed", experiment_paths)
    plt.close()

def generate_memory_connectivity_plots(experiment_paths: Dict[str, List[str]]):
    """Generate all memory and connectivity related plots."""
    for scenario, paths in experiment_paths.items():
        # Create scenario-level directory for combined plots
        scenario_dir = os.path.dirname(paths[0])  # Get directory containing dataset paths
        scenario_images_dir = os.path.join(scenario_dir, 'images')
        os.makedirs(scenario_images_dir, exist_ok=True)

        # Generate individual dataset plots
        for dataset_path in paths:
            save_dir = os.path.join(dataset_path, 'images')
            os.makedirs(save_dir, exist_ok=True)

            dataset_name = os.path.basename(dataset_path)
            dataset_name = "fashion-mnist" if dataset_name.startswith("fashion_mnist") else dataset_name.split('_')[0]

            # Memory stats plots
            memory_stats_path = os.path.join(dataset_path, 'memory_stats.csv')
            if os.path.exists(memory_stats_path):
                memory_stats_df = load_csv_data(memory_stats_path)

                plot_slot_distribution(
                    memory_stats_df,
                    f'Slot Distribution - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}_memory_stats',
                    experiment_paths
                )

                plot_memory_usage_over_time(
                    memory_stats_df,
                    f'Memory Usage Over Iterations - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}_memory_stats',
                    experiment_paths
                )

            # Memory usage plots
            memory_path = os.path.join(dataset_path, 'memory_usage.csv')
            if os.path.exists(memory_path):
                memory_df = load_csv_data(memory_path)
                plot_memory_usage(
                    memory_df,
                    f'Memory Usage - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}_memory_usage',
                    experiment_paths
                )

            # Node connectivity plots
            connectivity_path = os.path.join(dataset_path, 'node_connectivity.csv')
            search_stats_path = os.path.join(dataset_path, 'search_query_stats.csv')

            if os.path.exists(connectivity_path):
                connectivity_df = load_csv_data(connectivity_path)
                dataset_name = os.path.basename(dataset_path)
                dataset_name = "fashion-mnist" if dataset_name.startswith("fashion_mnist") else dataset_name.split('_')[0]

                # Original connectivity plots
                plot_node_connectivity(
                    connectivity_df,
                    dataset_name,
                    save_dir,
                    f'{scenario}_node_connectivity',
                    experiment_paths
                )

                # New level-specific plots
                plot_level_connectivity(
                    connectivity_df,
                    f'Node Connectivity by Level - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths
                )

                plot_level_unreachable(
                    connectivity_df,
                    f'Unreachable Points by Level - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths
                )

                plot_level_nodes(
                    connectivity_df,
                    f'Number of Nodes by Level - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths
                )

                plot_level_distances(
                    connectivity_df,
                    dataset_name,
                    f'Distance between Neighbor Nodes by Level - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}',
                    experiment_paths
                )

                # Plot scatter and recall plots if search stats exist
                if os.path.exists(search_stats_path):
                    search_df = load_csv_data(search_stats_path)
                    dataset_name = search_df['dataset'].iloc[0]
                    if len(search_df) == len(connectivity_df):
                        # New scatter plot with mean recall colormap
                        plot_connectivity_scatter(
                            connectivity_df,
                            search_df,
                            f'Node Connectivity vs Distances Computed at Search - {scenario.title()} ({dataset_name})',
                            save_dir,
                            f'{scenario}',
                            experiment_paths
                        )

