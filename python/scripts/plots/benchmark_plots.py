"""Module for plotting benchmark data from HNSW index experiments."""

import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.plots.plot_utils import (load_csv_data, calculate_error_bounds,
                        plot_with_error_bounds, setup_plot_style, save_plot, set_axis_limits)

def plot_benchmark_metrics(df: pd.DataFrame,
                         metric: str,
                         title: str,
                         ylabel: str,
                         save_dir: str,
                         filename: str):
    """Plot benchmark metrics with error bounds."""
    try:
        # Validate inputs
        if df.empty:
            print(f"Warning: Empty DataFrame for {filename}")
            return

        # Check if required columns exist
        required_cols = ['iteration', f'mean_{metric}']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns for {filename}")
            return

        fig, ax = plt.subplots()
        setup_plot_style()

        # Calculate error bounds
        try:
            lower_bound, upper_bound = calculate_error_bounds(
                df,
                f'mean_{metric}',
                std_col=f'stddev_{metric}'
            )
        except ValueError as e:
            print(f"Warning: Could not calculate error bounds for {filename}: {str(e)}")
            return

        # Plot with error bounds for mean
        plot_with_error_bounds(
            ax,
            df['iteration'],
            df[f'mean_{metric}'],
            lower_bound,
            upper_bound,
            label=f'Mean {metric}'
        )

        # Add median line if median column exists
        if f'median_{metric}' in df.columns:
            ax.plot(df['iteration'], df[f'median_{metric}'],
                   label=f'Median {metric}',
                   color='red',
                   linestyle='--',
                   linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

        # Set axis limits without extra space
        set_axis_limits(ax, df['iteration'])

        save_plot(fig, save_dir, filename)
    except Exception as e:
        print(f"Error plotting benchmark metrics for {filename}: {str(e)}")

def plot_benchmark_correlations(dfs: Dict[str, pd.DataFrame],
                             title: str,
                             save_dir: str,
                             filename: str):
    """Plot mean and median times for add, search, and delete operations over iterations."""
    try:
        # Validate inputs
        if not dfs or len(dfs) == 0:
            print(f"Warning: No data provided for {filename}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        setup_plot_style()

        # Colors for different operations
        colors = {
            'bm_add.csv': 'blue',
            'bm_search.csv': 'green',
            'bm_delete.csv': 'red'
        }

        # Plot mean and median time for each operation
        for bm_file, df in dfs.items():
            if df is None or df.empty:
                continue

            operation = bm_file.replace('.csv', '').replace('bm_', '')
            base_color = colors.get(bm_file, 'gray')

            # Plot mean time
            if 'iteration' in df.columns and 'mean_time' in df.columns:
                ax.plot(df['iteration'],
                       df['mean_time'],
                       label=f'{operation.title()} Mean',
                       color=base_color,
                       linewidth=2)

            # Plot median time with dashed line
            if 'iteration' in df.columns and 'median_time' in df.columns:
                ax.plot(df['iteration'],
                       df['median_time'],
                       label=f'{operation.title()} Median',
                       color=base_color,
                       linestyle='--',
                       linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set axis limits without extra space
        # Use the maximum iteration across all dataframes
        max_iter = max(df['iteration'].max() for df in dfs.values() if df is not None and not df.empty)
        set_axis_limits(ax, pd.Series(range(max_iter + 1)))

        save_plot(fig, save_dir, filename)
    except Exception as e:
        print(f"Error plotting benchmark correlations for {filename}: {str(e)}")

def generate_benchmark_plots(experiment_paths: Dict[str, List[str]]):
    """Generate all benchmark-related plots."""
    metrics = ['time']
    benchmark_files = ['bm_add.csv', 'bm_delete.csv', 'bm_search.csv']

    for scenario, paths in experiment_paths.items():
        for dataset_path in paths:
            save_dir = os.path.join(dataset_path, 'images')

            # Extract dataset name from the folder path
            dataset_name = os.path.basename(dataset_path)
            # Handle special case for fashion-mnist
            if dataset_name.startswith("fashion_mnist"):
                dataset_name = "fashion-mnist"
            else:
                dataset_name = dataset_name.split('_')[0]

            # Load benchmark data
            benchmark_dfs = {}
            for bm_file in benchmark_files:
                filepath = os.path.join(dataset_path, bm_file)
                if os.path.exists(filepath):
                    try:
                        benchmark_dfs[bm_file] = load_csv_data(filepath)
                    except Exception as e:
                        print(f"Error loading {bm_file}: {str(e)}")
                        continue

            # Generate individual benchmark plots
            for bm_file, df in benchmark_dfs.items():
                for metric in metrics:
                    plot_benchmark_metrics(
                        df,
                        metric,
                        f'{bm_file.replace(".csv", "").split("_")[1].title()} {metric.title()} - {scenario.title()} ({dataset_name})',
                        f'{metric.title()} (seconds)',
                        save_dir,
                        f'{scenario}_{bm_file.replace(".csv", "")}_{metric}.png'
                    )

            # Generate comparison plot with all operations
            if benchmark_dfs:
                plot_benchmark_correlations(
                    benchmark_dfs,
                    f'Operation Times Comparison - {scenario.title()} ({dataset_name})',
                    save_dir,
                    f'{scenario}_bm_times_comparison.png'
                )
