"""Module for plotting search query statistics and early termination analysis."""
import os
from typing import Dict, List
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scripts.plots.plot_utils import (load_csv_data, calculate_error_bounds,
                        plot_with_error_bounds, setup_plot_style, save_plot, set_axis_limits)

def plot_early_termination_analysis(
    df: pd.DataFrame,
    save_dir: str,
    filename: str
) -> None:
    """Create multiple plots for early termination analysis using raw query data."""
    setup_plot_style()

    # Check if DataFrame is empty
    if df.empty:
        print(f"Warning: No data available for {filename}")
        return


    # Extract dataset name from the filename
    dataset_name = df['dataset'].iloc[0]

    # Check if required columns exist
    required_cols = ['iteration']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: Missing required columns for {filename}")
        return

    # Get max iteration for consistent x-axis limits
    max_iter = df['iteration'].max()
    min_iter = df['iteration'].min()

    # 1. Distribution of Early Termination Iterations
    fig, ax = plt.subplots(figsize=(10, 6))
    # Count number of early terminated queries per iteration
    early_term_counts = df.groupby('iteration').size()

    if not early_term_counts.empty:
        # Calculate number of iterations for tick spacing
        tick_spacing = max(1, max_iter // 10)  # One tick per 10 iterations, or 1 if less than 10

        # Create bar plot using matplotlib with lighter blue fill and darker blue edge
        ax.bar(early_term_counts.index,
               early_term_counts.values,
               color='#6495ED',  # Cornflower blue for fill
               edgecolor='#4169E1',  # Royal blue for edge
               linewidth=0.5,  # Subtle border
               width=1.0)
        ax.set_title(f'Distribution of Early Termination Iterations - {filename.title()} ({dataset_name})')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Early Terminated Queries')

        # Set x-axis to start at 0 and end exactly at max iteration
        ax.set_xlim(0, max_iter)

        # Set x-axis ticks with appropriate spacing
        ax.set_xticks(range(0, max_iter + 1, tick_spacing))

        # Ensure y-axis starts at 0 and ticks are not rotated
        ax.set_ylim(bottom=0)
        yticks = ax.get_yticks()
        if yticks[0] != 0:
            yticks = [0] + list(yticks)
            ax.set_yticks(yticks)
        ax.tick_params(axis='y', rotation=0)

        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        save_plot(fig, save_dir, f"{filename}_distribution")
    plt.close()

    # 2. Recall by Early Termination Iteration
    if 'recall' in df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))
        # Calculate mean recall per iteration
        recall_by_iter = df.groupby('iteration')['recall'].mean()
        if not recall_by_iter.empty:
            ax.plot(recall_by_iter.index, recall_by_iter.values, linewidth=2)
            ax.set_title(f'Recall by Iteration - {filename.title()} ({dataset_name})')
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean Recall')

            # Set x-axis to start at 0 and end at max iteration
            ax.set_xlim(0, max_iter)

            # Set x-axis ticks with appropriate spacing
            ax.set_xticks(range(0, max_iter + 1, tick_spacing))

            # Ensure y-axis starts at 0 and ticks are not rotated
            ax.set_ylim(bottom=0)
            yticks = ax.get_yticks()
            if yticks[0] != 0:
                yticks = [0] + list(yticks)
                ax.set_yticks(yticks)
            ax.tick_params(axis='y', rotation=0)

            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            save_plot(fig, save_dir, f"{filename}_recall_by_iter")
        plt.close()

    # 4. Scatter plots for key relationships
    scatter_cols = ['computed_distances', 'visited_members', 'recall']
    if any(col in df.columns for col in scatter_cols):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Early Termination Analysis: Key Relationships - {filename.title()} ({dataset_name})')

        # Computed distances vs Iteration
        if 'computed_distances' in df.columns:
            sns.scatterplot(data=df, x='iteration', y='computed_distances', ax=axes[0,0])
            axes[0,0].set_xlim(0, max_iter)
            axes[0,0].set_ylim(bottom=0)
            axes[0,0].set_xticks(range(0, max_iter + 1, tick_spacing))
            yticks = axes[0,0].get_yticks()
            if yticks[0] != 0:
                yticks = [0] + list(yticks)
                axes[0,0].set_yticks(yticks)
            axes[0,0].tick_params(axis='y', rotation=0)
            axes[0,0].set_ylabel('Euclidean Distance')
        axes[0,0].set_title(f'Computed Distances vs Iteration - {filename.title()} ({dataset_name})')

        # Visited members vs Iteration
        if 'visited_members' in df.columns:
            sns.scatterplot(data=df, x='iteration', y='visited_members', ax=axes[0,1])
            axes[0,1].set_xlim(0, max_iter)
            axes[0,1].set_ylim(bottom=0)
            axes[0,1].set_xticks(range(0, max_iter + 1, tick_spacing))
            yticks = axes[0,1].get_yticks()
            if yticks[0] != 0:
                yticks = [0] + list(yticks)
                axes[0,1].set_yticks(yticks)
            axes[0,1].tick_params(axis='y', rotation=0)
        axes[0,1].set_title(f'Visited Members vs Iteration - {filename.title()} ({dataset_name})')

        # Recall vs Computed Distances
        if 'recall' in df.columns and 'computed_distances' in df.columns:
            sns.scatterplot(data=df, x='computed_distances', y='recall', ax=axes[1,0])
            axes[1,0].set_ylim(bottom=0)
            yticks = axes[1,0].get_yticks()
            if yticks[0] != 0:
                yticks = [0] + list(yticks)
                axes[1,0].set_yticks(yticks)
            axes[1,0].tick_params(axis='y', rotation=0)
            axes[1,0].set_xlabel('Euclidean Distance')
        axes[1,0].set_title(f'Recall vs Computed Distances - {filename.title()} ({dataset_name})')

        # Recall vs Visited Members
        if 'recall' in df.columns and 'visited_members' in df.columns:
            sns.scatterplot(data=df, x='visited_members', y='recall', ax=axes[1,1])
            axes[1,1].set_ylim(bottom=0)
            yticks = axes[1,1].get_yticks()
            if yticks[0] != 0:
                yticks = [0] + list(yticks)
                axes[1,1].set_yticks(yticks)
            axes[1,1].tick_params(axis='y', rotation=0)
        axes[1,1].set_title(f'Recall vs Visited Members - {filename.title()} ({dataset_name})')

        # Rotate x-axis labels for all subplots
        for ax in axes.flat:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        plt.tight_layout()
        save_plot(fig, save_dir, f"{filename}_relationships")

def plot_visited_vs_computed(df: pd.DataFrame,
                      save_dir: str,
                      filename: str):
    """Plot the relationship between visited members and computed distances."""
    setup_plot_style()

    # Extract dataset name from the title
    dataset_name = df['dataset'].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter plot with color gradient
    scatter = ax.scatter(df['mean_visited_members'],
                        df['mean_computed_distances'],
                        c=df['mean_recall'],
                        cmap='viridis',
                        alpha=0.6)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Mean Recall', rotation=270, labelpad=15)

    ax.set_xlabel('Mean Number of Visited Members')
    ax.set_ylabel('Mean Number of Computed Distances')
    ax.set_title(f'Visited Members vs Computed Distances - {filename.split("_")[1].title()} ({dataset_name})')
    ax.grid(True)

    # Set axis limits without forcing x-axis to start at 0
    set_axis_limits(ax, df['mean_visited_members'], force_x_zero=False, y_padding=0.2)

    # Adjust layout
    fig.set_constrained_layout(True)
    save_plot(fig, save_dir, f"{filename}_visited_vs_computed")
    plt.close()

def plot_search_metric_over_time(df: pd.DataFrame,
                               metric: str,
                               title: str,
                               ylabel: str,
                               save_dir: str,
                               filename: str):
    """Plot a single search metric over time with mean and median."""
    try:
        # Validate inputs
        if df.empty:
            print(f"Warning: Empty DataFrame for {filename}")
            return

        # Check if required columns exist
        required_cols = ['iteration', f'mean_{metric}', f'median_{metric}']
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: Missing required columns for {filename}")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        setup_plot_style()

        # Plot mean with standard deviation error bounds if available
        if f'stddev_{metric}' in df.columns:
            lower_bound = df[f'mean_{metric}'] - df[f'stddev_{metric}']
            upper_bound = df[f'mean_{metric}'] + df[f'stddev_{metric}']
            ax.fill_between(df['iteration'], lower_bound, upper_bound,
                          alpha=0.2, color='blue', label='±1 Std Dev')

        # Plot mean line
        ax.plot(df['iteration'], df[f'mean_{metric}'],
               label=f'Mean {metric}',
               color='blue',
               linewidth=2)

        # Plot median line
        ax.plot(df['iteration'], df[f'median_{metric}'],
               label=f'Median {metric}',
               color='red',
               linestyle='--',
               linewidth=2)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Set y-axis to start at 0 and add padding at top if needed
        if metric == 'recall':
            ax.set_ylim(bottom=0, top=1.2)  # Set fixed y-axis range for recall plots
        else:
            # Add padding for other metrics but ensure bottom starts at 0
            set_axis_limits(ax, df['iteration'], y_padding=0.2)
            ax.set_ylim(bottom=0)

        save_plot(fig, save_dir, filename)
    except Exception as e:
        print(f"Error plotting search metric {metric} for {filename}: {str(e)}")

def generate_search_analysis_plots(experiment_paths: Dict[str, List[str]]):
    """Generate all search-related plots."""
    search_metrics = ['recall', 'computed_distances', 'visited_members']

    for scenario, paths in experiment_paths.items():
        for dataset_path in paths:
            save_dir = os.path.join(dataset_path, 'images')

            # Search query stats plots (aggregated data)
            search_stats_path = os.path.join(dataset_path, 'search_query_stats.csv')
            if os.path.exists(search_stats_path):
                search_df = load_csv_data(search_stats_path)
                dataset_name = search_df['dataset'].iloc[0]

                # Generate individual metric plots
                for metric in search_metrics:
                    plot_search_metric_over_time(
                        search_df,
                        metric,
                        f'Search {metric.replace("_", " ").title()} - {scenario.title()} ({dataset_name})',
                        metric.replace("_", " ").title(),
                        save_dir,
                        f'{scenario}_search_{metric}.png'
                    )

                plot_visited_vs_computed(
                    search_df,
                    save_dir,
                    f'{scenario}_search_efficiency'
                )

            # Early termination analysis (raw data)
            early_termination_path = os.path.join(dataset_path, 'early_terminated_queries.csv')
            if os.path.exists(early_termination_path):
                early_termination_df = load_csv_data(early_termination_path)
                plot_early_termination_analysis(
                    early_termination_df,
                    save_dir,
                    f'{scenario}_early_termination'
                )
