"""Utility functions for plotting HNSW index experiment results."""

import os
import matplotlib
# Force matplotlib to use the Agg backend
matplotlib.use('Agg')
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colorbar import Colorbar

def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load CSV data and handle missing iteration column."""
    df = pd.read_csv(filepath)
    if 'iteration' not in df.columns:
        df['iteration'] = range(len(df))
    return df

def calculate_error_bounds(df: pd.DataFrame,
                         mean_col: str,
                         std_col: str = None) -> Tuple[pd.Series, pd.Series]:
    """Calculate error bounds using standard deviation if available."""
    if std_col and std_col in df.columns:
        # Use standard deviation for error bounds
        lower_bound = df[mean_col] - df[std_col]
        upper_bound = df[mean_col] + df[std_col]
    else:
        raise ValueError("Standard deviation column not available.")
    return lower_bound, upper_bound

def plot_with_error_bounds(ax: plt.Axes,
                         x: pd.Series,
                         y: pd.Series,
                         lower_bound: pd.Series,
                         upper_bound: pd.Series,
                         label: str,
                         color: str = 'blue'):
    """Plot data with error bounds."""
    # Plot error bounds
    ax.fill_between(x, lower_bound, upper_bound,
                   alpha=0.2, color=color, label=f'{label} ±1 Std Dev')
    # Plot mean line
    ax.plot(x, y, label=label, color=color, linewidth=2)

def setup_plot_style():
    """Set up the plot style for publication-quality figures."""
    plt.style.use('seaborn-paper')

    # Use Times New Roman with fallbacks
    font_family = ['Times New Roman', 'DejaVu Serif', 'Serif']
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = font_family

    # Font sizes
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 9
    plt.rcParams['ytick.labelsize'] = 9
    plt.rcParams['legend.fontsize'] = 9

    # Figure size and DPI
    plt.rcParams['figure.figsize'] = [8, 6]
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300

    # Line widths and styles
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['grid.linewidth'] = 0.6
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['lines.markersize'] = 6

    # Grid settings
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    # Remove default margins
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.ymargin'] = 0.1
    plt.rcParams['figure.constrained_layout.use'] = True

    # Color cycle - colorblind friendly
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#0077BB',  # Blue
        '#EE7733',  # Orange
        '#009988',  # Teal
        '#CC3311',  # Red
        '#33BBEE',  # Cyan
        '#EE3377',  # Magenta
        '#BBBBBB',  # Gray
    ])

def set_axis_limits(ax: plt.Axes, x_data: pd.Series, y_padding: float = 0.0, force_x_zero: bool = True):
    """Set axis limits without extra space.

    Args:
        ax: The matplotlib axes to modify
        x_data: The x-axis data
        y_padding: Additional padding to add to the top of the y-axis (as a fraction of the y-range)
        force_x_zero: Whether to force x-axis to start at 0
    """
    # Set x-axis limits
    if force_x_zero:
        ax.set_xlim(0, x_data.max())
    else:
        x_min, x_max = x_data.min(), x_data.max()
        x_range = x_max - x_min
        x_margin = x_range * 0.05  # 5% margin
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    # Set y-axis to start at 0 with optional padding
    y_min, y_max = ax.get_ylim()
    if y_padding > 0:
        y_range = y_max - y_min
        y_max = y_max + (y_range * y_padding)
    ax.set_ylim(0, y_max)

    # Ensure 0 is shown as first tick and no rotation
    yticks = ax.get_yticks()
    if yticks[0] != 0:
        yticks = [0] + list(yticks)
        ax.set_yticks(yticks)

    # Ensure y-axis tick labels are not rotated
    ax.tick_params(axis='y', rotation=0)

def create_combined_plot(plot_files: List[str],
                     scenario: str,
                     plot_type: str,
                     save_dir: str):
    """Create a combined plot from individual plot files.

    Args:
        plot_files: List of paths to individual plot files
        scenario: Name of the scenario
        plot_type: Type/name of the plot (used in title and filename)
        save_dir: Directory to save the combined plot
    """
    if not plot_files:
        return

    # Calculate grid dimensions
    num_plots = len(plot_files)
    n_cols = min(2, num_plots)  # Use 2 columns unless only 1 plot
    n_rows = (num_plots + 1) // 2  # Round up for odd numbers

    # Create figure with gridspec for more control over spacing
    fig = plt.figure(figsize=(15 * n_cols/2, 6 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.2)

    # Add overall title
    fig.suptitle(f'{scenario} - {plot_type}', fontsize=16, y=0.95)

    # Create subplot for each plot
    for idx, plot_file in enumerate(sorted(plot_files)):
        # Get dataset name from plot file path
        dataset_name = os.path.basename(os.path.dirname(os.path.dirname(plot_file)))

        # Create subplot using gridspec
        row = idx // n_cols
        col = idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Load and display image
        img = plt.imread(plot_file)
        ax.imshow(img)
        ax.axis('off')  # Hide axes
        ax.set_title(dataset_name, pad=10)  # Reduce padding between title and plot

    # Save combined plot
    combined_filename = f'{scenario}_combined_{plot_type}.png'
    plt.savefig(os.path.join(save_dir, combined_filename),
                bbox_inches='tight',
                dpi=300,
                pad_inches=0.2)  # Reduce padding around the entire figure
    plt.close()

def combine_scenario_plots(experiment_paths: Dict[str, List[str]]):
    """Combine plots from all datasets in each scenario.

    Args:
        experiment_paths: Dictionary mapping scenario names to lists of dataset paths
    """
    for scenario, dataset_paths in experiment_paths.items():
        # Create scenario-level directory for combined plots if it doesn't exist
        scenario_dir = os.path.dirname(dataset_paths[0])
        scenario_images_dir = os.path.join(scenario_dir, 'images')
        os.makedirs(scenario_images_dir, exist_ok=True)

        # Get all unique plot types from the first dataset's images directory
        first_dataset_images = os.path.join(dataset_paths[0], 'images')
        if not os.path.exists(first_dataset_images):
            continue

        plot_types = set()
        for filename in os.listdir(first_dataset_images):
            if filename.endswith('.png'):
                # Extract the plot type from the filename
                # Assuming format: scenario_plottype.png or scenario_plottype_suffix.png
                parts = filename.replace('.png', '').split('_')
                if len(parts) > 1:
                    plot_type = '_'.join(parts[1:])  # Join all parts after scenario
                    plot_types.add(plot_type)

        # For each plot type, collect corresponding plots from all datasets
        for plot_type in plot_types:
            plot_files = []
            for dataset_path in dataset_paths:
                images_dir = os.path.join(dataset_path, 'images')
                if not os.path.exists(images_dir):
                    continue

                # Look for matching plot file
                for filename in os.listdir(images_dir):
                    if filename.endswith('.png') and plot_type in filename:
                        plot_files.append(os.path.join(images_dir, filename))
                        break  # Take the first matching file

            if plot_files:
                create_combined_plot(plot_files, scenario, plot_type, scenario_images_dir)

def save_plot(fig: plt.Figure,
             save_dir: str,
             filename: str,
             experiment_paths: Optional[Dict[str, List[str]]] = None,
             dpi: int = 300) -> None:
    """Save a plot with publication-quality settings.

    Args:
        fig: The matplotlib figure to save
        save_dir: Directory to save the plot in
        filename: Name of the plot file (without extension)
        experiment_paths: Optional dictionary mapping scenarios to dataset paths
        dpi: Resolution in dots per inch
    """
    try:
        # Validate inputs
        if fig is None:
            print("Error: Figure is None")
            return

        if not save_dir or not filename:
            print("Error: save_dir and filename must be provided")
            return

        # Create main directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Create PDF subdirectory
        pdf_dir = os.path.join(save_dir, "pdf")
        os.makedirs(pdf_dir, exist_ok=True)

        # Remove any existing extensions from the filename
        base_name = os.path.splitext(filename)[0]
        base_path = os.path.join(save_dir, base_name)
        pdf_path = os.path.join(pdf_dir, base_name)
        png_path = f"{base_path}.png"

        # Check if figure has any content
        if not fig.get_axes():
            print(f"Warning: Figure has no axes, skipping save for {filename}")
            return

        # Adjust layout based on whether the figure has a colorbar
        has_colorbar = any(isinstance(c, Colorbar) for c in fig.get_children())
        if has_colorbar:
            # For plots with colorbars, use tight_layout with minimal padding
            fig.tight_layout(pad=0.5)
        else:
            # For regular plots, use constrained_layout with minimal padding
            fig.set_constrained_layout(True)
            fig.set_constrained_layout_pads(w_pad=0.5, h_pad=0.5)

        # Save as PNG with minimal padding
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight', pad_inches=0.05)

        # Save as PDF (vector format) with minimal padding
        fig.savefig(f"{pdf_path}.pdf", format='pdf', bbox_inches='tight', pad_inches=0.05)

        print(f"Plot saved successfully as PNG and PDF: {os.path.basename(png_path)}")

    except Exception as e:
        print(f"Error saving plot {filename}: {str(e)}")
    finally:
        plt.close(fig)  # Close the figure to free memory

def get_experiment_paths(base_dir: str) -> Dict[str, List[str]]:
    """Get paths for all experiment data files."""
    experiments = ['fullcoverage', 'newdata', 'random']
    paths = {}

    for exp in experiments:
        exp_dir = os.path.join(base_dir, exp)
        if not os.path.exists(exp_dir):
            continue

        paths[exp] = []
        for dataset_dir in os.listdir(exp_dir):
            dataset_path = os.path.join(exp_dir, dataset_dir)
            if os.path.isdir(dataset_path):
                paths[exp].append(dataset_path)

    return paths
