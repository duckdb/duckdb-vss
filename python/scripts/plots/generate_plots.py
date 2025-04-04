"""Main script for generating all plots from HNSW index experiments."""

import os
from scripts.plots.plot_utils import get_experiment_paths, combine_scenario_plots
from scripts.plots.benchmark_plots import generate_benchmark_plots
from scripts.plots.memory_connectivity_plots import generate_memory_connectivity_plots
from scripts.plots.search_analysis_plots import generate_search_analysis_plots

def main():
    """Generate all plots for the HNSW index experiments."""
    # Get the base directory for experiment results
    # Go up 2 levels from the script location to reach the python directory
    python_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    # Then go up one more level and into embedded-c++/usearch/results
    base_dir = os.path.join(os.path.dirname(python_dir), "embedded-c++", "usearch", "results")

    # Get paths to experiment results
    experiment_paths = get_experiment_paths(base_dir)

    print("Generating benchmark plots...")
    generate_benchmark_plots(experiment_paths)

    print("Generating memory and connectivity plots...")
    generate_memory_connectivity_plots(experiment_paths)

    print("Generating search analysis plots...")
    generate_search_analysis_plots(experiment_paths)

    print("Generating combined plots for all scenarios...")
    combine_scenario_plots(experiment_paths)

    print("All plots generated successfully!")

if __name__ == "__main__":
    main()
