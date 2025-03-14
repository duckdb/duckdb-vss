import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams.update({'font.size': 12})

# Function to load the CSV data
def load_data(csv_filepath):
    """Load data from CSV file"""
    return pd.read_csv(csv_filepath)

# Function to create individual line graphs for each metric
def create_line_graphs(data, dataset_name, output_dir="./plots/"):
    """Create separate line graphs for each metric in the dataset"""
    # Create output directory if it doesn't exist
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define metrics to plot (excluding dataset, iteration, and num_queries)
    metrics = [
        'mean_recall',
        'median_recall', 
        'stddev_recall',
        'var_recall',
        'min_recall',
        'max_recall',
        'p25_recall',
        'p75_recall',
        'p95_recall'
    ]
    
    # Create a separate line graph for each metric
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Create line graph with markers
        plt.plot(data['iteration'], data[metric], 
                marker='o', linestyle='-', linewidth=2, markersize=8)
        
        # Add labels and title
        plt.xlabel('Iteration')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs Iteration - {dataset_name}')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Ensure x-axis shows all iteration points
        plt.xticks(data['iteration'])
        
        # Add tight layout and save
        plt.tight_layout()
        plt.savefig(f"{dataset_dir}/{dataset_name}_{metric}_line.png", dpi=300)
        plt.close()
    
    print(f"Generated line graphs for {dataset_name}")
    return data  # Return the data for potential later use in comparison

# Function to compare the same metric across multiple datasets
def compare_datasets(datasets_data, output_dir="./plots/comparison/"):
    """Create line graphs comparing the same metric across different datasets"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to plot
    metrics = [
        'mean_recall',
        'median_recall', 
        'stddev_recall',
        'var_recall',
        'min_recall',
        'max_recall',
        'p25_recall',
        'p75_recall',
        'p95_recall'
    ]
    
    # Set up colors for different datasets
    colors = plt.cm.tab10(range(len(datasets_data)))
    
    # For each metric, create a plot comparing across datasets
    for metric in metrics:
        plt.figure(figsize=(12, 7))
        
        # Plot each dataset with a different color and marker
        for i, (dataset_name, data) in enumerate(datasets_data.items()):
            # Sort by iteration to ensure proper line connections
            sorted_data = data.sort_values('iteration')
            
            plt.plot(sorted_data['iteration'], sorted_data[metric], 
                    marker=['o', 's', '^', 'v', 'D', '*', 'x', '+', '.'][i % 9], 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    color=colors[i],
                    label=dataset_name)
        
        # Add labels and title
        plt.xlabel('Iteration')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Comparison of {metric.replace("_", " ").title()} Across Datasets')
        
        # Add grid for better readability
        plt.grid(True, alpha=0.3)
        
        # Add legend
        plt.legend(loc='best')
        
        # Add tight layout and save
        plt.tight_layout()
        plt.savefig(f"{output_dir}comparison_{metric}.png", dpi=300)
        plt.close()
    
    print(f"Generated comparison graphs for all metrics")

# Main function to run the script
def main():
    # File path to the CSV data
    csv_filepath = "data/results/"  # Update this to your file path
    datasets = ["fashion_mnist", "mnist"]
    
    # Dictionary to store data from each dataset for comparison
    all_datasets_data = {}
    
    for dataset in datasets:
        dataset_filepath = f"{csv_filepath}{dataset}_merged_report.csv"
        try:
            # Load the data
            data = load_data(dataset_filepath)
            
            # Add dataset name as a column
            data['dataset'] = dataset
            
            # Create line graph visualizations and store data for comparison
            all_datasets_data[dataset] = create_line_graphs(data, dataset)
            
            print(f"Visualizations for {dataset} created successfully!")
            
        except Exception as e:
            print(f"An error occurred with {dataset}: {e}")
    
    # After processing all datasets, create comparison visualizations
    if len(all_datasets_data) > 1:
        try:
            compare_datasets(all_datasets_data)
            print("Comparison visualizations created successfully!")
        except Exception as e:
            print(f"An error occurred during comparison: {e}")

if __name__ == "__main__":
    main()