import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Add these functions to your existing script

def load_data(csv_filepath):
    """Load data from CSV file"""
    return pd.read_csv(csv_filepath)

def create_normalized_comparison(datasets_data, output_dir="./plots/comparison/"):
    """Create normalized line graphs to compare trends regardless of scale"""
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
    
    # For each metric, create normalized plots
    for metric in metrics:
        # 1. Min-Max Normalized Plot (0-1 scale)
        plt.figure(figsize=(12, 7))
        
        for i, (dataset_name, data) in enumerate(datasets_data.items()):
            sorted_data = data.sort_values('iteration')
            # Min-max normalization
            min_val = sorted_data[metric].min()
            max_val = sorted_data[metric].max()
            normalized = (sorted_data[metric] - min_val) / (max_val - min_val) if max_val > min_val else sorted_data[metric]
            
            plt.plot(sorted_data['iteration'], normalized, 
                    marker=['o', 's', '^', 'v', 'D'][i % 5], 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    color=colors[i],
                    label=f"{dataset_name} (min={min_val:.3f}, max={max_val:.3f})")
        
        plt.xlabel('Iteration')
        plt.ylabel(f'Normalized {metric.replace("_", " ").title()} (0-1)')
        plt.title(f'Min-Max Normalized {metric.replace("_", " ").title()} Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"{output_dir}normalized_{metric}_minmax.png", dpi=300)
        plt.close()
        
        # 2. Z-Score Normalized Plot
        plt.figure(figsize=(12, 7))
        
        for i, (dataset_name, data) in enumerate(datasets_data.items()):
            sorted_data = data.sort_values('iteration')
            # Z-score normalization
            mean_val = sorted_data[metric].mean()
            std_val = sorted_data[metric].std()
            z_scores = (sorted_data[metric] - mean_val) / std_val if std_val > 0 else sorted_data[metric]
            
            plt.plot(sorted_data['iteration'], z_scores, 
                    marker=['o', 's', '^', 'v', 'D'][i % 5], 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    color=colors[i],
                    label=f"{dataset_name} (mean={mean_val:.3f}, std={std_val:.3f})")
        
        plt.xlabel('Iteration')
        plt.ylabel(f'{metric.replace("_", " ").title()} (Standard Deviations from Mean)')
        plt.title(f'Z-Score Normalized {metric.replace("_", " ").title()} Comparison')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"{output_dir}normalized_{metric}_zscore.png", dpi=300)
        plt.close()
        
        # 3. Percent Change from First Iteration
        plt.figure(figsize=(12, 7))
        
        for i, (dataset_name, data) in enumerate(datasets_data.items()):
            sorted_data = data.sort_values('iteration')
            # Percent change from first value
            first_val = sorted_data[metric].iloc[0]
            pct_change = (sorted_data[metric] - first_val) / first_val * 100 if first_val != 0 else sorted_data[metric]
            
            plt.plot(sorted_data['iteration'], pct_change, 
                    marker=['o', 's', '^', 'v', 'D'][i % 5], 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    color=colors[i],
                    label=f"{dataset_name} (start={first_val:.3f})")
        
        plt.xlabel('Iteration')
        plt.ylabel(f'Percent Change in {metric.replace("_", " ").title()} (%)')
        plt.title(f'Percent Change in {metric.replace("_", " ").title()} from First Iteration')
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig(f"{output_dir}pct_change_{metric}.png", dpi=300)
        plt.close()
    
    print("Generated normalized comparison graphs for all metrics")

def create_dual_axis_comparison(datasets_data, output_dir="./plots/comparison/"):
    """Create dual-axis plots to compare trends with different scales"""
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
    
    # For each metric, create dual-axis plots (works best for comparing two datasets)
    for metric in metrics:
        if len(datasets_data) == 2:  # Dual axis works best with exactly 2 datasets
            plt.figure(figsize=(12, 7))
            
            # Get the dataset names
            dataset_names = list(datasets_data.keys())
            
            # First dataset (left axis)
            sorted_data1 = datasets_data[dataset_names[0]].sort_values('iteration')
            ax1 = plt.gca()
            line1 = ax1.plot(sorted_data1['iteration'], sorted_data1[metric], 
                    marker='o', 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    color='blue',
                    label=f"{dataset_names[0]}")
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel(f'{metric.replace("_", " ").title()} - {dataset_names[0]}', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Second dataset (right axis)
            sorted_data2 = datasets_data[dataset_names[1]].sort_values('iteration')
            ax2 = ax1.twinx()
            line2 = ax2.plot(sorted_data2['iteration'], sorted_data2[metric], 
                    marker='s', 
                    linestyle='-', 
                    linewidth=2, 
                    markersize=8,
                    color='red',
                    label=f"{dataset_names[1]}")
            ax2.set_ylabel(f'{metric.replace("_", " ").title()} - {dataset_names[1]}', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Combine legends
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper center')
            
            plt.title(f'Dual-Axis Comparison of {metric.replace("_", " ").title()}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}dual_axis_{metric}.png", dpi=300)
            plt.close()
        
        else:
            print("Dual-axis plots work best with exactly 2 datasets")
    
    print("Generated dual-axis comparison graphs for all metrics")

def calculate_trend_correlation(datasets_data, output_dir="./plots/comparison/"):
    """Calculate and visualize correlation between dataset trends"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Define metrics to analyze
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
    
    # Extract dataset names
    dataset_names = list(datasets_data.keys())
    
    # Create a table to store correlation coefficients
    correlation_results = {}
    
    # For each metric, calculate correlations between datasets
    for metric in metrics:
        correlation_results[metric] = {}
        
        # Calculate correlation for each dataset pair
        for i in range(len(dataset_names)):
            for j in range(i+1, len(dataset_names)):
                dataset1 = dataset_names[i]
                dataset2 = dataset_names[j]
                
                # Get data for each dataset
                data1 = datasets_data[dataset1].sort_values('iteration')[metric]
                data2 = datasets_data[dataset2].sort_values('iteration')[metric]
                
                # Calculate correlation if both datasets have the same length
                if len(data1) == len(data2):
                    correlation = np.corrcoef(data1, data2)[0, 1]
                    correlation_results[metric][f"{dataset1} vs {dataset2}"] = correlation
                else:
                    # Handle different lengths (use common iterations)
                    common_iterations = set(datasets_data[dataset1]['iteration']).intersection(
                        set(datasets_data[dataset2]['iteration']))
                    
                    if common_iterations:
                        data1_common = datasets_data[dataset1][
                            datasets_data[dataset1]['iteration'].isin(common_iterations)][metric]
                        data2_common = datasets_data[dataset2][
                            datasets_data[dataset2]['iteration'].isin(common_iterations)][metric]
                        
                        correlation = np.corrcoef(data1_common, data2_common)[0, 1]
                        correlation_results[metric][f"{dataset1} vs {dataset2}"] = correlation
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 10))
    
    # Prepare data for heatmap
    metrics_list = []
    comparisons_list = []
    correlations_list = []
    
    for metric in correlation_results:
        for comparison in correlation_results[metric]:
            metrics_list.append(metric)
            comparisons_list.append(comparison)
            correlations_list.append(correlation_results[metric][comparison])
    
    # Create DataFrame for heatmap
    heatmap_df = pd.DataFrame({
        'Metric': metrics_list,
        'Comparison': comparisons_list,
        'Correlation': correlations_list
    })
    
    # Pivot for heatmap format
    pivot_df = heatmap_df.pivot(index='Metric', columns='Comparison', values='Correlation')
    
    # Create heatmap
    sns.heatmap(pivot_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, 
                linewidths=.5, cbar_kws={'label': 'Correlation Coefficient'})
    
    plt.title('Trend Correlation Between Datasets')
    plt.tight_layout()
    plt.savefig(f"{output_dir}trend_correlation_heatmap.png", dpi=300)
    plt.close()
    
    # Create a scatter plot for each dataset pair for the mean_recall metric
    if 'mean_recall' in metrics:
        for i in range(len(dataset_names)):
            for j in range(i+1, len(dataset_names)):
                dataset1 = dataset_names[i]
                dataset2 = dataset_names[j]
                
                # Get data
                data1 = datasets_data[dataset1].sort_values('iteration')['mean_recall']
                data2 = datasets_data[dataset2].sort_values('iteration')['mean_recall']
                
                # Check if datasets have the same number of points
                if len(data1) == len(data2):
                    plt.figure(figsize=(10, 8))
                    plt.scatter(data1, data2, alpha=0.7, s=100)
                    
                    # Add iteration labels to points
                    for k, (x, y) in enumerate(zip(data1, data2)):
                        plt.annotate(str(k+1), (x, y), xytext=(5, 5), textcoords='offset points')
                    
                    # Calculate and add trendline
                    z = np.polyfit(data1, data2, 1)
                    p = np.poly1d(z)
                    plt.plot(data1, p(data1), "r--", alpha=0.7)
                    
                    # Add correlation coefficient to plot
                    corr = np.corrcoef(data1, data2)[0, 1]
                    plt.annotate(f"Correlation: {corr:.3f}", xy=(0.05, 0.95), 
                                 xycoords='axes fraction', fontsize=12)
                    
                    plt.xlabel(f"{dataset1} Mean Recall")
                    plt.ylabel(f"{dataset2} Mean Recall")
                    plt.title(f"Correlation Between {dataset1} and {dataset2} Mean Recall")
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{output_dir}correlation_scatter_{dataset1}_vs_{dataset2}.png", dpi=300)
                    plt.close()
    
    print("Generated correlation analysis visualizations")

# Add these function calls to your main function:
def main():
    csv_filepath = "data/results/" # Update this to your file path
    datasets = ["fashion_mnist", "mnist"]
    
    # Dictionary to store data from each dataset for comparison
    all_datasets_data = {}
    # After processing all datasets, create comparison visualizations
    for dataset in datasets:
        dataset_filepath = f"{csv_filepath}{dataset}_merged_report.csv"
        try:
            # Load the data
            data = load_data(dataset_filepath)
            
            # Add dataset name as a column
            data['dataset'] = dataset
            
            # Create line graph visualizations and store data for comparison
            all_datasets_data[dataset] = data
            
            print(f"Visualizations for {dataset} created successfully!")
        except Exception as e:
            print(f"An error occurred with {dataset}: {e}")

    if len(all_datasets_data) > 1:
        try:
            
            # Add these new visualization functions
            create_normalized_comparison(all_datasets_data)
            create_dual_axis_comparison(all_datasets_data)
            calculate_trend_correlation(all_datasets_data)
            
            print("Comparison visualizations created successfully!")
        except Exception as e:
            print(f"An error occurred during comparison: {e}")


if __name__ == "__main__":
    main()