import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_recall_correlations(data, dataset_name, output_dir="./plots/correlations/"):
    """
    Analyze and visualize correlations between recall metrics and network structure metrics.
    
    Parameters:
    data (DataFrame): DataFrame containing both recall and network metrics
    dataset_name (str): Name of the dataset for plot titles and filenames
    output_dir (str): Directory to save the plots
    """
    # Create output directory if it doesn't exist
    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Define recall metrics
    recall_metrics = [
        'mean_recall',
        'median_recall', 
    ]
    
    # Define network structure metrics
    network_metrics = [
        'nodes_count',
        'unreachable_count',
        'orphaned_count',
        'avg_connections',
        'disconnected_nodes',
        'avg_conn_l0', 'avg_conn_l1', 'avg_conn_l2', 'avg_conn_l3', 'avg_conn_l4',
        'unreachable_l0', 'unreachable_l1', 'unreachable_l2', 'unreachable_l3', 'unreachable_l4',
    ]
    
    # Calculate correlation matrix
    correlation_columns = recall_metrics + network_metrics
    correlation_matrix = data[correlation_columns].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(14, 12))
    mask = np.zeros_like(correlation_matrix)
    mask[np.triu_indices_from(mask)] = True  # Mask upper triangle for clarity
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
                center=0, linewidths=0.5, fmt='.3f', mask=mask)
    
    plt.title(f'Correlation Matrix of Recall and Network Metrics - {dataset_name}')
    plt.tight_layout()
    plt.savefig(f"{dataset_dir}/correlation_matrix.png", dpi=300)
    plt.close()
    
    # Create scatterplots between mean_recall and each network metric
    for network_metric in network_metrics:
        plt.figure(figsize=(10, 6))
        
        # Calculate correlation for title
        corr = data['mean_recall'].corr(data[network_metric])
        
        # Create scatter plot with regression line
        sns.regplot(x=network_metric, y='mean_recall', data=data, 
                    scatter_kws={'alpha':0.7, 's':100}, 
                    line_kws={'color':'red'})
        
        # Add iteration labels to points
        for i, point in data.iterrows():
            plt.annotate(f"Iter {point['iteration']}", 
                        (point[network_metric], point['mean_recall']),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'Mean Recall vs {network_metric.replace("_", " ").title()} (r={corr:.3f}) - {dataset_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{dataset_dir}/mean_recall_vs_{network_metric}.png", dpi=300)
        plt.close()
    
    # Create scatterplots between unreachable_count and each recall metric
    for recall_metric in recall_metrics:
        plt.figure(figsize=(10, 6))
        
        # Calculate correlation for title
        corr = data[recall_metric].corr(data['unreachable_count'])
        
        # Create scatter plot with regression line
        sns.regplot(x='unreachable_count', y=recall_metric, data=data, 
                    scatter_kws={'alpha':0.7, 's':100}, 
                    line_kws={'color':'red'})
        
        # Add iteration labels to points
        for i, point in data.iterrows():
            plt.annotate(f"Iter {point['iteration']}", 
                        (point['unreachable_count'], point[recall_metric]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.title(f'{recall_metric.replace("_", " ").title()} vs Unreachable Count (r={corr:.3f}) - {dataset_name}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{dataset_dir}/{recall_metric}_vs_unreachable.png", dpi=300)
        plt.close()
    
    # Create a multi-line plot showing the trend of recall and unreachable_count over iterations
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot mean_recall on left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Mean Recall', color=color)
    ax1.plot(data['iteration'], data['mean_recall'], color=color, marker='o', linewidth=2)
    ax1.tick_params(axis='y', labelcolor=color)
    
    # Create second y-axis for unreachable_count
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Unreachable Count', color=color)
    ax2.plot(data['iteration'], data['unreachable_count'], color=color, marker='s', linewidth=2)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Calculate correlation for title
    corr = data['mean_recall'].corr(data['unreachable_count'])
    plt.title(f'Mean Recall and Unreachable Count Over Iterations (r={corr:.3f}) - {dataset_name}')
    plt.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"{dataset_dir}/recall_unreachable_trend.png", dpi=300)
    plt.close()
    
    # Create top correlations summary table
    plt.figure(figsize=(10, 8))
    
    # Extract correlations between recall metrics and network metrics
    recall_network_corr = correlation_matrix.loc[recall_metrics, network_metrics]
    
    # Flatten and get top absolute correlations
    corr_data = []
    for recall_metric in recall_metrics:
        for network_metric in network_metrics:
            corr_value = correlation_matrix.loc[recall_metric, network_metric]
            corr_data.append((recall_metric, network_metric, corr_value))
    
    # Sort by absolute correlation value
    corr_data.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Create table for top 10 correlations
    top_corr = corr_data[:10]
    table_data = [
        [i+1, f"{item[0].replace('_', ' ').title()}", 
         f"{item[1].replace('_', ' ').title()}", f"{item[2]:.3f}"]
        for i, item in enumerate(top_corr)
    ]
    
  
    # Return the correlation matrix for potential further analysis
    return correlation_matrix

def main():
    # File paths to the CSV data for each dataset
    datasets = {
        "mnist": "data/results/mnist_merged_report.csv",
        "fashion_mnist": "data/results/fashion_mnist_merged_report.csv"
    }
    
    try:
        for dataset_name, csv_filepath in datasets.items():
            # Load the data
            data = pd.read_csv(csv_filepath)
            
            # Analyze correlations for the dataset
            correlation_matrix = analyze_recall_correlations(data, dataset_name)
            
            print(f"Correlation analysis for {dataset_name} created successfully!")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
