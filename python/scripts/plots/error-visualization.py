import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def analyze_failed_queries(failed_queries_file, output_dir="./plots/failed_queries/"):
    """
    Analyze and visualize patterns in queries that couldn't return the full number of neighbors.
    
    Parameters:
    failed_queries_file (str): Path to the CSV file with failed query data
    output_dir (str): Directory to save the visualization outputs
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    data = pd.read_csv(failed_queries_file)
    
    # Display basic statistics
    print(f"Total failed queries: {len(data)}")
    print(f"Unique test vectors with failures: {data['test_vector_id'].nunique()}")
    print(f"Average neighbors returned: {data['results_returned'].mean():.2f} out of {data['top_limit'].mean():.2f} requested")
    
    # Group by dataset and iteration
    grouped = data.groupby(['dataset', 'iteration']).agg({
        'test_vector_id': 'count',
        'results_returned': 'mean',
        'iteration_cycles': 'mean',
        'visits': 'mean',
        'number_of_cand_neighbors_last_node': 'mean'
    }).reset_index()
    
    grouped = grouped.rename(columns={'test_vector_id': 'failure_count'})
    
    # 1. Failure count by iteration
    plt.figure(figsize=(12, 7))
    sns.barplot(x='iteration', y='failure_count', hue='dataset', data=grouped)
    plt.title('Number of Failed Queries by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Number of Failures')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}failures_by_iteration.png", dpi=300)
    plt.close()
    
    # 2. Average results returned by iteration
    plt.figure(figsize=(12, 7))
    sns.barplot(x='iteration', y='results_returned', hue='dataset', data=grouped)
    plt.title('Average Results Returned for Failed Queries by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Avg. Results Returned')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}avg_results_by_iteration.png", dpi=300)
    plt.close()
    
    # 3. Correlation between visits and results returned
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='visits', y='results_returned', hue='iteration', 
                   size='number_of_cand_neighbors_last_node', sizes=(50, 200),
                   palette='viridis', data=data)
    plt.title('Relationship Between Visits and Results Returned')
    plt.xlabel('Number of Visits')
    plt.ylabel('Results Returned')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}visits_vs_results.png", dpi=300)
    plt.close()
    
    # 4. Histograms of key metrics to identify distributions
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    
    # Results returned distribution
    sns.histplot(data=data, x='results_returned', kde=True, ax=axs[0, 0])
    axs[0, 0].set_title('Distribution of Results Returned')
    axs[0, 0].grid(True, alpha=0.3)
    
    # Visits distribution
    sns.histplot(data=data, x='visits', kde=True, ax=axs[0, 1])
    axs[0, 1].set_title('Distribution of Visits')
    axs[0, 1].grid(True, alpha=0.3)
    
    # Iteration cycles distribution
    sns.histplot(data=data, x='iteration_cycles', kde=True, ax=axs[1, 0])
    axs[1, 0].set_title('Distribution of Iteration Cycles')
    axs[1, 0].grid(True, alpha=0.3)
    
    # Number of candidates in last node distribution
    sns.histplot(data=data, x='number_of_cand_neighbors_last_node', kde=True, ax=axs[1, 1])
    axs[1, 1].set_title('Distribution of Candidates in Last Node')
    axs[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}metric_distributions.png", dpi=300)
    plt.close()
    
    # Correlation matrix between all numeric variables
    numeric_cols = ['results_returned', 'top_n_neighbors', 'top_limit', 
                   'iteration_cycles', 'visits', 'last_node_expanded',
                   'number_of_cand_neighbors_last_node']
    
    plt.figure(figsize=(12, 10))
    correlation_matrix = data[numeric_cols].corr()
    
    mask = np.zeros_like(correlation_matrix)
    mask[np.triu_indices_from(mask)] = True
    
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
               mask=mask, linewidths=0.5, fmt='.2f')
    plt.title('Correlation Matrix of Query Metrics')
    plt.tight_layout()
    plt.savefig(f"{output_dir}correlation_matrix.png", dpi=300)
    plt.close()
    
    # Visualize relationship between last node expanded and results
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='last_node_expanded', y='results_returned', 
                   hue='number_of_cand_neighbors_last_node', size='visits',
                   sizes=(20, 200), palette='viridis', data=data)
    plt.title('Last Node Expanded vs Results Returned')
    plt.xlabel('Last Node ID')
    plt.ylabel('Results Returned')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}last_node_vs_results.png", dpi=300)
    plt.close()
    
    # Box plot of results returned by iteration
    plt.figure(figsize=(12, 7))
    sns.boxplot(x='iteration', y='results_returned', data=data)
    plt.title('Distribution of Results Returned by Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Results Returned')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}results_boxplot_by_iteration.png", dpi=300)
    plt.close()
    
    # Are certain vector IDs consistently failing?
    # Count failures per test vector ID
    vector_failure_counts = data['test_vector_id'].value_counts().reset_index()
    vector_failure_counts.columns = ['test_vector_id', 'failure_count']
    
    # Plot vectors with multiple failures
    plt.figure(figsize=(14, 8))
    multiple_failures = vector_failure_counts[vector_failure_counts['failure_count'] > 1]
    
    if not multiple_failures.empty:
        multiple_failures = multiple_failures.sort_values('failure_count', ascending=False)
        sns.barplot(x='test_vector_id', y='failure_count', data=multiple_failures.head(30))
        plt.title('Test Vectors with Multiple Failures')
        plt.xlabel('Test Vector ID')
        plt.ylabel('Number of Failures')
        plt.xticks(rotation=90)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(f"{output_dir}repeat_failure_vectors.png", dpi=300)
        plt.close()
    else:
        print("No test vectors with multiple failures found.")
    
    # Analyze the relationship between visited nodes and candidate neighbors
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='visits', y='number_of_cand_neighbors_last_node', 
                   hue='results_returned', size='iteration_cycles',
                   sizes=(50, 200), palette='viridis', data=data)
    plt.title('Visits vs Candidate Neighbors in Last Node')
    plt.xlabel('Number of Nodes Visited')
    plt.ylabel('Candidate Neighbors in Last Node')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}visits_vs_candidates.png", dpi=300)
    plt.close()
    
    # Create a summary table of factors that might indicate why queries fail
    # Create a function to categorize potential failure reasons
    def categorize_failure(row):
        categories = []
        
        if row['visits'] <= 1:
            categories.append("Limited exploration (≤1 visit)")
        
        if row['number_of_cand_neighbors_last_node'] <= 1:
            categories.append("Dead-end node (≤1 candidate)")
            
        if row['iteration_cycles'] > data['iteration_cycles'].median():
            categories.append("High computation (>median cycles)")
            
        if not categories:
            categories.append("Unknown")
            
        return "|".join(categories)
    
    # Add failure category to data
    data['failure_category'] = data.apply(categorize_failure, axis=1)
    
    # Count occurrences of each category
    failure_categories = data['failure_category'].value_counts().reset_index()
    failure_categories.columns = ['Category', 'Count']
    
    # Plot failure categories
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Category', y='Count', data=failure_categories)
    plt.title('Potential Reasons for Query Failures')
    plt.xlabel('Failure Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(f"{output_dir}failure_categories.png", dpi=300)
    plt.close()
    
    # Return summary statistics for further analysis
    return {
        "total_failures": len(data),
        "failures_by_iteration": grouped,
        "failure_categories": failure_categories
    }

def main():
    # Specify the path to your failed queries data
    failed_queries_file = "data/results/mnist_early_termination_reports.csv"
    
    try:
        # Analyze failed queries
        results = analyze_failed_queries(failed_queries_file)
        print("Failed queries analysis completed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()