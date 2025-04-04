"""Module for generating line graphs from CSV data."""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_line_csv_graphs(dataset_folder):
    """
    Reads all CSV files in the given dataset folder and plots graphs for each numerical column
    with iteration as the x-axis. Saves the plots in an 'images' folder inside the dataset folder.
    """

    # Create an images folder if it doesn't exist
    images_folder = os.path.join(dataset_folder, "images")
    os.makedirs(images_folder, exist_ok=True)

    # List all CSV files in the dataset folder
    csv_files = [f for f in os.listdir(dataset_folder) if f.endswith('.csv')]

    for csv_file in csv_files:
        file_path = os.path.join(dataset_folder, csv_file)
        df = pd.read_csv(file_path)

        # Automatically detect the iteration column
        iteration_col = None
        for col in df.columns:
            if 'iteration' in col.lower():
                iteration_col = col
                break
            else:
                # If iteration does not exist use the index as the x-axis
                iteration_col = df.columns[0]


        # Plot each numerical column against iteration
        for col in df.columns:
            if col != iteration_col and pd.api.types.is_numeric_dtype(df[col]):
                plt.figure(figsize=(10, 5))
                plt.plot(df[iteration_col], df[col], marker='o', linestyle='-')
                plt.xlabel(iteration_col)
                plt.ylabel(col)
                plt.title(f"{csv_file} - {col} vs {iteration_col}")
                plt.grid(True)

                # Save plot in images folder
                save_path = os.path.join(images_folder, f"{csv_file}_{col}.png")
                plt.savefig(save_path)
                plt.close()

        print(f"Processed {csv_file}")



def main():
    """Process CSV files from multiple experiment folders and generate line graphs."""
    # Example usage
    # File path to the CSV data
    csv_filepath = "../embedded-c++/usearch/results/"
    experiments = ["fullcoverage", "newdata", "random"]


    for experiment in experiments:
        dataset_filepath = f"{csv_filepath}/{experiment}/"
        #for each folder in the dataset_filepath:
        for folder in os.listdir(dataset_filepath):
            if os.path.isdir(os.path.join(dataset_filepath, folder)):
                plot_line_csv_graphs(os.path.join(dataset_filepath, folder))
                print(f"Processed {folder}")







if __name__ == "__main__":
    main()
