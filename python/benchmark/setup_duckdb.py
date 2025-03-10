import h5py
import duckdb
import pandas as pd
import os

dataset = 'all'

dataset_dict = {
    "fashion_mnist": "../data/raw/fashion-mnist-784-euclidean.hdf5",
    "mnist": "../data/raw/mnist-784-euclidean.hdf5",
    "sift": "../data/raw/sift-128-euclidean.hdf5",
    "gist": "../data/raw/gist-960-euclidean.hdf5",
}

def get_train():
    with h5py.File(dataset_dict[dataset], "r") as f:
        # Print the structure of the HDF5 file
        print(f"HDF5 File Structure for {dataset}:")
        print("Keys:", list(f.keys()))
        train = f["train"][()]
        return train

def get_test():
    with h5py.File(dataset_dict[dataset], "r") as f:
        test = f["test"][()]
        return test

def get_neighbors():
    with h5py.File(dataset_dict[dataset], "r") as f:
        neighbors = f["neighbors"][()]
        return neighbors

def get_distances():
    with h5py.File(dataset_dict[dataset], "r") as f:
        distances = f["distances"][()]
        return distances

def transform_data(data, column_name):
    df = pd.DataFrame(data)
    df_transformed = df.apply(lambda x: x.tolist(), axis=1).to_frame(name=column_name)
    return df_transformed

def create_tables():
    con = duckdb.connect(database="../../embedded-c++/raw.db")

    # Load the df
    train = transform_data(get_train(), "vec")
    test = transform_data(get_test(), "vec")
    neighbors = transform_data(get_neighbors(), "neighbor_ids")

    dimensionality = len(train["vec"].iloc[0])

    # Add an id column to the dataframes
    train["id"] = train.index
    test["id"] = test.index
    neighbors["id"] = neighbors.index

    con.execute(
        f"CREATE OR REPLACE TABLE {dataset}_train (id INTEGER, vec FLOAT[{dimensionality}])"
    )
    con.execute(f"INSERT INTO {dataset}_train SELECT id, vec FROM train")
    train_result = con.execute(f"SELECT * FROM {dataset}_train limit 1").fetchall()
    assert len(train_result) > 0
    print(f"Created table {dataset}_train")

    # Merge test and neighbors by id to create a combined dataframe
    test_neighbors_df = pd.merge(test, neighbors, on="id")
    # Reorder columns to match desired order: id, vec, neighbor_ids
    test_neighbors_df = test_neighbors_df[["id", "vec", "neighbor_ids"]]

    con.execute(
        f"CREATE OR REPLACE TABLE {dataset}_test (id INTEGER, vec FLOAT[{dimensionality}], neighbor_ids INTEGER[100])"
    )
    con.execute(f"INSERT INTO {dataset}_test SELECT id, vec, neighbor_ids FROM test_neighbors_df")
    test_result = con.execute(f"SELECT * FROM {dataset}_test limit 1").fetchall()
    assert len(test_result) > 0
    print(f"Created table {dataset}_test")

    con.close()

if dataset == "all":
    datasets = dataset_dict.keys()
    for dataset in datasets:
        create_tables()
else:
    create_tables()
