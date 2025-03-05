import h5py
import duckdb
import pandas as pd
import os

# dataset = os.getenv("DATASET")
# dataset = 'all'
dataset = "mnist"

dataset_dict = {
    "fashion_mnist": "python/data/raw/fashion-mnist-784-euclidean.hdf5",
    "mnist": "python/data/raw/mnist-784-euclidean.hdf5",
    "sift": "python/data/raw/sift-128-euclidean.hdf5",
    "gist": "python/data/raw/gist-960-euclidean.hdf5",
}


def get_combined_data():
    with h5py.File(dataset_dict[dataset], "r") as f:
        train = f["train"][()]
        test = f["test"][()]

        train_df = pd.DataFrame(train)
        test_df = pd.DataFrame(test)

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        return combined_df


def get_data():
    with h5py.File(dataset_dict[dataset], "r") as f:
        train = f["train"][()]
        return train


def get_test():
    with h5py.File(dataset_dict[dataset], "r") as f:
        test = f["test"][()]
        return test


def transform_data(data):
    df = pd.DataFrame(data)
    df_transformed = df.apply(lambda x: x.tolist(), axis=1).to_frame(name="vec")
    return df_transformed


def create_tables():
    con = duckdb.connect(database="raw.db")

    # Load the df
    train = transform_data(get_data())
    test = transform_data(get_test())

    print(train.size)
    print(train.head())
    dimensionality = len(train["vec"].iloc[0])
    print(dimensionality)

    # Add an id column to the dataframes
    train["id"] = train.index
    test["id"] = test.index

    con.execute(
        f"CREATE OR REPLACE TABLE {dataset}_train (id INTEGER, vec FLOAT[{dimensionality}])"
    )

    con.execute(f"INSERT INTO {dataset}_train SELECT id, vec FROM train")
    train_result = con.execute(f"SELECT * FROM {dataset}_train limit 10").fetchall()
    assert len(train_result) > 0
    print(f"Created table {dataset}_train")

    con.execute(
        f"CREATE OR REPLACE TABLE {dataset}_test (id INTEGER, vec FLOAT[{dimensionality}])"
    )

    con.execute(f"INSERT INTO {dataset}_test SELECT id, vec FROM test")
    test_result = con.execute(f"SELECT * FROM {dataset}_test limit 10").fetchall()

    for row in test_result:
        print(row)
    assert len(test_result) > 0
    print(f"Created table {dataset}_test")

    con.close()


if dataset == "all":
    datasets = dataset_dict.keys()
    for dataset in datasets:
        create_tables()
else:
    create_tables()
