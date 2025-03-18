# HNSW Visualization Tools

This repository contains tools for visualizing experimental results of Hierarchical Navigable Small World (HNSW) approximate nearest neighbor search algorithms.

## Overview

This project provides scripts for:

- Populating a DuckDB database with experimental data from HDF5 files
- Generating visualizations of experiment results

## Project Structure

```
.
├── scripts/
│   ├── setup/          # Database setup script
│   └── plots/          # Visualization scripts
├── data/               # Data directory
│   └── raw/            # Raw datasets (HDF5 format)
│   └── results/        # Results from embedded-c++ used to generate plots
├── notebooks           # Old experimental notebooks (dead)
├── pyproject.toml      # Poetry configuration
├── poetry.lock         # Poetry lock file
└── Makefile            # Build automation
```

## Datasets

The experimental data is based on standardized ANN datasets in HDF5 format:

- MNIST (784-dimensional, Euclidean distance)
- Fashion-MNIST (784-dimensional, Euclidean distance)
- SIFT (128-dimensional, Euclidean distance)
- GIST (960-dimensional, Euclidean distance)

Each dataset file contains:

- `train` data points used to build the index
- `test` query points
- `neighbors` ground truth nearest neighbors

## Installation

### Using Poetry (recommended)

```bash
# Install dependencies using Poetry
make install
```

### Using pip

```bash
# Generate requirements.txt from Poetry configuration
make requirements.txt

# Install using pip
pip install -r requirements.txt
```

## Generating Plots

To generate visualization plots of the experimental results:

```bash
make plots
```

This will run all the plotting scripts in the `scripts/plots/` directory.

## Running database setup

To populate the persistent raw.db file used in c++ experiments:

```
make setup
```

### Clean Environment

```bash
make clean
```
