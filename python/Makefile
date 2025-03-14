# Define Python and run command
PYTHON := python3
RUN_CMD :=

# Define directories
ROOT_DIR := scripts/
BENCHMARK_DIR := setup
PLOTS_DIR := plots

# Collect scripts
BENCHMARKS := $(wildcard ${ROOT_DIR}$(BENCHMARK_DIR)/*.py)
PLOTS := $(wildcard ${ROOT_DIR}$(PLOTS_DIR)/*.py)

.PHONY: install plots clean

install:
	@echo "Setting up environment..."
	@poetry install --no-interaction

requirements.txt: pyproject.toml poetry.lock
	@echo "Generating requirements.txt from Poetry..."
	@poetry export --without-hashes --format=requirements.txt > requirements.txt

plots:
	@echo "Generating plots..."
	@for script in $(PLOTS); do \
		$(PYTHON) $$script; \
	done

clean:
	@echo "Cleaning up..."
	@poetry env remove python || true