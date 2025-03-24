# Define Python and run command
PYTHON := python3
RUN_CMD :=

# Define directories
ROOT_DIR := scripts/
SETUP_DIR := setup
PLOTS_DIR := plots
VENV_DIR := .venv

# Collect scripts
SETUPS := $(wildcard ${ROOT_DIR}$(SETUP_DIR)/*.py)
PLOTS := $(wildcard ${ROOT_DIR}$(PLOTS_DIR)/*.py)

.PHONY: all venv install setup plots clean

all: venv install setup plots

venv:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@test -d $(VENV_DIR) || $(PYTHON) -m venv $(VENV_DIR)
	@echo "Virtual environment created successfully."
	@echo "To activate: source $(VENV_DIR)/bin/activate"

install: venv
	@echo "Setting up environment..."
	@$(VENV_DIR)/bin/pip install -U pip setuptools wheel
	@$(VENV_DIR)/bin/pip install poetry
	@$(VENV_DIR)/bin/poetry install --no-interaction

requirements.txt: pyproject.toml poetry.lock
	@echo "Generating requirements.txt from Poetry..."
	@poetry export --without-hashes --format=requirements.txt > requirements.txt

plots:
	@echo "Generating plots..."
	@for script in $(PLOTS); do \
		$(VENV_DIR)/bin/$(PYTHON) $$script; \
	done

setup:
	@echo "Setting up benchmarks..."
	@for script in $(SETUPS); do \
		$(VENV_DIR)/bin/$(PYTHON) $$script; \
	done

clean:
	@echo "Cleaning up..."
	@rm -rf $(VENV_DIR)