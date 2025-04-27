.PHONY: duckdb-vss duckdb-vss-debug clean prepare-build build run-generate-ground-truth run-create-usearch-indexes run-usearch-random run-usearch-full-coverage run-usearch-new-data run-usearch-unreachable-points run-all-usearch-experiments run-hnswlib-random run-hnswlib-full-coverage run-hnswlib-new-data run-all-hnswlib-experiments run-all run-all-experiments

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

clean:
	rm -rf build

prepare-build:
	mkdir -p build

build: clean prepare-build
	cd build && cmake .. && make

run-generate-ground-truth: build
	./build/generate_ground_truth


# usearch experiments

run-create-usearch-indexes: build
	./build/usearch_create_indexes
	
run-usearch-random: build
	./build/usearch_random
run-usearch-full-coverage: build
	./build/usearch_full_coverage
run-usearch-new-data: build
	./build/usearch_new_data
run-usearch-unreachable-points: build
	./build/usearch_unreachable_points

run-all-usearch-experiments: build
	@echo "Running all usearch experiments..."
	./build/usearch_random
	./build/usearch_full_coverage
	./build/usearch_new_data
	./build/usearch_unreachable_points
	@echo "All usearch experiments completed."
	

# hnswlib experiments

run-hnswlib-create-indexes: build
	./build/hnswlib_create_indexes

run-hnswlib-random: build
	./build/hnswlib_random
run-hnswlib-full-coverage: build
	./build/hnswlib_full_coverage
run-hnswlib-new-data: build
	./build/hnswlib_new_data
run-hnswlib-unreachable-points: build
	./build/hnswlib_unreachable_points

run-all-hnswlib-experiments: build
	@echo "Running all hnswlib experiments..."
	./build/hnswlib_random
	./build/hnswlib_full_coverage
	./build/hnswlib_new_data
	./build/hnswlib_unreachable_points
	@echo "All hnswlib experiments completed."


# everything in order
run-all: build
	@echo "Running full setup and experiments..."
	# generate ground truth to raw.db
	./build/generate_ground_truth
	# create usearch indexes
	./build/usearch_create_indexes
	# run usearch experiments
	./build/usearch_random
	./build/usearch_full_coverage
	./build/usearch_new_data
	./build/usearch_unreachable_points
	# create hnswlib indexes
	./build/hnswlib_create_indexes
	# run hnswlib experiments
	./build/hnswlib_random
	./build/hnswlib_full_coverage
	./build/hnswlib_new_data
	./build/hnswlib_unreachable_points
	@echo "All experiments completed."

# only experiments in order (no setup)
run-all-experiments: build
	@echo "Running all experiments..."
	./build/usearch_random
	./build/usearch_full_coverage
	./build/usearch_new_data
	./build/usearch_unreachable_points
	./build/hnswlib_random
	./build/hnswlib_full_coverage
	./build/hnswlib_new_data
	./build/hnswlib_unreachable_points
	@echo "All experiments completed."
