.PHONY: clean prepare-build duckdb-vss duckdb-vss-debug build run-create-indexes run-random run-full-coverage run-new-data run-unreachable-points run-all-experiments

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

run-create-indexes: build
	./build/usearch_create_indexes
	
run-random: build
	./build/usearch_random
run-full-coverage: build
	./build/usearch_full_coverage
run-new-data: build
	./build/usearch_new_data
run-unreachable-points: build
	./build/usearch_unreachable_points

run-all-experiments: build
	@echo "Running all usearch experiments..."
	./build/usearch_random
	./build/usearch_full_coverage
	./build/usearch_new_data
	./build/usearch_unreachable_points
	@echo "All usearch experiments completed."