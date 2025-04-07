.PHONY: clean prepare-build duckdb-vss duckdb-vss-debug bm_performance recall recall-debug usearch_random usearch_full_coverage usearch_new_data usearch_create_indexes all-usearch build-usearch

# Build everything once, then run all programs
all-usearch: build-usearch
	build/usearch_unreachable_points
	build/usearch_full_coverage
	build/usearch_random
	build/usearch_new_data

# Build all executables only once
build-usearch: clean prepare-build
	cd build && cmake .. && make

clean:
	rm -rf build

prepare-build:
	mkdir -p build

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

usearch_random: build-usearch
	build/usearch_random

usearch_unreachable_points: build-usearch
	build/usearch_unreachable_points

usearch_full_coverage: build-usearch
	build/usearch_full_coverage

usearch_new_data: build-usearch
	build/usearch_new_data

usearch_create_indexes: build-usearch
	build/usearch_create_indexes
