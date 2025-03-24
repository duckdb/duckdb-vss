.PHONY: clean prepare-build duckdb-vss duckdb-vss-debug bm_performance recall recall-debug usearch_random usearch_full_coverage usearch_new_data

all-usearch: usearch_full_coverage usearch_random usearch_new_data

clean:
	rm -rf build

prepare-build:
	mkdir -p build

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

usearch_random: clean prepare-build
	cd build && cmake .. && make
	build/usearch_random

usearch_full_coverage: clean prepare-build
	cd build && cmake .. && make
	build/usearch_full_coverage

usearch_new_data: clean prepare-build
	cd build && cmake .. && make
	build/usearch_new_data
