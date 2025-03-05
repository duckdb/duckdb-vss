.PHONY: clean duckdb-vss duckdb-vss-debug bm_performance results

all: duckdb-vss bm_performance

#all-bm_memory: duckdb-vss bm_memory

all-main: duckdb-vss main

clean:
	rm -rf build

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

prepare-build:
	mkdir -p build

bm_performance: clean prepare-build
	cd build && cmake .. && make
	build/improved_deletions --benchmark_out=bm_vss_deletion_time_hmm.json --benchmark_out_format=json

	
#build/benchmark_search --benchmark_out=bm_vss_search_time_hmm.json --benchmark_out_format=json

# Uncomment JSON output in physical operator before running this
#bm_memory: clean prepare-build
#	cd build && cmake .. && make
#	build/benchmark_memory
