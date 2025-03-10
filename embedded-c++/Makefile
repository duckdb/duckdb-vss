.PHONY: clean prepare-build duckdb-vss duckdb-vss-debug bm_performance recall recall-debug

all: duckdb-vss recall

all-debug: duckdb-vss-debug recall-debug

clean:
	rm -rf build

prepare-build:
	mkdir -p build

duckdb-vss:
	cd ../ && make

duckdb-vss-debug:
	cd ../ && make debug

bm_performance: clean prepare-build
	cd build && cmake .. && make
	build/benchmark_deletions --benchmark_out=bm_delete_insert.json --benchmark_out_format=json

recall: clean prepare-build
	cd build && cmake .. && make
	build/recall

recall-debug: clean prepare-build
	cd build && cmake -DCMAKE_BUILD_TYPE=Debug .. && make
	build/recall
