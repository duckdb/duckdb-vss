//
// Created by root on 6/6/24.
//

#ifndef GRAPH_SEARCH_UTIL_H
#define GRAPH_SEARCH_UTIL_H
#include "iostream"
#include "vector"
#include <thread>
#include <atomic>
#include <mutex>
#include <fstream>
#include <sstream>
#include <filesystem>

#include "hnswlib/hnswlib.h"
#include "thread_pool.h"
#include "duckdb.hpp"
#include "duckdb/common/exception/conversion_exception.hpp"

using namespace duckdb;

class util{
public:

    static void query_hnsw(hnswlib::HierarchicalNSW<float>& alg_hnsw, const std::vector<std::vector<float>>& queries, int k, int num_threads, std::vector<std::vector<size_t>>& results,  std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>>& search_results,
        std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>>& early_term_results,
        std::vector<std::tuple<std::string, int, double>>& search_benchmarks , std::mutex& results_mutex, const std::string& table_name, int iteration,
        std::vector<int>& test_vector_indices, std::vector<duckdb::Value>& neighbor_ids_values,  std::unordered_map<hnswlib::labeltype, size_t>& index_map);

    static void markDeleteMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<size_t>& delete_indices, const std::unordered_map<size_t, size_t>& index_map, int num_threads, std::string dataset_name, int iteration, std::vector<std::tuple<std::string, int, double>>& benchmarks, std::mutex& bench_mutex);

    static void addPointsMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& points, const std::vector<size_t>& labels, int num_threads, std::string dataset_name, int iteration, std::vector<std::tuple<std::string, int, double>>& benchmarks, std::mutex& bench_mutex);

    static void query_hnsw_single(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times);

    template<class Function>
    inline static void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
        static ThreadPool pool(numThreads > 0 ? numThreads : std::thread::hardware_concurrency());

        if (numThreads <= 0) {
            numThreads = std::thread::hardware_concurrency();
        }

        if (numThreads == 1) {
            for (size_t id = start; id < end; id++) {
                fn(id, 0);
            }
        } else {
            std::atomic<size_t> current(start);

            // keep track of exceptions in threads
            std::exception_ptr lastException = nullptr;
            std::mutex lastExceptMutex;

            for (size_t threadId = 0; threadId < numThreads; ++threadId) {
                pool.enqueue([&, threadId] {
                    while (true) {
                        size_t id = current.fetch_add(1);

                        if (id >= end) {
                            break;
                        }

                        try {
                            fn(id, threadId);
                        } catch (...) {
                            std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                            lastException = std::current_exception();
                            current = end;
                            break;
                        }
                    }
                });
            }

            pool.waitForCompletion();

            if (lastException) {
                std::rethrow_exception(lastException);
            }
        }
    }

};

#endif //GRAPH_SEARCH_UTIL_H
