#pragma once

#include <hnswlib/hnswlib.h>
#include "duckdb.hpp"
#include "duckdb/common/exception/conversion_exception.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>
#include <atomic>

using namespace duckdb;
using namespace hnswlib;

// Helper functions should be moved to a utilities header
std::vector<float> ExtractFloatVector(const Value& value);
std::vector<size_t> ExtractSizeVector(const Value& value);

class HNSWLibIndexOperations {
public: 
    static void parallelRunTestQueries(Connection& con, HierarchicalNSW<float>& index, const std::string& table_name,
        const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, Appender& search_appender, 
        Appender& early_term_appender, int iteration, int dataset_size, std::unordered_map<hnswlib::labeltype, size_t>& index_map, bool new_data = false);
    
    static size_t parallelAdd(
        HierarchicalNSW<float>& index,
        const std::vector<std::vector<float>>& points, 
        const std::vector<size_t>& labels,
        const std::string& dataset_name,
        int iteration,
        Appender& add_bm_appender,
        int num_threads
    );
    
    static size_t singleRemove(
        HierarchicalNSW<float>& index,
        std::vector<size_t> delete_indices,
        std::unordered_map<hnswlib::labeltype, size_t> index_map,
        const std::string& dataset_name,
        int iteration,
        Appender& del_bm_appender
    );
    
    static size_t parallelRemove(
        HierarchicalNSW<float>& index,
        const std::vector<size_t>& delete_indices,
        const std::unordered_map<hnswlib::labeltype, size_t> &index_map,
        const std::string& dataset_name,
        int iteration,
        Appender& del_bm_appender,
        int num_threads
    );
};
