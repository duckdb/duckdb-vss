#pragma once

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include "duckdb.hpp"
#include "duckdb/common/exception/conversion_exception.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <mutex>
#include <atomic>

using namespace duckdb;
using namespace unum::usearch;

// Helper functions should be moved to a utilities header
std::vector<float> ExtractFloatVector(const Value& value);
std::vector<size_t> ExtractSizeVector(const Value& value);

class IndexOperations {
public:
    static void runTestQueries(Connection& con, index_dense_gt<row_t>& index, const std::string& table_name,
        const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, Appender& search_appender, 
        Appender& early_term_appender, int iteration, int dataset_size);
        
    static void parallelRunTestQueries(Connection& con, index_dense_gt<row_t>& index, const std::string& table_name,
        const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, Appender& search_appender, 
        Appender& early_term_appender, int iteration, int dataset_size);

    static size_t singleAdd(
        index_dense_gt<row_t>& index,
        const unique_ptr<MaterializedQueryResult>& sample_vecs,
        const std::string& dataset_name,
        int iteration,
        Appender& add_bm_appender
    );
    
    static size_t parallelAdd(
        index_dense_gt<row_t>& index,
        const unique_ptr<MaterializedQueryResult>& sample_vecs,
        const std::string& dataset_name,
        int iteration,
        Appender& add_bm_appender
    );
    
    static size_t singleRemove(
        index_dense_gt<row_t>& index,
        const unique_ptr<MaterializedQueryResult>& sample_vecs,
        const std::string& dataset_name,
        int iteration,
        Appender& del_bm_appender
    );
    
    static size_t parallelRemove(
        index_dense_gt<row_t>& index,
        const unique_ptr<MaterializedQueryResult>& sample_vecs,
        const std::string& dataset_name,
        int iteration,
        Appender& del_bm_appender
    );
};
