//
// Created by root on 6/6/24.
//

#include "util.h"
#include <vector>
#include <string>





void util::query_hnsw(hnswlib::HierarchicalNSW<float>& alg_hnsw, const std::vector<std::vector<float>>& queries, int k, int num_threads, std::vector<std::vector<size_t>>& results,  std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>>& search_results,
    std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>>& early_term_results,
    std::vector<std::tuple<std::string, int, double>>& search_benchmarks , std::mutex& results_mutex, const std::string& table_name, int iteration, std::vector<std::vector<std::size_t>>& test_neighbor_ids_vec,
    std::vector<int>& test_vector_indices, std::vector<duckdb::Value>& neighbor_ids_values, std::unordered_map<hnswlib::labeltype, size_t>& index_map) {
    size_t num_queries = queries.size();
    results.resize(num_queries, std::vector<size_t>(k));
    ParallelFor(0, num_queries, num_threads, [&](size_t row, size_t threadId) {
        try {

            auto& test_neighbor_ids = test_neighbor_ids_vec[row];
            int test_query_vector_index_int = test_vector_indices[row];
            const Value& neighbor_ids = neighbor_ids_values[row];
            auto start_time = std::chrono::high_resolution_clock::now();
            auto result = alg_hnsw.searchKnn(queries[row].data(), k);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end_time - start_time).count();
        
            unique_ptr<MaterializedQueryResult> result_ids = nullptr;

            // add for each neighbor_ids map it onto the correct idx


            //ADD CODE HERE FOR THIS TODO: 
            
            if(result.neighbors.size() != 0) {
                std::vector<size_t> result_vec_ids;
                result_vec_ids.reserve(result.neighbors.size());

                auto neighbors_size = result.neighbors.size();
                for (std::size_t j = 0; j < neighbors_size; ++j) {
                    size_t key = static_cast<size_t>(result.neighbors.top().second);
                    result.neighbors.pop();
                    result_vec_ids.push_back(key);
                }
                results[row] = result_vec_ids;

                // Create result value directly
                Value result_list_value;

                std::vector<Value> id_values;
                id_values.reserve(result_vec_ids.size());
                
                for (std::size_t j = 0; j < result_vec_ids.size(); ++j) {
                    size_t key = static_cast<size_t>(result_vec_ids[j]);
                    // get original key from index_map
                }
                
                result_list_value =  Value::LIST(LogicalType::INTEGER, std::move(id_values));

                // cast neighbor ids with index_map
                
            
                // Thread-safe collection of results
                {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    
                    // Store the search result
                    search_results.push_back({
                        table_name,
                        iteration,
                        test_query_vector_index_int,
                        neighbor_ids,
                        result_list_value,
                        Value::FLOAT(0.0), // recall (calculated later)
                        result.computed_distances,
                        result.visited_members,
                        result.count
                    });

                    // Store benchmark data
                    search_benchmarks.push_back({
                        table_name,
                        iteration,
                        duration
                    });

                    // Handle early termination case
                    if (result.neighbors.size() < 100) {
                        early_term_results.push_back({
                            table_name,
                            iteration,
                            test_query_vector_index_int,
                            neighbor_ids,
                            result_list_value,
                            Value::FLOAT(0.0), // recall (calculated later)
                            result.computed_distances,
                            result.visited_members,
                            result.count
                        });
                    }
                }
            } else {
                // Handle empty results case
                std::lock_guard<std::mutex> lock(results_mutex);
                search_results.push_back({
                    table_name,
                    iteration,
                    test_query_vector_index_int,
                    neighbor_ids,
                    Value(), // null for empty results
                    Value::FLOAT(0.0), // recall
                    result.computed_distances,
                    result.visited_members,
                    result.count
                });
                
                search_benchmarks.push_back({
                    table_name,
                    iteration,
                    duration
                });
                
                early_term_results.push_back({
                    table_name,
                    iteration,
                    test_query_vector_index_int,
                    neighbor_ids,
                    Value(), // null for empty results
                    Value::FLOAT(0.0), // recall
                    result.computed_distances,
                    result.visited_members,
                    result.count
                });
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(results_mutex);
            std::cerr << "Error processing test vector " << row << ": " << e.what() << std::endl;
        }
        
    });
}

void util::query_hnsw_single(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& queries, int dim, int k, std::vector<std::vector<size_t>>& labels, std::vector<double>& query_times) {
    size_t num_queries = queries.size();
    labels.resize(num_queries, std::vector<size_t>(k));
    query_times.resize(num_queries);

    for (size_t i = 0; i < num_queries; ++i) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto result = index.searchKnn(queries[i].data(), k);
        auto t2 = std::chrono::high_resolution_clock::now();
        query_times[i] = std::chrono::duration<double>(t2 - t1).count();
        for (size_t j = 0; j < k; ++j) {
            labels[i][j] = result.neighbors.top().second;
            result.neighbors.pop();
        }
    }
}


void util::markDeleteMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<size_t>& delete_indices, const std::unordered_map<size_t, size_t>& index_map, int num_threads, std::string dataset_name, int iteration, std::vector<std::tuple<std::string, int, double>>& benchmarks, std::mutex& bench_mutex) {
    size_t num_delete = delete_indices.size();

    ParallelFor(0, num_delete, num_threads, [&](size_t i, size_t) {
       try { size_t idx = index_map.at(delete_indices[i]);
        auto start_time = std::chrono::high_resolution_clock::now();
        index.markDelete(idx);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double>(end_time - start_time).count();
        std::lock_guard<std::mutex> lock(bench_mutex);
        benchmarks.push_back({dataset_name, iteration, duration});
        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(bench_mutex);
            std::cerr << "Error deleting vector " << i << ": " << e.what() << std::endl;
        }
    });
}

void util::addPointsMultiThread(hnswlib::HierarchicalNSW<float>& index, const std::vector<std::vector<float>>& points, const std::vector<size_t>& labels, int num_threads, std::string dataset_name, int iteration, std::vector<std::tuple<std::string, int, double>>& benchmarks, std::mutex& bench_mutex) {
    size_t num_points = points.size();

    ParallelFor(0, num_points, num_threads, [&](size_t i, size_t) {
        try {
            auto start_time = std::chrono::high_resolution_clock::now();
                    
            index.addPoint(points[i].data(), labels[i], true);

            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end_time - start_time).count();
            std::lock_guard<std::mutex> lock(bench_mutex);
            benchmarks.push_back({dataset_name, iteration, duration});
        }
        catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(bench_mutex);
            std::cerr << "Error adding vector " << i << ": " << e.what() << std::endl;
        }
    });
}






