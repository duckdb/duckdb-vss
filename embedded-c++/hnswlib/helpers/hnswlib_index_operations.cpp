#include "hnswlib_index_operations.h"
#include "util.h"
#include <hnswlib/hnswlib.h>

using namespace hnswlib;



std::vector<float> ExtractFloatVector(const Value& value) {

    // Handle FLOAT[N] array types
    if (value.type().id() == LogicalTypeId::ARRAY) {
        const auto& array_children = ArrayValue::GetChildren(value);
        std::vector<float> result;
        result.reserve(array_children.size());

        for (const auto& child : array_children) {
            if (child.IsNull()) {
                result.push_back(0.0f); // Handle NULL values as 0
            } else {
                // Convert various number types to float
                switch (child.type().id()) {
                case LogicalTypeId::FLOAT:
                    result.push_back(child.GetValue<float>());
                    break;
                case LogicalTypeId::DOUBLE:
                    result.push_back(static_cast<float>(child.GetValue<double>()));
                    break;
                case LogicalTypeId::INTEGER:
                    result.push_back(static_cast<float>(child.GetValue<int32_t>()));
                    break;
                case LogicalTypeId::BIGINT:
                    result.push_back(static_cast<float>(child.GetValue<int64_t>()));
                    break;
                default:
                    throw ConversionException("Cannot convert element to float");
                }
            }
        }
        return result;
    }

    // Handle LIST types
    if (value.type().id() == LogicalTypeId::LIST) {
        const auto& list_children = ListValue::GetChildren(value);
        std::vector<float> result;
        result.reserve(list_children.size());

        for (const auto& child : list_children) {
            if (child.IsNull()) {
                result.push_back(0.0f);
            } else {
                // Convert various number types to float
                switch (child.type().id()) {
                case LogicalTypeId::FLOAT:
                    result.push_back(child.GetValue<float>());
                    break;
                case LogicalTypeId::DOUBLE:
                    result.push_back(static_cast<float>(child.GetValue<double>()));
                    break;
                case LogicalTypeId::INTEGER:
                    result.push_back(static_cast<float>(child.GetValue<int32_t>()));
                    break;
                case LogicalTypeId::BIGINT:
                    result.push_back(static_cast<float>(child.GetValue<int64_t>()));
                    break;
                default:
                    throw ConversionException("Cannot convert element to float");
                }
            }
        }
        return result;
    }

    throw ConversionException("Not a list or array type: " + value.type().ToString());
}

std::vector<size_t> ExtractSizeVector(const Value& value) {
    // Handle ARRAY types (INTEGER[N])
    if (value.type().id() == LogicalTypeId::ARRAY) {
        const auto& array_children = ArrayValue::GetChildren(value);
        std::vector<size_t> result;
        result.reserve(array_children.size());

        for (const auto& child : array_children) {
            if (child.IsNull()) {
                result.push_back(0);
            } else {
                // Convert various number types to size_t
                switch (child.type().id()) {
                case LogicalTypeId::INTEGER:
                    result.push_back(static_cast<size_t>(child.GetValue<int32_t>()));
                    break;
                case LogicalTypeId::BIGINT:
                    result.push_back(static_cast<size_t>(child.GetValue<int64_t>()));
                    break;
                case LogicalTypeId::UBIGINT:
                    result.push_back(static_cast<size_t>(child.GetValue<uint64_t>()));
                    break;
                default:
                    throw ConversionException("Cannot convert element to size_t");
                }
            }
        }
        return result;
    }

    // Handle LIST types
    if (value.type().id() == LogicalTypeId::LIST) {
        const auto& list_children = ListValue::GetChildren(value);
        std::vector<size_t> result;
        result.reserve(list_children.size());

        for (const auto& child : list_children) {
            if (child.IsNull()) {
                result.push_back(0);
            } else {
                // Convert various number types to size_t
                switch (child.type().id()) {
                case LogicalTypeId::INTEGER:
                    result.push_back(static_cast<size_t>(child.GetValue<int32_t>()));
                    break;
                case LogicalTypeId::BIGINT:
                    result.push_back(static_cast<size_t>(child.GetValue<int64_t>()));
                    break;
                case LogicalTypeId::UBIGINT:
                    result.push_back(static_cast<size_t>(child.GetValue<uint64_t>()));
                    break;
                default:
                    throw ConversionException("Cannot convert element to size_t");
                }
            }
        }
        return result;
    }

    throw ConversionException("Not a list or array type: " + value.type().ToString());
}

/**
 * Performs vector addition to the index (single thread)
 * 
 * @param index The HNSWLib index to add vectors to
 * @param sample_vecs The result set containing vectors to add
 * @param dataset_name The name of the dataset for benchmarking
 * @param iteration The current iteration number
 * @param add_bm_appender Appender for benchmarking results
 * @return Number of vectors successfully added
 */
size_t HNSWLibIndexOperations::singleAdd(
    HierarchicalNSW<float>& index,
    const unique_ptr<MaterializedQueryResult>& sample_vecs,
    const std::string& dataset_name,
    int iteration,
    Appender& add_bm_appender
) {
    std::cout << "🔵 ADDING SAMPLE VECTORS 🔵" << std::endl;
    size_t added_count = 0;
    try {
        for (idx_t i = 0; i < sample_vecs->RowCount(); i++) {
            auto id = sample_vecs->GetValue<int>(0, i);
            auto vec = ExtractFloatVector(sample_vecs->GetValue(1, i));
            auto start_time = std::chrono::high_resolution_clock::now();
            index.addPoint(&vec[0], id, true);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end_time - start_time).count();
            add_bm_appender.AppendRow(
                Value(dataset_name),
                Value::INTEGER(iteration),
                Value::FLOAT(duration)
            );
            added_count++;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error adding vectors: " << e.what() << std::endl;
    }
    return added_count;
}

/**
 * Performs parallel vector addition to the index
 * 
 * @param index The USearch index to add vectors to
 * @param sample_vecs The result set containing vectors to add
 * @param dataset_name The name of the dataset for benchmarking
 * @param iteration The current iteration number
 * @param add_bm_appender Appender for benchmarking results
 * @return Number of vectors successfully added
 */

size_t HNSWLibIndexOperations::parallelAdd(
    HierarchicalNSW<float>& index,
    const std::vector<std::vector<float>>& points, 
    const std::vector<size_t>& labels,
    const std::string& dataset_name,
    int iteration,
    Appender& add_bm_appender,
    int num_threads
) {
    std::cout << "🔵 ADDING SAMPLE VECTORS 🔵" << std::endl;
    
    size_t added_count = 0;
    
    try {
        
        
        // Create mutex and result collection
        std::mutex bench_mutex;
        std::vector<std::tuple<std::string, int, double>> benchmarks;
        std::atomic<size_t> success_count(0);       
      
        
        std::cout << "Starting parallel add with " << num_threads << " threads" << std::endl;

        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        // Convert ids to std::vector<std::size_t>
        
        util::addPointsMultiThread(index, points, labels, num_threads, dataset_name, iteration, benchmarks, bench_mutex);


        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration<double>(batch_end - batch_start).count();
        std::cout << "Parallel add completed in " << batch_duration << "s" << std::endl;
        
        // Store benchmark data
        for (const auto& bm : benchmarks) {
            add_bm_appender.AppendRow(
                Value(std::get<0>(bm)),
                Value::INTEGER(std::get<1>(bm)),
                Value::FLOAT(std::get<2>(bm))
            );
        }
        
        added_count = success_count;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in parallel vector addition: " << e.what() << std::endl;
    }
    
    return added_count;
}

/**
 * Performs vector deletion from the index (single thread)
 * 
 * @param index The USearch index to remove vectors from
 * @param delete_indices The result set containing ids to remove
 * @param dataset_name The name of the dataset for benchmarking
 * @param index_map A map of internal IDs to index positions
 * @param iteration The current iteration number
 * @param del_bm_appender Appender for benchmarking results
 * @return Number of vectors successfully deleted
 */

size_t HNSWLibIndexOperations::singleRemove(
    HierarchicalNSW<float>& index,
    std::vector<size_t> delete_indices,
    std::unordered_map<hnswlib::labeltype, size_t> index_map,
    const std::string& dataset_name,
    int iteration,
    Appender& del_bm_appender
) {
    std::cout << "🔵 DELETING SAMPLE VECTORS 🔵" << std::endl;

    size_t removed_count = 0;

    auto batch_start = std::chrono::high_resolution_clock::now();

    try {
        for (idx_t i = 0; i < delete_indices.size(); i++) {
            size_t idx = index_map.at(delete_indices[i]);
            auto start_time = std::chrono::high_resolution_clock::now();
            index.markDelete(idx);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end_time - start_time).count();
            del_bm_appender.AppendRow(
                Value(dataset_name),
                Value::INTEGER(iteration),
                Value::FLOAT(duration)
            );
            removed_count++;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error deleting vectors: " << e.what() << std::endl;
    }

    auto batch_end = std::chrono::high_resolution_clock::now();
    auto batch_duration = std::chrono::duration<double>(batch_end - batch_start).count();
    std::cout << "Single-threaded delete completed in " << batch_duration << "s" << std::endl;

    return removed_count;
}

/**
 * Performs parallel vector deletion from the index with improved threading
 * 
 * @param index The USearch index to remove vectors from
 * @param sample_vecs The result set containing vectors to remove
 * @param dataset_name The name of the dataset for benchmarking
 * @param iteration The current iteration number
 * @param del_bm_appender Appender for benchmarking results
 * @return Number of vectors successfully deleted
 */
size_t HNSWLibIndexOperations::parallelRemove(
    HierarchicalNSW<float>& index,
    const std::vector<size_t>& delete_indices,
    const std::unordered_map<hnswlib::labeltype, size_t> &index_map,
    const std::string& dataset_name,
    int iteration,
    Appender& del_bm_appender,
    int num_threads
) {
    std::cout << "🔵 DELETING SAMPLE VECTORS 🔵" << std::endl;
    
    size_t removed_count = 0;
    
    try {

        // Create mutex and result collection
        std::mutex benchmark_mutex;
        std::vector<std::tuple<std::string, int, double>> benchmarks;
        
        // Determine batch size and thread count
        std::size_t total_vectors = delete_indices.size();
     
        
        // Use fewer threads but batch delete operations for better performance
        
      
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        // Process batches in parallel

        std::cout << "Starting parallel delete with " << num_threads << " threads" << std::endl;
        util::markDeleteMultiThread(index, delete_indices, index_map, num_threads, dataset_name, iteration, benchmarks, benchmark_mutex);

        
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration<double>(batch_end - batch_start).count();
        std::cout << "Parallel delete completed in " << batch_duration << "s" << std::endl;
        
        // Store benchmark data
        for (const auto& bm : benchmarks) {
            del_bm_appender.AppendRow(
                Value(std::get<0>(bm)),
                Value::INTEGER(std::get<1>(bm)),
                Value::FLOAT(std::get<2>(bm))
            );
        }
        
        removed_count = benchmarks.size();
    }
    catch (const std::exception& e) {
        std::cerr << "Error in parallel vector deletion: " << e.what() << std::endl;
    }
    
    return removed_count;
    
}

/**
 * Runs test queries on the index (single thread)
 * 
 * @param con The DuckDB connection
 * @param index The USearch index to run queries on
 * @param table_name The name of the dataset
 * @param test_vectors The result set containing test vectors
 * @param appender Appender for results table
 * @param search_appender Appender for search benchmarking table
 * @param early_term_appender Appender for early termination table
 * @param iteration The current iteration number
 * @param dataset_size The size of the dataset
 */
void HNSWLibIndexOperations::runTestQueries(Connection& con, HierarchicalNSW<float>& index, const std::string& table_name,
    const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, Appender& search_appender, Appender& early_term_appender, int iteration, int dataset_size) {
        std::cout << "🧪 RUNNING TEST QUERIES 🧪" << std::endl;

        std::priority_queue<std::pair<float, hnswlib::labeltype>> search_results;
    
        // Run KNN search for each test vector
        for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
            try {
                auto test_vec = ExtractFloatVector(test_vectors->GetValue(1, i));
                auto test_neighbor_ids = ExtractSizeVector(test_vectors->GetValue(2, i));
                int test_query_vector_index_int = test_vectors->GetValue(0, i).GetValue<int>();
                Value neighbor_ids = test_vectors->GetValue(2, i);
    
                auto start_time = std::chrono::high_resolution_clock::now();
                auto results = index.searchKnn(&test_vec[0], 100); // TODO wanted hardcoded
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double>(end_time - start_time).count();
    
                if(results.neighbors.size() != 0) {
                    std::vector<Value> result_vec_ids;
                    result_vec_ids.reserve(results.neighbors.size());
    
                    // Create result value directly
                    Value result_list_value;
    
                    for (std::size_t j = 0; j < results.neighbors.size(); ++j){
                        size_t key = static_cast<size_t>(results.neighbors.top().first);
                        results.neighbors.pop();
                        
                        result_vec_ids.push_back(Value::INTEGER(key));
                    }
    
                    result_list_value =  Value::LIST(LogicalType::INTEGER, std::move(result_vec_ids));
    
                    // Append search results to results table
                    try {
                        appender.AppendRow(
                            Value(table_name),                              // dataset name
                            Value::INTEGER(iteration),                      // iteration number
                            Value::INTEGER(test_query_vector_index_int),    // test vector ID
                            neighbor_ids,                                   // ground truth neighbors
                            result_list_value,                              // result vector IDs
                            Value(),                                      // recall (computed later)
                            // have to add back
                            Value::INTEGER(results.computed_distances),     // computed distances
                            Value::INTEGER(results.visited_members),        // visited members
                            Value::INTEGER(results.count)                   // count
                        );
                    } catch (const std::exception& e) {
                        std::cerr << "Error appending row: " << e.what() << std::endl;
                        continue;
                    }
    
                    // If < 100 results are returned, append search query stats to early termination table
                    if (results.neighbors.size() < 100) {
                        try {
                            early_term_appender.AppendRow(
                                Value(table_name),                              // dataset name
                                Value::INTEGER(iteration),                      // iteration number
                                Value::INTEGER(test_query_vector_index_int),    // test vector ID
                                neighbor_ids,                                   // ground truth neighbors
                                result_list_value,                              // result vector IDs
                                Value(),
                                // Have to add back                                     // recall (computed later)
                                Value::INTEGER(results.computed_distances),     // computed distances
                                Value::INTEGER(results.visited_members),        // visited members
                                Value::INTEGER(results.count)                   // count
                            );
                        } catch (const std::exception& e) {
                            std::cerr << "Error appending row: " << e.what() << std::endl;
                            continue;
                        }
                    }
                // 0 results returned case
                } else {
                    // Append search results to results table
                    try {
                        appender.AppendRow(
                            Value(table_name),                              // dataset name
                            Value::INTEGER(iteration),                      // iteration number
                            Value::INTEGER(test_query_vector_index_int),    // test vector ID
                            neighbor_ids,                                   // ground truth neighbors
                            Value(),                                        // result vector IDs
                            Value(),
                            // have to add back                                        // recall (computed later)
                            Value::INTEGER(results.computed_distances),     // computed distances
                            Value::INTEGER(results.visited_members),        // visited members
                            Value::INTEGER(results.count)                   // count
                        );
                    } catch (const std::exception& e) {
                        std::cerr << "Error appending row: " << e.what() << std::endl;
                        continue;
                    }
    
                    // Append search query stats to early termination table
                    try {
                        early_term_appender.AppendRow(
                            Value(table_name),                              // dataset name
                            Value::INTEGER(iteration),                      // iteration number
                            Value::INTEGER(test_query_vector_index_int),    // test vector ID
                            neighbor_ids,                                   // ground truth neighbors
                            Value(),                                        // result vector IDs
                            Value(),                                   // recall (computed later)
                            // have to add back                                        // recall (computed later)
                            Value::INTEGER(results.computed_distances),     // computed distances
                            Value::INTEGER(results.visited_members),        // visited members
                            Value::INTEGER(results.count)                   // count
                        );
                    } catch (const std::exception& e) {
                        std::cerr << "Error appending row: " << e.what() << std::endl;
                        continue;
                    }
                }
    
                // Append search bm stats
                try {
                    search_appender.AppendRow(
                        Value(table_name),
                        Value::INTEGER(iteration),
                        Value::FLOAT(duration)
                    );
                } catch (const std::exception& e) {
                    std::cerr << "Error appending search bm row: " << e.what() << std::endl;
                    continue;
                }
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing test vector " << i << ": " << e.what() << std::endl;
                continue;
            }
        }
}

/**
 * Runs test queries on the index in parallel
 * 
 * @param con The DuckDB connection
 * @param index The USearch index to query
 * @param table_name The name of the dataset
 * @param test_vectors The result set containing test vectors
 * @param appender Appender for search results
 * @param search_appender Appender for benchmarking search results
 * @param early_term_appender Appender for early termination results
 * @param iteration The current iteration number
 * @param dataset_size The size of the dataset
 */
void HNSWLibIndexOperations::parallelRunTestQueries(Connection& con, HierarchicalNSW<float>& index, const std::string& table_name,
    const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, Appender& search_appender, 
    Appender& early_term_appender, int iteration, int dataset_size, std::unordered_map<hnswlib::labeltype, size_t>& index_map) {
    std::cout << "🧪 RUNNING TEST QUERIES 🧪" << std::endl;

    try {
        // Extract all test vectors upfront to avoid DB access in threads
        std::vector<std::vector<float>> test_vecs;
        std::vector<std::vector<size_t>> test_neighbor_ids_vec;
        std::vector<int> test_vector_indices;
        std::vector<Value> neighbor_ids_values;

        test_vecs.reserve(test_vectors->RowCount());
        test_neighbor_ids_vec.reserve(test_vectors->RowCount());
        test_vector_indices.reserve(test_vectors->RowCount());
        neighbor_ids_values.reserve(test_vectors->RowCount());

        for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
            test_vecs.push_back(ExtractFloatVector(test_vectors->GetValue(1, i)));

            // Extract neighbor IDs and filter to include only those present in the index
            auto original_neighbors = ExtractSizeVector(test_vectors->GetValue(2, i));
            std::vector<size_t> filtered_neighbors;
            filtered_neighbors.reserve(original_neighbors.size());
            // Filter neighbors to only include IDs that exist in the index
            for (auto& neighbor_id : original_neighbors) {
                if (index.getDataByInternalId(neighbor_id)) {
                    auto idx = index_map.find(neighbor_id);
                    filtered_neighbors.push_back(idx->first);
                }
            }
            // Convert filtered neighbors back to DuckDB Value
            std::vector<Value> filtered_values;
            filtered_values.reserve(filtered_neighbors.size());
            for (auto& id : filtered_neighbors) {
                filtered_values.push_back(Value::INTEGER(id));
            }
            // Store the filtered neighbor IDs
            Value filtered_list_value = Value::LIST(LogicalType::INTEGER, std::move(filtered_values));
            test_neighbor_ids_vec.push_back(filtered_neighbors);
            test_vector_indices.push_back(test_vectors->GetValue(0, i).GetValue<int>());
            neighbor_ids_values.push_back(filtered_list_value);
        }

        // Thread-safe containers for results
        std::mutex results_mutex;
        std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>> search_results;
        std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>> early_term_results;
        std::vector<std::tuple<std::string, int, double>> search_benchmarks;

        std::size_t executor_threads = std::min(std::thread::hardware_concurrency(), 
                                                static_cast<unsigned int>(test_vectors->RowCount()));
        
        
        std::cout << "Starting parallel search with " << executor_threads << " threads" << std::endl;
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        std::vector<std::vector<size_t>> converted_test_vector_indices = {std::vector<size_t>(test_vector_indices.begin(), test_vector_indices.end())};
        util::query_hnsw(index, test_vecs, 100, executor_threads, converted_test_vector_indices, search_results, early_term_results, search_benchmarks, results_mutex, table_name, iteration, test_neighbor_ids_vec, test_vector_indices, neighbor_ids_values, index_map);
       
   
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration<double>(batch_end - batch_start).count();
        std::cout << "Parallel search completed in " << batch_duration << "s" << std::endl;

        // Bulk append all results
        std::cout << "Appending " << search_results.size() << " search results" << std::endl;
        for (const auto& result : search_results) {
            try {
                appender.AppendRow(
                    Value(std::get<0>(result)),                // dataset name
                    Value::INTEGER(std::get<1>(result)),       // iteration number
                    Value::INTEGER(std::get<2>(result)),       // test vector ID
                    std::get<3>(result),                       // neighbor IDs
                    std::get<4>(result),                       // result vector IDs
                    std::get<5>(result),                       // recall (to be calculated later)
                    Value::INTEGER(std::get<6>(result)),       // computed distances
                    Value::INTEGER(std::get<7>(result)),       // visited members
                    Value::INTEGER(std::get<8>(result))        // count
                );
            } catch (const std::exception& e) {
                std::cerr << "Error appending search result: " << e.what() << std::endl;
            }
        }
        
        // Append early termination results if any
        if (early_term_results.size() > 0) {
            std::cout << "Appending " << early_term_results.size() << " early termination results" << std::endl;
            for (const auto& result : early_term_results) {
                try {
                    early_term_appender.AppendRow(
                        Value(std::get<0>(result)),                // dataset name
                        Value::INTEGER(std::get<1>(result)),       // iteration number
                        Value::INTEGER(std::get<2>(result)),       // test vector ID
                        std::get<3>(result),                       // neighbor IDs
                        std::get<4>(result),                       // result vector IDs
                        std::get<5>(result),                       // recall (to be calculated later)
                        Value::INTEGER(std::get<6>(result)),       // computed distances
                        Value::INTEGER(std::get<7>(result)),       // visited members
                        Value::INTEGER(std::get<8>(result))        // count
                    );
                } catch (const std::exception& e) {
                    std::cerr << "Error appending early termination result: " << e.what() << std::endl;
                }
            }
        }
        
        // Append search benchmark data
        for (const auto& bm : search_benchmarks) {
            try {
                search_appender.AppendRow(
                    Value(std::get<0>(bm)),
                    Value::INTEGER(std::get<1>(bm)),
                    Value::FLOAT(std::get<2>(bm))
                );
            } catch (const std::exception& e) {
                std::cerr << "Error appending search benchmark: " << e.what() << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error in parallel search: " << e.what() << std::endl;
    }
}


    

