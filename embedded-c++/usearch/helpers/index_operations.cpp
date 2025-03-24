#include "index_operations.h"

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
 * @param index The USearch index to add vectors to
 * @param sample_vecs The result set containing vectors to add
 * @param dataset_name The name of the dataset for benchmarking
 * @param iteration The current iteration number
 * @param add_bm_appender Appender for benchmarking results
 * @return Number of vectors successfully added
 */
size_t IndexOperations::singleAdd(
    index_dense_gt<row_t>& index,
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
            auto size = (index.size() + 1);
            if (index.size() + 1 > index.capacity()) {
                index.reserve(NextPowerOfTwo(size));
            }
            auto start_time = std::chrono::high_resolution_clock::now();
            index.add(id, &vec[0]);
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

size_t IndexOperations::parallelAdd(
    index_dense_gt<row_t>& index,
    const unique_ptr<MaterializedQueryResult>& sample_vecs,
    const std::string& dataset_name,
    int iteration,
    Appender& add_bm_appender
) {
    std::cout << "🔵 ADDING SAMPLE VECTORS 🔵" << std::endl;
    
    size_t added_count = 0;
    
    try {
        std::vector<int> ids;
        std::vector<std::vector<float>> vectors;
        ids.reserve(sample_vecs->RowCount());
        vectors.reserve(sample_vecs->RowCount());
        
        for (idx_t i = 0; i < sample_vecs->RowCount(); i++) {
            ids.push_back(sample_vecs->GetValue<int>(0, i));
            vectors.push_back(ExtractFloatVector(sample_vecs->GetValue(1, i)));
        }
        
        // Create mutex and result collection
        std::mutex bench_mutex;
        std::vector<std::tuple<std::string, int, double>> benchmarks;
        std::atomic<size_t> success_count(0);
        
        // Multi-threaded execution
        std::size_t executor_threads = std::min(std::thread::hardware_concurrency(), 
                                            static_cast<unsigned int>(sample_vecs->RowCount()));
        executor_default_t executor(executor_threads);
        
        std::cout << "Starting parallel add with " << executor_threads << " threads" << std::endl;

        // Ensure enough capacity
        auto size = index.size() + sample_vecs->RowCount();
        if (size > index.capacity()) {
            index.reserve(index_limits_t {NextPowerOfTwo(size), executor.size()});
        }
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        executor.fixed(vectors.size(), [&](std::size_t thread, std::size_t task) {
            try {
                int id = ids[task];
                auto& vec = vectors[task];
                
                auto start_time = std::chrono::high_resolution_clock::now();
                index.add(id, vec.data(), thread);
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double>(end_time - start_time).count();
                
                success_count++;
                
                std::lock_guard<std::mutex> lock(bench_mutex);
                benchmarks.push_back({dataset_name, iteration, duration});
            }
            catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(bench_mutex);
                std::cerr << "Error adding vector " << task << ": " << e.what() << std::endl;
            }
        });
        
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
 * @param sample_vecs The result set containing vectors to remove
 * @param dataset_name The name of the dataset for benchmarking
 * @param iteration The current iteration number
 * @param del_bm_appender Appender for benchmarking results
 * @return Number of vectors successfully deleted
 */

size_t IndexOperations::singleRemove(
    index_dense_gt<row_t>& index,
    const unique_ptr<MaterializedQueryResult>& sample_vecs,
    const std::string& dataset_name,
    int iteration,
    Appender& del_bm_appender
) {
    std::cout << "🔵 DELETING SAMPLE VECTORS 🔵" << std::endl;

    size_t removed_count = 0;

    auto batch_start = std::chrono::high_resolution_clock::now();

    try {
        for (idx_t i = 0; i < sample_vecs->RowCount(); i++) {
            auto id = sample_vecs->GetValue<int>(0, i);
            auto start_time = std::chrono::high_resolution_clock::now();
            index.remove(id);
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
size_t IndexOperations::parallelRemove(
    index_dense_gt<row_t>& index,
    const unique_ptr<MaterializedQueryResult>& sample_vecs,
    const std::string& dataset_name,
    int iteration,
    Appender& del_bm_appender
) {
    std::cout << "🔵 DELETING SAMPLE VECTORS 🔵" << std::endl;
    
    size_t removed_count = 0;
    
    try {
        // Collect all IDs to delete upfront
        std::vector<int> ids_to_delete;
        ids_to_delete.reserve(sample_vecs->RowCount());
        
        for (idx_t i = 0; i < sample_vecs->RowCount(); i++) {
            ids_to_delete.push_back(sample_vecs->GetValue<int>(0, i));
        }

        // Create mutex and result collection
        std::mutex benchmark_mutex;
        std::vector<std::tuple<std::string, int, double>> benchmarks;
        std::atomic<size_t> success_count(0);
        
        // Determine batch size and thread count
        std::size_t total_vectors = ids_to_delete.size();
        std::size_t hardware_threads = std::thread::hardware_concurrency();
        std::size_t max_threads = std::min(hardware_threads, static_cast<std::size_t>(total_vectors));
        
        // Use fewer threads but batch delete operations for better performance
        std::size_t effective_threads = std::max(std::size_t(1), max_threads / 2);
        std::size_t batch_size = std::max(std::size_t(1), total_vectors / effective_threads);
        
        std::cout << "Starting optimized parallel delete with " << effective_threads 
                << " threads, batch size: " << batch_size << std::endl;
        
        // Create thread pool
        executor_default_t executor(effective_threads);
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        // Process in batches for better performance
        std::vector<std::size_t> batch_starts;
        for (std::size_t i = 0; i < total_vectors; i += batch_size) {
            batch_starts.push_back(i);
        }
        
        // Process batches in parallel
        executor.fixed(batch_starts.size(), [&](std::size_t thread, std::size_t batch_idx) {
            std::size_t start_idx = batch_starts[batch_idx];
            std::size_t end_idx = std::min(start_idx + batch_size, total_vectors);
            
            // Process each item in this batch
            for (std::size_t i = start_idx; i < end_idx; i++) {
                try {
                    int id = ids_to_delete[i];
                    
                    auto start_time = std::chrono::high_resolution_clock::now();
                    auto result = index.remove(id);
                    auto end_time = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration<double>(end_time - start_time).count();
                    
                    if (result.completed) {
                        success_count++;
                    }
                    
                    // Lock only when updating shared data
                    std::lock_guard<std::mutex> lock(benchmark_mutex);
                    benchmarks.push_back({dataset_name, iteration, duration});
                }
                catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(benchmark_mutex);
                    std::cerr << "Error deleting vector " << i << ": " << e.what() << std::endl;
                }
            }
        });
        
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
        
        removed_count = success_count;
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
void IndexOperations::runTestQueries(Connection& con, index_dense_gt<row_t>& index, const std::string& table_name,
    const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, Appender& search_appender, Appender& early_term_appender, int iteration, int dataset_size) {
    std::cout << "🧪 RUNNING TEST QUERIES 🧪" << std::endl;

    std::vector<index_dense_gt<row_t>::search_result_t> search_results;
    search_results.reserve(test_vectors->RowCount());

    // Run KNN search for each test vector
    for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
        try {
            auto test_vec = ExtractFloatVector(test_vectors->GetValue(1, i));
            auto test_neighbor_ids = ExtractSizeVector(test_vectors->GetValue(2, i));
            int test_query_vector_index_int = test_vectors->GetValue(0, i).GetValue<int>();
            Value neighbor_ids = test_vectors->GetValue(2, i);

            auto start_time = std::chrono::high_resolution_clock::now();
            auto results = index.search(&test_vec[0], 100); // TODO wanted hardcoded
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration<double>(end_time - start_time).count();

            if(results.size() != 0) {
                std::vector<Value> result_vec_ids;
                result_vec_ids.reserve(results.size());

                // Create result value directly
                Value result_list_value;

                for (std::size_t j = 0; j < results.size(); ++j){
                    size_t key = static_cast<size_t>(results[j].member.key);
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
                        Value(),                                        // recall (computed later)
                        Value::INTEGER(results.computed_distances),     // computed distances
                        Value::INTEGER(results.visited_members),        // visited members
                        Value::INTEGER(results.count)                   // count
                    );
                } catch (const std::exception& e) {
                    std::cerr << "Error appending row: " << e.what() << std::endl;
                    continue;
                }

                // If < 100 results are returned, append search query stats to early termination table
                if (results.size() < 100) {
                    try {
                        early_term_appender.AppendRow(
                            Value(table_name),                              // dataset name
                            Value::INTEGER(iteration),                      // iteration number
                            Value::INTEGER(test_query_vector_index_int),    // test vector ID
                            neighbor_ids,                                   // ground truth neighbors
                            result_list_value,                              // result vector IDs
                            Value(),                                        // recall (computed later)
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
                        Value(),                                        // recall (computed later)
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
                        Value(),                                        // recall (computed later)
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
 * Runs exact search on the index in parallel
 * 
 * @param con The DuckDB connection
 * @param index The USearch index to query
 * @param test_vectors The result set containing test vectors
 */
TestVectorData IndexOperations::parallelExactSearch(Connection& con, index_dense_gt<row_t>& index,
    const unique_ptr<MaterializedQueryResult>& test_vectors) {
    
    TestVectorData data;

    try {
        // Extract all test vectors upfront to avoid DB access in threads
        std::vector<std::vector<float>> test_vecs;
        std::vector<int> test_vector_indices;

        test_vecs.reserve(test_vectors->RowCount());
        test_vector_indices.reserve(test_vectors->RowCount());

        for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
            test_vecs.push_back(ExtractFloatVector(test_vectors->GetValue(1, i)));
            test_vector_indices.push_back(test_vectors->GetValue(0, i).GetValue<int>());
        }

        // Thread-safe containers for results
        std::mutex results_mutex;

        data.test_vecs = test_vecs;  // Copy vectors to result
        data.test_vector_indices = test_vector_indices;  // Copy indices to result
        data.neighbor_ids_values.reserve(test_vectors->RowCount());

        std::size_t executor_threads = std::min(std::thread::hardware_concurrency(), 
                                              static_cast<unsigned int>(test_vectors->RowCount()));
        executor_default_t executor(executor_threads);
        
        std::cout << "Starting parallel exact search with " << executor_threads << " threads" << std::endl;
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        executor.fixed(test_vecs.size(), [&](std::size_t thread, std::size_t task) {
            try {
                auto& test_vec = test_vecs[task];
                int test_query_vector_index_int = test_vector_indices[task];

                auto results = index.search(test_vec.data(), 100, thread, true); // Get 100 nearest neighbors

                if(results.size() != 0) {
                    // Create result value directly
                    std::vector<Value> id_values;
                    id_values.reserve(results.size());
                    
                    for (std::size_t j = 0; j < results.size(); ++j) {
                        size_t key = static_cast<size_t>(results[j].member.key);
                        id_values.push_back(Value::INTEGER(key));
                    }
                    
                    Value result_list_value = Value::LIST(LogicalType::INTEGER, std::move(id_values));

                    // Thread-safe collection of results
                    {
                        std::lock_guard<std::mutex> lock(results_mutex);
                        
                        // Store neighbor IDs for this vector
                        // Use task as index to ensure we put values in the right position
                        if (task >= data.neighbor_ids_values.size()) {
                            data.neighbor_ids_values.resize(task + 1);
                        }
                        data.neighbor_ids_values[task] = result_list_value;
                    }
                } else {
                    std::lock_guard<std::mutex> lock(results_mutex);
                    std::cerr << "No results returned for test vector " << task << " during exact search" << std::endl;
                    
                    // Add an empty entry for this vector
                    if (task >= data.neighbor_ids_values.size()) {
                        data.neighbor_ids_values.resize(task + 1);
                    }
                    data.neighbor_ids_values[task] = Value();  // Empty value
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(results_mutex);
                std::cerr << "Error processing test vector " << task << ": " << e.what() << std::endl;
            }
        });
        
        auto batch_end = std::chrono::high_resolution_clock::now();
        auto batch_duration = std::chrono::duration<double>(batch_end - batch_start).count();
        std::cout << "Parallel exact search completed in " << batch_duration << "s" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error in parallel exact search: " << e.what() << std::endl;
    }
    
    return data;
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
void IndexOperations::parallelRunTestQueries(Connection& con, index_dense_gt<row_t>& index, const std::string& table_name,
    const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, Appender& search_appender, 
    Appender& early_term_appender, int iteration, int dataset_size, bool get_neighbors) {
    std::cout << "🧪 RUNNING TEST QUERIES 🧪" << std::endl;

    try {
        // Extract all test vectors upfront to avoid DB access in threads
        std::vector<std::vector<float>> test_vecs;
        std::vector<int> test_vector_indices;
        std::vector<Value> neighbor_ids_values;

        test_vecs.reserve(test_vectors->RowCount());
        test_vector_indices.reserve(test_vectors->RowCount());
        neighbor_ids_values.reserve(test_vectors->RowCount());

        // For new data scenario we have to get the neighbors for each iteration
        if (get_neighbors) {
            TestVectorData data = parallelExactSearch(con, index, test_vectors);
            test_vecs = data.test_vecs;
            test_vector_indices = data.test_vector_indices;
            neighbor_ids_values = data.neighbor_ids_values;
        } else {
            for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
                test_vecs.push_back(ExtractFloatVector(test_vectors->GetValue(1, i)));
                test_vector_indices.push_back(test_vectors->GetValue(0, i).GetValue<int>());
                neighbor_ids_values.push_back(test_vectors->GetValue(2, i));
            }
        }

        // Thread-safe containers for results
        std::mutex results_mutex;
        std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>> search_results;
        std::vector<std::tuple<std::string, int, int, Value, Value, Value, int, int, int>> early_term_results;
        std::vector<std::tuple<std::string, int, double>> search_benchmarks;

        std::size_t executor_threads = std::min(std::thread::hardware_concurrency(), 
                                                static_cast<unsigned int>(test_vectors->RowCount()));
        executor_default_t executor(executor_threads);
        
        std::cout << "Starting parallel search with " << executor_threads << " threads" << std::endl;
        
        auto batch_start = std::chrono::high_resolution_clock::now();
        
        executor.fixed(test_vecs.size(), [&](std::size_t thread, std::size_t task) {
            try {
                auto& test_vec = test_vecs[task];
                int test_query_vector_index_int = test_vector_indices[task];
                const Value& neighbor_ids = neighbor_ids_values[task];

                auto start_time = std::chrono::high_resolution_clock::now();
                auto results = index.search(test_vec.data(), 100, thread); 
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration<double>(end_time - start_time).count();

                unique_ptr<MaterializedQueryResult> result_ids = nullptr;

                if(results.size() != 0) {
                    std::vector<size_t> result_vec_ids;
                    result_vec_ids.reserve(results.size());

                    for (std::size_t j = 0; j < results.size(); ++j) {
                        size_t key = static_cast<size_t>(results[j].member.key);
                        result_vec_ids.push_back(key);
                    }

                    // Create result value directly
                    Value result_list_value;

                    std::vector<Value> id_values;
                    id_values.reserve(results.size());
                    
                    for (std::size_t j = 0; j < results.size(); ++j) {
                        size_t key = static_cast<size_t>(results[j].member.key);
                        id_values.push_back(Value::INTEGER(key));
                    }
                    
                    result_list_value =  Value::LIST(LogicalType::INTEGER, std::move(id_values));
                
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
                            results.computed_distances,
                            results.visited_members,
                            results.count
                        });

                        // Store benchmark data
                        search_benchmarks.push_back({
                            table_name,
                            iteration,
                            duration
                        });

                        // Handle early termination case
                        if (results.size() < 100) {
                            early_term_results.push_back({
                                table_name,
                                iteration,
                                test_query_vector_index_int,
                                neighbor_ids,
                                result_list_value,
                                Value::FLOAT(0.0), // recall (calculated later)
                                results.computed_distances,
                                results.visited_members,
                                results.count
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
                        results.computed_distances,
                        results.visited_members,
                        results.count
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
                        results.computed_distances,
                        results.visited_members,
                        results.count
                    });
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(results_mutex);
                std::cerr << "Error processing test vector " << task << ": " << e.what() << std::endl;
            }
        });
        
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
    

