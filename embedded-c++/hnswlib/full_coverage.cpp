#include <hnswlib/hnswlib.h>
#include "duckdb.hpp"
#include "hnswlib/helpers/hnswlib_index_operations.h"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/query_runner.h"
#include "usearch/helpers/file_operations.h"
#include "hnswlib/helpers/util.h"

using namespace duckdb;
using namespace hnswlib;

std::string experiment;

// ==================== Main Full Coverage USearch Runner ====================
class HNSWLibFullCoverageRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;
    int threads;

public:
HNSWLibFullCoverageRunner(int iterations, int threads) : db(nullptr), con(db), max_iterations(iterations) {
        con.Query("SET THREADS TO " + std::to_string(threads) + ";");
        datasets = DatabaseSetup::getDatasetConfigs();
    }

    void runTest(int datasetIdx) {
        try {
            // Cleanup intermediate files
            FileOperations::cleanupOutputFiles(std::filesystem::current_path());

            // Limit to valid dataset indices
            if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
                std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
                return;
            }

            const auto& dataset = datasets[datasetIdx];

            std::cout << "📊 TESTING DATASET: " << dataset.name << " 📊" << std::endl;

            // Setup database tables
            DatabaseSetup::initializeResultsTable(con, dataset.name);
            DatabaseSetup::initializeBMTable(con, dataset.name + "_del");
            DatabaseSetup::initializeBMTable(con, dataset.name + "_add");
            DatabaseSetup::initializeBMTable(con, dataset.name + "_search");
            DatabaseSetup::intializeEarlyTermTable(con);
            DatabaseSetup::setupFullDataset(con, dataset);

            // Load the hnswlib index
            auto dataset_cardinality = con.Query("SELECT COUNT(*) FROM " + dataset.name + "_train;")->GetValue<int64_t>(0, 0);
            L2Space space(dataset.dimensions);
            std::string index_path = "hnswlib/indexes/" + dataset.name + "_index.bin";
            HierarchicalNSW<float> index(&space, index_path, false, dataset_cardinality, true);

            // Load the index_map
            std::string index_map_path = "hnswlib/indexes/" + dataset.name + "_index_map.txt";
            std::unordered_map<size_t, size_t> index_map;
            index_map.reserve(dataset_cardinality);
            std::ifstream index_map_file(index_map_path);
            size_t key, value;
            while (index_map_file >> key >> value) {
                index_map[key] = value;
            }
            index_map_file.close();
            
            // Get test vectors
            auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_test;");
            auto test_vectors_count = test_vectors->RowCount();

            // Dataset vectors
            auto dataset_vectors = con.Query("SELECT * FROM " + dataset.name + "_train;");

            // Create appender for results
            Appender appender(con, dataset.name + "_results");
            Appender del_bm_appender(con, dataset.name + "_del_bm");
            Appender add_bm_appender(con, dataset.name + "_add_bm");
            Appender search_bm_appender(con, dataset.name + "_search_bm");
            Appender early_termination_appender(con, "early_terminated_queries");

            // Initial query run (multi-threaded)
            HNSWLibIndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, search_bm_appender, early_termination_appender, 0, dataset_cardinality, index_map);

            auto partitions = QueryRunner::partitionDataset(con, dataset.name, 100);

            // Log initial index stats
            index.log_memory_stats();
            index.log_connectivity_stats(&space);
            

            // Run iterations
            size_t last_idx = 0;
            for (int iteration = 1; iteration <= partitions.size(); iteration++) {
                std::cout << "▶️ ITERATION " << iteration << " ▶️" << std::endl;

                // Get vectors from partition
                auto& partition = partitions[iteration-1];


                std::unordered_set<size_t> delete_indices_set;

                size_t start_idx = last_idx;
                auto num_to_delete = partition->RowCount();
                last_idx = start_idx+num_to_delete;
                
                for (size_t idx = start_idx; idx < start_idx+num_to_delete; ++idx) {
                    delete_indices_set.insert(idx);
                }

                std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());

                // save the vectors being deleted before deleting them
                std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dataset.dimensions));
                for (size_t i = 0; i < delete_indices.size(); ++i) {
                    size_t idx = delete_indices[i];
                    deleted_vectors[i] = ExtractFloatVector(dataset_vectors->GetValue(1, idx));
                }

                // Delete vectors from this partition to the index
                size_t removed = HNSWLibIndexOperations::singleRemove(index, delete_indices, index_map, dataset.name, 
                                                            iteration, del_bm_appender);

                // Re-add the deleted vectors with their original labels
                std::vector<size_t> new_indices(delete_indices.size());
                for (size_t i = 0; i < delete_indices.size(); ++i) {
                    size_t idx = index_map[delete_indices[i]];
                    size_t new_idx = (idx < dataset_cardinality) ? idx + dataset_cardinality : idx - dataset_cardinality;
                    new_indices[i] = new_idx;
                    index_map[delete_indices[i]] = new_idx;
                }
                
                // Re-add vectors from this partition to the index
                size_t added = HNSWLibIndexOperations::parallelAdd(index, deleted_vectors, new_indices,  dataset.name, 
                                                        iteration, add_bm_appender, threads);

                // Log index stats
                index.log_memory_stats();
                index.log_connectivity_stats(&space);
           
                // Run test queries (multi-threaded)
                HNSWLibIndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, search_bm_appender, early_termination_appender, iteration, dataset_cardinality, index_map);

                std::cout << "✅ FINISHED ITERATION " << iteration << " ✅" << std::endl;

            }
        
            appender.Close();
            early_termination_appender.Close();
            del_bm_appender.Close();
            add_bm_appender.Close();
            search_bm_appender.Close();

            // Calculate recall and aggregate stats
            QueryRunner::calculateRecall(con, dataset.name);
            QueryRunner::aggregateRecallStats(con, dataset.name);

            // Aggregate bm stats
            QueryRunner::aggregateBMStats(con, dataset.name + "_del", test_vectors_count, (dataset_cardinality/partitions.size()));
            QueryRunner::aggregateBMStats(con, dataset.name + "_add", test_vectors_count, (dataset_cardinality/partitions.size()));
            QueryRunner::aggregateBMStats(con, dataset.name + "_search", test_vectors_count, (dataset_cardinality/partitions.size()));

            // Output experiment results to CSV
            // dir name: usearch/results/{experiment}/{dataset_name}_{num_queries}q_{num_iterations}i_{partition_size}p/
            std::string output_dir = "hnswlib/results/fullcoverage/" + experiment + dataset.name + "_" + std::to_string(test_vectors_count) + "q_" + std::to_string(max_iterations) + "i_" + std::to_string((int) (dataset_cardinality/partitions.size())) + "r/";
            // Create the directory if it doesn't exist
            std::filesystem::create_directories(output_dir);
            FileOperations::cleanupOutputFiles(output_dir);
            QueryRunner::outputTableAsCSV(con, "recall_stats", output_dir + "search_query_stats.csv");
            QueryRunner::outputTableAsCSV(con, "early_terminated_queries", output_dir + "early_terminated_queries.csv");
            QueryRunner::outputTableAsCSV(con, dataset.name + "_del_bm_stats", output_dir + "bm_delete.csv");
            QueryRunner::outputTableAsCSV(con, dataset.name + "_add_bm_stats", output_dir + "bm_add.csv");
            QueryRunner::outputTableAsCSV(con, dataset.name + "_search_bm_stats", output_dir + "bm_search.csv");

            // Move lib output files to output dir
            FileOperations::copyFileTo("node_connectivity.csv", output_dir + "node_connectivity.csv");
            FileOperations::copyFileTo("memory_stats.csv", output_dir + "memory_stats.csv");

            // Save the final index
            std::string s_path = output_dir + "full_coverage_" + dataset.name + "_index.bin";
            index.saveIndex(s_path);
            std::cout << "Index saved to: " << s_path << std::endl;

            // Cleanup intermediate files
            FileOperations::cleanupOutputFiles(std::filesystem::current_path());

        } catch (std::exception& e) {
            std::cerr << "Error running test: " << e.what() << std::endl;
        }
    }
};

// ==================== Main Function ====================
int main() {

    /**
     * FULLCOVERAGE: 
     * 100 iterations where each dataset is segmented into 100 parts. Every
     * iteration involves the deletion and reinsertion of a portion, enabling 
     * the assessment of the impact of complete coverage on the index structure 
     * and performance.
     */
    
    int max_iterations = 100;
    std::size_t executor_threads = (std::thread::hardware_concurrency());

    experiment = "hnswlib_";

    try {
        // fashion_mnist
        HNSWLibFullCoverageRunner fm_runner(max_iterations, executor_threads);
        fm_runner.runTest(0);

        // mnist
        HNSWLibFullCoverageRunner m_runner(max_iterations, executor_threads);
        fm_runner.runTest(1);

          // sift
        HNSWLibFullCoverageRunner s_runner(max_iterations, executor_threads);
        fm_runner.runTest(2);

        // gist
        HNSWLibFullCoverageRunner g_runner(max_iterations, executor_threads);
        fm_runner.runTest(3);

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
