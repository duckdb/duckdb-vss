#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include "duckdb.hpp"
#include "usearch/helpers/index_operations.h"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/query_runner.h"
#include "usearch/helpers/file_operations.h"

using namespace duckdb;
using namespace unum::usearch;

std::string experiment;

// ==================== Main New Data USearch Runner ====================
class USearchNewDataRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;
    int threads;

public:
USearchNewDataRunner(int iterations, int threads) : db(nullptr), con(db), max_iterations(iterations), threads(threads) {
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
            DatabaseSetup::setupGroundTruthTable(con, dataset.name, dataset.dimensions);

            // Create the usearch index
            std::size_t vector_size = dataset.dimensions;
            auto scalar_kind = scalar_kind_t::f32_k;
            auto metric_kind = metric_kind_t::l2sq_k;

            metric_punned_t metric(vector_size, metric_kind, scalar_kind);
            index_dense_config_t config = {};

            // Set M and efConstruction based on dataset
            config.expansion_add = dataset.ef_construction;
            config.connectivity = dataset.m;

            auto index = index_dense_gt<row_t>::make(metric, config);

            auto dataset_cardinality = con.Query("SELECT COUNT(*) FROM " + dataset.name + "_train;")->GetValue<int64_t>(0, 0);
            // Partition dataset into 20
            auto partitions = QueryRunner::partitionDataset(con, dataset.name, 20);

            // Load half index (first 10 partitions)
            std::string path = "usearch/indexes/" + dataset.name + "_index_half.usearch";
            index.load(path.c_str());

            // Log initial index stats
            index.log_links();
            
            // Create appender for results
            Appender appender(con, dataset.name + "_results");
            Appender del_bm_appender(con, dataset.name + "_del_bm");
            Appender add_bm_appender(con, dataset.name + "_add_bm");
            Appender search_bm_appender(con, dataset.name + "_search_bm");
            Appender early_termination_appender(con, "early_terminated_queries");

            // Get current keys in index from partitions 1- 10
            std::unordered_set<size_t> current_idx_keys_set;
            current_idx_keys_set.reserve(dataset_cardinality/2);
            for (int i = 0; i < 10; i++) {
                for (idx_t j = 0; j < partitions[i]->RowCount(); j++) {
                    current_idx_keys_set.insert(partitions[i]->GetValue<int>(0, j));
                }
            }
            assert(current_idx_keys_set.size() == (dataset_cardinality/2));

            // Get test vectors with ground truth neighbor ids (from brute force knn)
            auto test_vectors = QueryRunner::getCurrentTopKNeighbors(con, dataset.name, current_idx_keys_set);
            auto test_vectors_count = test_vectors->RowCount();

            // Initial query run (multi-threaded)
            IndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, search_bm_appender, early_termination_appender, 0, dataset_cardinality);

            // Run iterations
            for (int iteration = 1; iteration <= 10; iteration++) {
                std::cout << "▶️ ITERATION " << iteration << " ▶️" << std::endl;

                // Get vectors from partition
                auto& partitions_to_remove = partitions[iteration-1];
                auto& partitions_to_add = partitions[9 + iteration];
                
                // Delete vectors from first half
                size_t removed = IndexOperations::singleRemove(index, partitions_to_remove, dataset.name, 
                                                            iteration, del_bm_appender);
                
                // Add vectors from second half
                size_t added = IndexOperations::parallelAdd(index, partitions_to_add, dataset.name, 
                                                        iteration, add_bm_appender);

                // Log index stats
                index.log_links();

                // Remove vectors from current_idx_keys_set
                for (int i = 0; i < partitions_to_remove->RowCount(); i++) {
                    current_idx_keys_set.erase(partitions_to_remove->GetValue<int>(0, i));
                }
                // Add new indices to current_idx_keys_set
                for (int i = 0; i < partitions_to_add->RowCount(); i++) {
                    current_idx_keys_set.insert(partitions_to_add->GetValue<int>(0, i));
                }

                assert(current_idx_keys_set.size() == (dataset_cardinality/2));

                // Get current top 100 neighbors
                auto current_top_100_neighbors = QueryRunner::getCurrentTopKNeighbors(con, dataset.name, current_idx_keys_set);

                // Run test queries (multi-threaded)
                IndexOperations::parallelRunTestQueries(con, index, dataset.name, current_top_100_neighbors, appender, 
                                        search_bm_appender, early_termination_appender, 
                                        iteration, dataset_cardinality);

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
            // dir name: usearch/results/{experiment}/{dataset_name}_{num_queries}q_{num_iterations}i_{partition_size}r/
            std::string output_dir = "usearch/results/newdata/" + experiment + dataset.name + "_" + std::to_string(test_vectors_count) + "q_" + std::to_string(max_iterations) + "i_" + std::to_string((int) (dataset_cardinality/partitions.size())) + "r/";
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
            FileOperations::copyFileTo("node_neighbors.csv", output_dir + "node_neighbors.csv");

            // Cleanup intermediate files
            FileOperations::cleanupOutputFiles(std::filesystem::current_path());

            // Save the final index
            std::string s_path = output_dir + "new_data_" + dataset.name + "_index.usearch";
            index.save(s_path.c_str());
            std::cout << "Index saved to: " << s_path << std::endl;

        } catch (std::exception& e) {
            std::cerr << "Error running test: " << e.what() << std::endl;
        }
    }
};

// ==================== Main Function ====================
int main() {

    /**
     * NEWDATA: 
     * initialize the index with the first 1/2 data points and conduct 10 
     * iterations. Each iteration deletes 10% points from the first 1/2 and 
     * inserts 10% points from the second half. This process aims to assess the 
     * index’s performance and adaptability with continuous data introduction.
     * The final index consists of the second half data points.
     */
    
    int max_iterations = 10;
    std::size_t executor_threads = (std::thread::hardware_concurrency());
    
    // original_ - tests original USearch implementation w/o changing source code
    experiment = "usearch_";

    try {
        // fashion_mnist
        USearchNewDataRunner fm_runner(max_iterations, executor_threads);
        fm_runner.runTest(0);

        // mnist
        USearchNewDataRunner m_runner(max_iterations, executor_threads);
        m_runner.runTest(1);

        // sift
        USearchNewDataRunner s_runner(max_iterations, executor_threads);
        s_runner.runTest(2);

        // gist
        USearchNewDataRunner g_runner(max_iterations, executor_threads);
        g_runner.runTest(3);

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
