#include <hnswlib/hnswlib.h>
#include "duckdb.hpp"
#include "hnswlib/helpers/hnswlib_index_operations.h"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/query_runner.h"
#include "usearch/helpers/file_operations.h"

using namespace duckdb;
using namespace hnswlib;

std::string experiment;

// ==================== Main New Data USearch Runner ====================
class HNSWLibNewDataRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;
    int threads;

public:
HNSWLibNewDataRunner(int iterations = 10, int threads = 32) : db(nullptr), con(db), max_iterations(iterations) {
        con.Query("SET THREADS TO " + std::to_string(threads) + ";");
        datasets = DatabaseSetup::getDatasetConfigs();
    }

    void runTest(int datasetIdx = 0) {
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

            // Load the hnswlib index
            auto dataset_cardinality = con.Query("SELECT COUNT(*) FROM " + dataset.name + "_train;")->GetValue<int64_t>(0, 0);
            L2Space space(dataset.dimensions);
            std::string index_path = "hnswlib/indexes/" + dataset.name + "_index_half.bin";
            HierarchicalNSW<float> index(&space, index_path, false, dataset_cardinality/2, true);

            // Load the index_map
            std::string index_map_path = "hnswlib/indexes/" + dataset.name + "_index_map_half.txt";
            std::unordered_map<size_t, size_t> index_map;
            index_map.reserve(dataset_cardinality/2);
            std::ifstream index_map_file(index_map_path);
            size_t key, value;
            while (index_map_file >> key >> value) {
                index_map[key] = value;
            }
            index_map_file.close();

            // Partition dataset into 20
            auto partitions = QueryRunner::partitionDataset(con, dataset.name, 20);
            auto half_count = dataset_cardinality / 2;
                        
            // Log initial index stats
            index.log_memory_stats();
            index.log_connectivity_stats(&space);
            
            // Get test vectors with ground truth neighbor ids (from brute force knn)
            auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_ground_truth;");
            auto test_vectors_count = con.Query("SELECT distinct id FROM " + dataset.name + "_ground_truth;")->RowCount();

            // Create appender for results
            Appender appender(con, dataset.name + "_results");
            Appender del_bm_appender(con, dataset.name + "_del_bm");
            Appender add_bm_appender(con, dataset.name + "_add_bm");
            Appender search_bm_appender(con, dataset.name + "_search_bm");
            Appender early_termination_appender(con, "early_terminated_queries");

            // Initial query run (multi-threaded)
            HNSWLibIndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, search_bm_appender, early_termination_appender, 0, dataset_cardinality, index_map);

            // Run iteration
            
            for (int iteration = 1; iteration <= max_iterations; iteration++) {
                std::cout << "▶️ ITERATION " << iteration << " ▶️" << std::endl;

                // Get vectors from partition
                auto& partitions_to_remove = partitions[iteration-1];
                auto& partitions_to_add = partitions[9 + iteration];

                std::vector<size_t> delete_indices;
                delete_indices.reserve(partitions_to_remove->RowCount());
                for (idx_t i = 0; i < partitions_to_remove->RowCount(); i++) {
                    delete_indices.push_back(partitions_to_remove->GetValue<int>(0, i));
                }

                // Delete vectors from first half
                size_t removed = HNSWLibIndexOperations::singleRemove(index, delete_indices, index_map, dataset.name, 
                                                          iteration, del_bm_appender);
                
                std::vector<size_t> new_indices;
                new_indices.reserve(partitions_to_add->RowCount());
                std::vector<std::vector<float>> new_vectors;
                new_vectors.reserve(partitions_to_add->RowCount());
                for (idx_t i = 0; i < partitions_to_add->RowCount(); i++) {
                    new_indices.push_back(partitions_to_add->GetValue<int>(0, i));
                    new_vectors.push_back(ExtractFloatVector(partitions_to_add->GetValue(1, i)));
                }
                // Add vectors from second half
                size_t added = HNSWLibIndexOperations::parallelAdd(index, new_vectors, new_indices,  dataset.name, 
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
            std::string output_dir = "hnswlib/results/newdata/" +  experiment + dataset.name + "_" + std::to_string(test_vectors_count) + "q_" + std::to_string(max_iterations) + "i_" + std::to_string(dataset_cardinality/partitions.size()) + "r/";
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
            std::string s_path = output_dir + "new_data_" + dataset.name + "_index.bin";
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
     * NEWDATA: 
     * initialize the index with the first 1/2 data points and conduct 10 
     * iterations. Each iteration deletes 10% points from the first 1/2 and 
     * inserts 10% points from the second half. This process aims to assess the 
     * index’s performance and adaptability with continuous data introduction.
     * The final index consists of the second half data points.
     */
    
    int max_iterations = 10;
    int threads = 32;

    experiment = "hnswlib_";

    try {
        // fashion_mnist
        HNSWLibNewDataRunner fm_runner(max_iterations, threads);
        fm_runner.runTest(0);

        // mnist
        HNSWLibNewDataRunner m_runner(max_iterations, threads);
        fm_runner.runTest(1);

          // sift
          HNSWLibNewDataRunner s_runner(max_iterations, threads);
        fm_runner.runTest(2);

        // gist
        HNSWLibNewDataRunner g_runner(max_iterations, threads);
        fm_runner.runTest(3);

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
