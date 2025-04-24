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

// ==================== Main Random USearch Runner ====================
class USearchRandomRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;
    int threads;

public:
USearchRandomRunner(int iterations = 200, int threads = 64) : db(nullptr), con(db), max_iterations(iterations) {
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

            std::size_t executor_threads = std::min(std::thread::hardware_concurrency(),
                                                static_cast<unsigned int>(dataset_cardinality));
            executor_default_t executor(executor_threads);

            std::string path = "usearch/indexes/" + dataset.name + "_index.usearch";
            index.load(path.c_str());

            // Log initial index stats
            index.log_links();

            // Get test vectors
            auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_test;");
            auto test_vectors_count = test_vectors->RowCount();

            // TODO: hardcoded value
            auto perc = 0.01;
            std::ostringstream perc_str;
            perc_str << std::fixed << std::setprecision(2) << perc;
            auto sample_size = perc * dataset_cardinality;

            // Create appender for results
            Appender appender(con, dataset.name + "_results");
            Appender del_bm_appender(con, dataset.name + "_del_bm");
            Appender add_bm_appender(con, dataset.name + "_add_bm");
            Appender search_bm_appender(con, dataset.name + "_search_bm");
            Appender early_termination_appender(con, "early_terminated_queries");

            // Initial query run (multi-threaded)
            IndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, search_bm_appender, early_termination_appender, 0, dataset_cardinality);

            // Run iterations
            for (int iteration = 1; iteration <= max_iterations; iteration++) {
                std::cout << "▶️ ITERATION " << iteration << " ▶️" << std::endl;

                // Get sample vectors to delete and re-add
                auto sample_vecs = QueryRunner::getSampleVectors(con, dataset.name, sample_size);

                // Delete sample vectors
                size_t removed = IndexOperations::singleRemove(index, sample_vecs, dataset.name, 
                                                            iteration, del_bm_appender);
                
                // Re-add sample vectors (multi-threaded)
                size_t added = IndexOperations::parallelAdd(index, sample_vecs, dataset.name, 
                                                        iteration, add_bm_appender);

                // Log index stats
                index.log_links();

                // Run test queries (multi-threaded)
                IndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, 
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
            QueryRunner::aggregateBMStats(con, dataset.name + "_del", test_vectors_count, sample_size);
            QueryRunner::aggregateBMStats(con, dataset.name + "_add", test_vectors_count, sample_size);
            QueryRunner::aggregateBMStats(con, dataset.name + "_search", test_vectors_count, sample_size);

            // Output experiment results to CSV
            // dir name: usearch/results/{experiment}/{dataset_name}_{num_queries}q_{num_iterations}i_{sample_size}r/
            std::string output_dir = "usearch/results/random/" + experiment + dataset.name + "_" + std::to_string(test_vectors_count) + "q_" + std::to_string(max_iterations) + "i_" + std::to_string(sample_size) + "r/";
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
            std::string s_path = output_dir + "random_" + dataset.name + "_index.usearch";
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
     * RANDOM: 
     * 200 iterations. Within each iteration, 1% of the vectors are randomly 
     * generated for deletion and reinsertion, facilitating the evaluation of 
     * method performance and robustness in the face of random data 
     * manipulations.
     */
    
    int max_iterations = 200;
    int threads = 32;
    
    // original_ - tests original USearch implementation w/o changing source code
    experiment = "usearch_";

    try {
        // fashion_mnist
        USearchRandomRunner fm_runner(max_iterations, threads);
        fm_runner.runTest(0);

        // mnist
        USearchRandomRunner m_runner(max_iterations, threads);
        m_runner.runTest(1);

        // sift
        USearchRandomRunner s_runner(max_iterations, threads);
        s_runner.runTest(2);

        // gist
        USearchRandomRunner g_runner(max_iterations, threads);
        g_runner.runTest(3);

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
