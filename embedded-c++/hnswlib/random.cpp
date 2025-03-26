#include <hnswlib/hnswlib.h>
#include "duckdb.hpp"
#include "hnswlib/helpers/hnswlib_index_operations.h"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/query_runner.h"
#include "usearch/helpers/file_operations.h"

using namespace duckdb;
using namespace hnswlib;

// ==================== Main Random USearch Runner ====================
class HNSWLibRandomRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;
    int threads;

public:
HNSWLibRandomRunner(int iterations = 200, int threads = 64) : db(nullptr), con(db), max_iterations(iterations) {
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
            // Create the hnswlib index

            auto dataset_cardinality = con.Query("SELECT COUNT(*) FROM " + dataset.name + "_train;")->GetValue<int64_t>(0, 0);
            L2Space space(dataset.dimensions);
            std::cout << "Dataset size: " << dataset_cardinality << std::endl;
            HierarchicalNSW<float> index(&space, dataset_cardinality, dataset.m, dataset.ef_construction, 100, true);

            std::vector<int> ids;
            std::vector<std::vector<float>> vectors;
            ids.reserve(dataset_cardinality);
            vectors.reserve(dataset_cardinality);

            auto dataset_vectors = con.Query("SELECT * FROM " + dataset.name + "_train;");
            
            for (idx_t i = 0; i < dataset_cardinality; i++) {
                ids.push_back(dataset_vectors->GetValue<int>(0, i));
                vectors.push_back(ExtractFloatVector(dataset_vectors->GetValue(1, i)));
            }

            for (std::size_t task = 0; task < dataset_cardinality; ++task) {
                try {
                    int id = ids[task];
                    auto& vec = vectors[task];
                    
                    index.addPoint(vec.data(), id);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error adding vector " << task << ": " << e.what() << std::endl;
                }
            };

            std::unordered_map<size_t, size_t> index_map;
            std::cout << "Mapping index of size " << ids.size() << std::endl;
            for (size_t i = 0; i < ids.size(); ++i) {
                index_map[i] = ids[i];
            }

            // Log initial index stats
            //index.log_links();

            // Get test vectors
            auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_test;");
            auto test_vectors_count = test_vectors->RowCount();

            // TODO: hardcoded value
            auto perc = 0.01;
            std::ostringstream perc_str;
            perc_str << std::fixed << std::setprecision(2) << perc;
            std::string perc_formatted = perc_str.str();
            auto sample_size = perc * dataset_cardinality;

            // Create appender for results
            Appender appender(con, dataset.name + "_results");
            Appender del_bm_appender(con, dataset.name + "_del_bm");
            Appender add_bm_appender(con, dataset.name + "_add_bm");
            Appender search_bm_appender(con, dataset.name + "_search_bm");
            Appender early_termination_appender(con, "early_terminated_queries");

            // Initial query run (multi-threaded)
            HNSWLibIndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, search_bm_appender, early_termination_appender, 0, dataset_cardinality, index_map);

            // Run iterations
            size_t last_idx = 0;
            for (int iteration = 1; iteration <= max_iterations; iteration++) {
                std::cout << "▶️ ITERATION " << iteration << " ▶️" << std::endl;


                // Get sample vectors to delete and re-add
                auto sample_vecs = QueryRunner::getSampleVectors(con, dataset.name, 1);



                std::unordered_set<size_t> delete_indices_set;

                
                auto num_to_delete = sample_vecs->RowCount();
                
                
                for (size_t idx = 0; idx < sample_vecs->RowCount(); ++idx) {
                    delete_indices_set.insert(sample_vecs->GetValue<int>(0, idx));
                }

                std::vector<size_t> delete_indices(delete_indices_set.begin(), delete_indices_set.end());


                std::cout << "size of test_vectors is: " << dataset_vectors->RowCount() << std::endl;


                std::vector<std::vector<float>> deleted_vectors(delete_indices.size(), std::vector<float>(dataset.dimensions));

                for (size_t i = 0; i < delete_indices.size(); ++i) {
                    size_t idx = delete_indices[i];
                    deleted_vectors[i] = ExtractFloatVector(dataset_vectors->GetValue(1, idx));
                }

                // Delete sample vectors (multi-threaded)
                size_t removed = HNSWLibIndexOperations::parallelRemove(index, delete_indices, index_map, dataset.name, 
                    iteration, del_bm_appender, threads);

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
                //index.log_links();

                // Run test queries (multi-threaded)
                HNSWLibIndexOperations::parallelRunTestQueries(con, index, dataset.name, test_vectors, appender, 
                                        search_bm_appender, early_termination_appender, 
                                        iteration, dataset_cardinality, index_map);

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
            // dir name: usearch/results/{experiment}/{dataset_name}_{num_queries}q_{num_iterations}i_{sample_fraction}s/
            std::string output_dir = "hnswlib/results/random/" + dataset.name + "_" + std::to_string(test_vectors_count) + "q_" + std::to_string(max_iterations) + "i_" + perc_formatted + "s/";
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
     * RANDOM: 
     * 200 iterations. Within each iteration, 1% of the vectors are randomly 
     * generated for deletion and reinsertion, facilitating the evaluation of 
     * method performance and robustness in the face of random data 
     * manipulations.
     */
    
    int max_iterations = 200;
    int threads = 32;

    try {
        // fashion_mnist
        HNSWLibRandomRunner fm_runner(max_iterations, threads);
        fm_runner.runTest(0);

        // mnist
        HNSWLibRandomRunner m_runner(max_iterations, threads);
        m_runner.runTest(1);

        // sift
        HNSWLibRandomRunner s_runner(max_iterations, threads);
        s_runner.runTest(2);

        // gist
        HNSWLibRandomRunner g_runner(max_iterations, threads);
        g_runner.runTest(3);

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
