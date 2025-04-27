#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include "duckdb.hpp"
#include "usearch/helpers/index_operations.h"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/query_runner.h"
#include "usearch/helpers/file_operations.h"

using namespace duckdb;
using namespace unum::usearch;

// ==================== USearch Index Creator ====================
class USearchIndexCreator {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int threads;

public:
USearchIndexCreator(int threads) : db(nullptr), con(db) {
        con.Query("SET THREADS TO " + std::to_string(threads) + ";");
        datasets = DatabaseSetup::getDatasetConfigs();
    }

    void createIndexes(int datasetIdx) {
        try {
            // Limit to valid dataset indices
            if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
                std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
                return;
            }

            const auto& dataset = datasets[datasetIdx];

            std::cout << "Creating first half of index for dataset: " << dataset.name << "..." << std::endl;

            // Setup database tables
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
		    config.connectivity_base = config.connectivity * 2;

            auto index = index_dense_gt<row_t>::make(metric, config);

            auto dataset_cardinality = con.Query("SELECT COUNT(*) FROM " + dataset.name + "_train;")->GetValue<int64_t>(0, 0);

            // Partition dataset into 20
            auto partitions = QueryRunner::partitionDataset(con, dataset.name, 20);

            auto half_count = dataset_cardinality / 2;

            std::size_t executor_threads = std::min(std::thread::hardware_concurrency(),
                                                static_cast<unsigned int>(half_count));
            executor_default_t executor(executor_threads);

            index.reserve(index_limits_t {NextPowerOfTwo(half_count), executor.size()});

            std::vector<int> ids;
            std::vector<std::vector<float>> vectors;
            ids.reserve(half_count);
            vectors.reserve(half_count);

            // Populate index with first half
            for (idx_t i = 0; i < 10; i++) {
                auto& partition = partitions[i];
                for (idx_t j = 0; j < partition->RowCount(); j++) {
                    ids.push_back(partition->GetValue<int>(0, j));
                    vectors.push_back(ExtractFloatVector(partition->GetValue(1, j)));
                }
            }

            executor.fixed(half_count, [&](std::size_t thread, std::size_t task) {
                try {
                    int id = ids[task];
                    auto& vec = vectors[task];

                    index.add(id, vec.data(), thread);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error adding vector " << task << ": " << e.what() << std::endl;
                }
            });

            // Save the index to disk
            std::string path_half = "usearch/indexes/" + dataset.name + "_index_half.usearch";
            std::filesystem::create_directories("usearch/indexes/");
            index.save(path_half.c_str());
            std::cout << "Index saved to: " << path_half << std::endl;


            // Populate index with second half

            std::cout << "Creating second half of index for dataset: " << dataset.name << "..." << std::endl;

            index.reserve(index_limits_t {NextPowerOfTwo(dataset_cardinality), executor.size()});

            // Clear vectors and ids for second half
            ids.clear();
            vectors.clear();
            ids.reserve(half_count);
            vectors.reserve(half_count);

            // Add rest of vectors to index
            for (idx_t i = 10; i < 20; i++) {
                auto& partition = partitions[i];
                for (idx_t j = 0; j < partition->RowCount(); j++) {
                    ids.push_back(partition->GetValue<int>(0, j));
                    vectors.push_back(ExtractFloatVector(partition->GetValue(1, j)));
                }
            }

            executor.fixed(half_count, [&](std::size_t thread, std::size_t task) {
                try {
                    int id = ids[task];
                    auto& vec = vectors[task];

                    index.add(id, vec.data(), thread);
                }
                catch (const std::exception& e) {
                    std::cerr << "Error adding vector " << task << ": " << e.what() << std::endl;
                }
            });

            // Save the index to disk
            std::string path = "usearch/indexes/" + dataset.name + "_index.usearch";
            std::filesystem::create_directories("usearch/indexes/");
            index.save(path.c_str());
            std::cout << "Index saved to: " << path << std::endl;

        } catch (std::exception& e) {
            std::cerr << "Error creating index: " << e.what() << std::endl;
        }
    }
};

// ==================== Main Function ====================
int main() {

    /**
     * Create and save USearch index for each dataset. The index will be reused
     * for all experiments.
     */

    try {
        // fashion_mnist
        USearchIndexCreator fm_creator(64);
        fm_creator.createIndexes(0);
        // mnist
        USearchIndexCreator m_creator(64);
        m_creator.createIndexes(1);
        // sift
        USearchIndexCreator s_creator(64);
        s_creator.createIndexes(2);
        // gist
        USearchIndexCreator g_creator(64);
        g_creator.createIndexes(3);
        std::cout << "All indexes created successfully." << std::endl;

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
