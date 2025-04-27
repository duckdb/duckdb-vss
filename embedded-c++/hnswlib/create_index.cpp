#include <hnswlib/hnswlib.h>
#include "duckdb.hpp"
#include "hnswlib/helpers/hnswlib_index_operations.h"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/query_runner.h"
#include "usearch/helpers/file_operations.h"

using namespace duckdb;
using namespace hnswlib;

// ==================== HNSWLib Index Creator ====================
class HNSWLibIndexCreator {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int threads;

public:
HNSWLibIndexCreator(int threads = 64) : db(nullptr), con(db) {
        con.Query("SET THREADS TO " + std::to_string(threads) + ";");
        datasets = DatabaseSetup::getDatasetConfigs();
    }

    void createIndexes(int datasetIdx = 0) {
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

            // Create the hnswlib index
            auto dataset_cardinality = con.Query("SELECT COUNT(*) FROM " + dataset.name + "_train;")->GetValue<int64_t>(0, 0);
            L2Space space(dataset.dimensions);
            HierarchicalNSW<float> index(&space, dataset_cardinality, dataset.m, dataset.ef_construction, 100, true);

            // Partition dataset into 20
            auto partitions = QueryRunner::partitionDataset(con, dataset.name, 20);

            auto half_count = dataset_cardinality / 2;

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

            for (std::size_t task = 0; task < dataset_cardinality; ++task) {             
                try {
                    int id = ids[task];
                    auto& vec = vectors[task];
                    if (task < half_count){
                        index.addPoint(vec.data(), id);
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error adding vector " << task << ": " << e.what() << std::endl;
                }
            }
            
            std::unordered_map<size_t, size_t> index_map;
            index_map.reserve(dataset_cardinality);
            for (size_t i = 0; i < ids.size(); ++i) {
                index_map[i] = ids[i];
            }

            // Save the index to disk
            std::string path_half = "hnswlib/indexes/" + dataset.name + "_index_half.bin";
            std::filesystem::create_directories("hnswlib/indexes/");
            index.saveIndex(path_half);
            std::cout << "Index saved to: " << path_half << std::endl;

            // Save index_map to disk
            std::string index_map_path_half = "hnswlib/indexes/" + dataset.name + "_index_map_half.txt";
            std::filesystem::create_directories("hnswlib/indexes/");
            std::ofstream index_map_file_half(index_map_path_half);
            for (const auto& [key, value] : index_map) {
                index_map_file_half << key << " " << value << std::endl; 
            }
            index_map_file_half.close();
            std::cout << "Index map saved to: " << index_map_path_half << std::endl;


            // Populate index with second half

            std::cout << "Creating second half of index for dataset: " << dataset.name << "..." << std::endl;

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

            for (std::size_t task = 0; task < dataset_cardinality; ++task) {             
                try {
                    int id = ids[task];
                    auto& vec = vectors[task];
                    if (task < half_count){
                        index.addPoint(vec.data(), id);
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error adding vector " << task << ": " << e.what() << std::endl;
                }
            }

            // Add to index_map
            size_t offset = index_map.size();
            for (size_t i = 0; i < ids.size(); ++i) {
                index_map[offset + i] = ids[i];
            }
            assert(index_map.size() == dataset_cardinality);

            // Save the index to disk
            std::string path = "hnswlib/indexes/" + dataset.name + "_index.bin";
            std::filesystem::create_directories("hnswlib/indexes/");
            index.saveIndex(path);
            std::cout << "Index saved to: " << path << std::endl;

            // Save index_map to disk
            std::string index_map_path = "hnswlib/indexes/" + dataset.name + "_index_map.txt";
            std::filesystem::create_directories("hnswlib/indexes/");
            std::ofstream index_map_file(index_map_path);
            for (const auto& [key, value] : index_map) {
                index_map_file << key << " " << value << std::endl; 
            }
            index_map_file.close();
            std::cout << "Index map saved to: " << index_map_path << std::endl;

        } catch (std::exception& e) {
            std::cerr << "Error creating index: " << e.what() << std::endl;
        }
    }
};

// ==================== Main Function ====================
int main() {

    /**
     * Create and save hnswlib index for each dataset. The index will be reused
     * for all experiments.
     */

    try {
        // fashion_mnist
        HNSWLibIndexCreator fm_creator(64);
        fm_creator.createIndexes(0);
        // mnist
        HNSWLibIndexCreator m_creator(64);
        m_creator.createIndexes(1);
        // sift
        HNSWLibIndexCreator s_creator(64);
        s_creator.createIndexes(2);
        // gist
        HNSWLibIndexCreator g_creator(64);
        g_creator.createIndexes(3);
        std::cout << "All indexes created successfully." << std::endl;

        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
