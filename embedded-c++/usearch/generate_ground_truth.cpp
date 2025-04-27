#include "duckdb.hpp"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/index_operations.h"
#include "usearch/helpers/query_runner.h"
#include <iostream>

using namespace duckdb;

// ==================== Ground Truth Generator ====================
class GroundTruthGenerator {
private:
	DuckDB db;
	Connection con;
	std::vector<DatasetConfig> datasets;
	int numThreads;

    // Structure to hold vector data
    struct VectorData {
        int id;
        std::vector<float> vec;
    };

    // Structure to hold distance result
    struct DistanceResult {
        int query_id;
        int neighbor_id;
        float distance;
    };

    // Compute L2 (Euclidean) distance between two vectors
    float computeDistance(const std::vector<float>& vec1, const std::vector<float>& vec2) {
        float sum = 0.0f;
        for (size_t i = 0; i < vec1.size(); i++) {
            float diff = vec1[i] - vec2[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }

    // Worker function for multi-threaded processing
    void processTestVectorBatch(
        const std::vector<VectorData>& testVectors,
        const std::vector<VectorData>& trainVectors,
        size_t start_idx,
        size_t end_idx,
        std::vector<DistanceResult>& results,
        std::mutex& resultsMutex
    ) {
        std::vector<DistanceResult> localResults;
        
        // For each test vector in this batch
        for (size_t i = start_idx; i < end_idx && i < testVectors.size(); i++) {
            const auto& testVec = testVectors[i];
            
            // Calculate distance to all training vectors
            for (const auto& trainVec : trainVectors) {
                DistanceResult result;
                result.query_id = testVec.id;
                result.neighbor_id = trainVec.id;
                result.distance = computeDistance(testVec.vec, trainVec.vec);
                localResults.push_back(result);
            }
            
            // // Progress indicator every 10 test vectors
            // if ((i - start_idx) % 10 == 0) {
            //     std::cout << "Thread processed " << (i - start_idx) << " test vectors" << std::endl;
            // }
        }
        
        // Add local results to global results vector
        std::lock_guard<std::mutex> lock(resultsMutex);
        results.insert(results.end(), localResults.begin(), localResults.end());
    }

public:
	GroundTruthGenerator(int numThreads) : db(nullptr), con(db), numThreads(numThreads) {
		con.Query("SET THREADS TO " + std::to_string(numThreads) + ";");
		datasets = DatabaseSetup::getDatasetConfigs();
	}

	void runGenerator(int datasetIdx) {
		try {

            auto startTime = std::chrono::high_resolution_clock::now();

            con.Query("ATTACH 'raw.db';");

			// Limit to valid dataset indices
			if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
				std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
				return;
			}

			const auto &dataset = datasets[datasetIdx];

            std::cout << "Generating ground truth table for dataset: " << dataset.name << std::endl;

            auto test_res = con.Query("select * from raw." + dataset.name + "_test");
            auto train_res = con.Query("select * from raw." + dataset.name + "_train");

            std::vector<VectorData> testVectors;
            testVectors.reserve(test_res->RowCount());

            std::vector<VectorData> trainVectors;
            trainVectors.reserve(train_res->RowCount());

            for (idx_t i = 0; i < test_res->RowCount(); i++) {
                testVectors.push_back(VectorData{test_res->GetValue<int>(0, i), ExtractFloatVector(test_res->GetValue(1, i))});
            }
            for (idx_t i = 0; i < train_res->RowCount(); i++) {
                trainVectors.push_back(VectorData{train_res->GetValue<int>(0, i), ExtractFloatVector(train_res->GetValue(1, i))});
            }

            std::cout << "Loaded " << testVectors.size() << " test vectors and " 
                    << trainVectors.size() << " train vectors" << std::endl;
            
            std::cout << "Starting brute force KNN calculation using " << numThreads << " threads" << std::endl;
            
            // Calculate total expected results for preallocating memory
            size_t expectedResults = testVectors.size() * trainVectors.size();
            std::cout << "Expected result count: " << expectedResults << std::endl;
            
            // Vector to hold all distance results
            std::vector<DistanceResult> results;
            results.reserve(expectedResults);
            
            // Mutex for thread synchronization
            std::mutex resultsMutex;
            
            // Calculate batch size for each thread
            size_t batchSize = (testVectors.size() + numThreads - 1) / numThreads;
            
            // Create and start threads
            std::vector<std::thread> threads;
            for (unsigned int t = 0; t < numThreads; t++) {
                size_t start_idx = t * batchSize;
                size_t end_idx = (t + 1) * batchSize;
                
                threads.push_back(std::thread(
                    [this](const std::vector<VectorData>& testVectors, 
                        const std::vector<VectorData>& trainVectors,
                        size_t start_idx, size_t end_idx,
                        std::vector<DistanceResult>& results,
                        std::mutex& resultsMutex) {
                        this->processTestVectorBatch(testVectors, trainVectors, start_idx, end_idx, results, resultsMutex);
                    },
                    std::ref(testVectors),
                    std::ref(trainVectors),
                    start_idx,
                    end_idx,
                    std::ref(results),
                    std::ref(resultsMutex)
                ));
            }
            
            // Wait for all threads to complete
            for (auto& thread : threads) {
                thread.join();
            }
            
            std::cout << "All threads completed. Uploading results to database..." << std::endl;

            con.Query("CREATE OR REPLACE TABLE " + dataset.name + "_ground_truth (" +
                "query_id INTEGER, " +
                "neighbor_id INTEGER, distance FLOAT)");
            
            Appender appender(con, dataset.name + "_ground_truth");


            // Cache test vectors in a map for quick lookup
            std::unordered_map<int, std::vector<float>> testVectorMap;
            for (const auto& vec : testVectors) {
                testVectorMap[vec.id] = vec.vec;
            }

            // Append all results
            for (const auto& result : results) {
                appender.AppendRow(result.query_id, 
                                result.neighbor_id, 
                                result.distance);
            }

            // Complete the operation
            appender.Close();

            // Get train dataset index ids
            std::unordered_set<size_t> train_ids;
            for (const auto& vec : trainVectors) {
                train_ids.insert(vec.id);
            }

            auto gt_test_vectors = QueryRunner::getCurrentTopKNeighbors(con, dataset.name, train_ids);

            // gt_test_vectors should be the exact same as test_res
            for (idx_t i = 0; i < gt_test_vectors->RowCount(); i++) {
                auto gt_test_vector = gt_test_vectors->GetValue(0, i);
                auto test_vector = test_res->GetValue(0, i);
                if (gt_test_vector != test_vector) {
                    throw std::runtime_error("Setup failed: Inconsistent ground truth table ids");
                }
                auto gt_neighbor_ids = ExtractSizeVector(gt_test_vectors->GetValue(2, i));
                auto test_neighbor_ids = ExtractSizeVector(test_res->GetValue(2, i));
                if (gt_neighbor_ids != test_neighbor_ids) {
                    throw std::runtime_error("Setup failed: Inconsistent ground truth table neighbor ids");
                }
            }

            // Create raw table
            con.Query("CREATE OR REPLACE TABLE raw." + dataset.name + "_ground_truth as select * from " + dataset.name + "_ground_truth");

            auto ground_truth_count = con.Query("SELECT COUNT(*) FROM raw." + dataset.name + "_ground_truth");
            std::cout << "Ground truth vectors: " << ground_truth_count->GetValue<int64_t>(0, 0) << std::endl;
            auto expected_rows = train_res->RowCount() * test_res->RowCount();
            std::cout << "Expected rows: " << expected_rows << std::endl;
            // Verify data was inserted correctly
            if (ground_truth_count->GetValue<int64_t>(0, 0) != expected_rows) {
                throw std::runtime_error("Setup failed: Incorrect number of rows inserted in ground truth table");
            } else {
                std::cout << "Successfully inserted " << ground_truth_count 
                        << " rows into ground truth table" << std::endl;
            }
            
            std::cout << "Results uploaded to database successfully" << std::endl;
            
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
            std::cout << "Ground truth table generated for dataset: " << dataset.name << " in " << duration << " seconds" << std::endl;

		} catch (std::exception &e) {
			std::cerr << "Error generating ground truth table: " << e.what() << std::endl;
		}
	}
};

// ==================== Main Function ====================
int main() {

	/**
	 * GROUNDTRUTHGENERATOR:
	 * calculate ground truth table for all datasets. This is used in new data
	 * experiments due to the index at each iteration not being comprised of the whole
	 * dataset.
	 */

	std::size_t executor_threads = (std::thread::hardware_concurrency());

	try {
		// fashion_mnist
		GroundTruthGenerator fm_runner(executor_threads);
		fm_runner.runGenerator(0);

		// mnist
		GroundTruthGenerator m_runner(executor_threads);
		m_runner.runGenerator(1);

		// sift
		GroundTruthGenerator s_runner(executor_threads);
		s_runner.runGenerator(2);

		// gist
		GroundTruthGenerator g_runner(executor_threads);
		g_runner.runGenerator(3);

		return 0;
	} catch (std::exception &e) {
		std::cerr << "Fatal error: " << e.what() << std::endl;
		return 1;
	}
}
