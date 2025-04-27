#include "duckdb.hpp"
#include "usearch/helpers/database_setup.h"
#include "usearch/helpers/index_operations.h"

#include <iostream>

using namespace duckdb;

// ==================== Ground Truth Generator ====================
class GroundTruthGenerator {
private:
	DuckDB db;
	Connection con;
	std::vector<DatasetConfig> datasets;
	int threads;

public:
	GroundTruthGenerator(int threads) : db(nullptr), con(db) {
		con.Query("SET THREADS TO " + std::to_string(threads) + ";");
		datasets = DatabaseSetup::getDatasetConfigs();
	}

	void runGenerator(int datasetIdx) {
		try {
			// Limit to valid dataset indices
			if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
				std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
				return;
			}

			const auto &dataset = datasets[datasetIdx];

			std::cout << "Generating ground truth table for dataset: " << dataset.name << std::endl;

			// Generate ground truth table
			DatabaseSetup::generateGroundTruthTable(con, dataset.name, dataset.dimensions);

			std::cout << "Ground truth table generated for dataset: " << dataset.name << std::endl;

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
