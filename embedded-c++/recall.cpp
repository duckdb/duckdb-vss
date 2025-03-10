#include "duckdb.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>

using namespace duckdb;

// ==================== Dataset Configuration ====================
struct DatasetConfig {
    std::string name;
    int dimensions;
};

std::vector<DatasetConfig> getDatasetConfigs() {
    return {
        {"fashion_mnist", 784},
        {"mnist", 784},
        {"sift", 128},
        {"gist", 960}
    };
}

// ==================== Database Setup Functions ====================
class DatabaseSetup {
public:
    static void setupTrainTable(Connection& con, const std::string& table_name, int vector_dimensionality) {
        con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train" + 
                  " (id INTEGER, vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
        
        con.Query("INSERT INTO memory." + table_name + "_train" + 
                  " SELECT * FROM raw." + table_name + "_train;");
        
        auto res = con.Query("SELECT * FROM memory." + table_name + "_train" + " limit 1;");
        if (res->RowCount() != 1) {
            throw std::runtime_error("Setup failed: Expected 1 row in train table");
        }

        con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train_copy AS " +
                  "SELECT * FROM memory." + table_name + "_train;");
    }

    static void setupTestTable(Connection& con, const std::string& table_name, int vector_dimensionality) {
        con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test" + 
                  " (id INTEGER, vec FLOAT[" + std::to_string(vector_dimensionality) + "], neighbor_ids INTEGER[100])");
        
        con.Query("INSERT INTO memory." + table_name + "_test" + 
                  " SELECT * FROM raw." + table_name + "_test LIMIT 100;");
        
        auto test_res = con.Query("SELECT * FROM memory." + table_name + "_test" + " LIMIT 1;");
        if (test_res->RowCount() != 1) {
            throw std::runtime_error("Setup failed: Expected 1 row in test table");
        }
    }

    static void setupFullDataset(Connection& con, const DatasetConfig& config) {
        con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

        setupTrainTable(con, config.name, config.dimensions);
        setupTestTable(con, config.name, config.dimensions);

        con.Query("DETACH raw;");
    }

    static void initializeResultsTable(Connection& con, const std::string& table_name) {
        std::string recall_stats_query = "CREATE OR REPLACE TABLE memory.recall_stats (" +
                  std::string("dataset VARCHAR, iteration INT, num_queries INT, ") +
                  std::string("mean_recall FLOAT, median_recall FLOAT, stddev_recall FLOAT, ") +
                  std::string("var_recall FLOAT, min_recall FLOAT, max_recall FLOAT, ") +
                  std::string("p25_recall FLOAT, p75_recall FLOAT, p95_recall FLOAT);");
        con.Query(recall_stats_query);
        
        std::string results_query = "CREATE OR REPLACE TABLE " + table_name + "_results (" +
                  std::string("dataset VARCHAR, iteration INT, test_vec_id INT, ") +
                  std::string("neighbor_vec_ids INTEGER[100], result_vec_ids INTEGER[], recall FLOAT);");
        con.Query(results_query);
    }

    static void exportResultsToCSV(Connection& con, const std::string& table_name) {
        con.Query("COPY memory.recall_stats TO '" + table_name + "_output.csv' (HEADER, DELIMITER ',');");
    }
};

// ==================== HNSW Index Operations ====================
class IndexOperations {
public:
    static void createIndex(Connection& con, const std::string& table_name) {
        con.Query("CREATE INDEX hnsw_index ON " + table_name + "_train USING HNSW (vec);");
        auto indexes_count = con.Query("SELECT COUNT(index_name) FROM duckdb_indexes");
        
        // Safety check for index creation
        if (indexes_count->GetValue<int64_t>(0, 0) != 1) {
            throw std::runtime_error("Failed to create HNSW index");
        }
    }

    static void deleteSampleVectors(Connection& con, const std::string& table_name, 
                                    const std::string& sample_vectors_str) {
        std::cout << "🔵 DELETING SAMPLE VECTORS 🔵" << std::endl;
        con.Query("DELETE FROM " + table_name + "_train WHERE id IN " + sample_vectors_str + ";");
    }

    static void addSampleVectors(Connection& con, const std::string& table_name, 
                                   const std::string& sample_vectors_str) {
        std::cout << "🔵 ADDING SAMPLE VECTORS 🔵" << std::endl;
        con.Query("INSERT INTO " + table_name + "_train SELECT * FROM " + 
                  table_name + "_train_copy WHERE id IN " + sample_vectors_str + ";");
    }
    
    static std::string getSampleVectorsString(Connection& con, const std::string& table_name, double percentage = 1.0) {
        auto sample_vectors = con.Query(
            "SELECT array_agg(id) FROM " + table_name + "_train USING SAMPLE " + 
            std::to_string(percentage) + " PERCENT (reservoir);");
        
        std::string sample_vectors_str = sample_vectors->GetValue(0, 0).ToString();
        std::replace(sample_vectors_str.begin(), sample_vectors_str.end(), '[', '(');
        std::replace(sample_vectors_str.begin(), sample_vectors_str.end(), ']', ')');
        
        return sample_vectors_str;
    }
};

// ==================== Query Runner ====================
class QueryRunner {
public:
    static void runTestQueries(Connection& con, const std::string& table_name, int vector_dimensionality,
                               const unique_ptr<MaterializedQueryResult>& test_vectors, Appender& appender, int iteration) {
        std::cout << "🧪 RUNNING TEST QUERIES 🧪" << std::endl;
        std::string vec_dim_string = std::to_string(vector_dimensionality);
        
        // Process each test vector
        for (idx_t i = 0; i < test_vectors->RowCount(); i++) {
            // Extract query data from test vectors
            auto test_query_vector_string = test_vectors->GetValue(1, i).ToString();
            int test_query_vector_index_int = test_vectors->GetValue(0, i).GetValue<int>();
            Value neighbor_ids = test_vectors->GetValue(2, i);

            // Run KNN query
            auto result = con.Query(
                "SELECT array_agg(id), len(array_agg(id)) FROM (SELECT id, vec FROM " + 
                table_name + "_train ORDER BY array_distance(vec, " + 
                test_query_vector_string + "::FLOAT[" + vec_dim_string + "]) LIMIT 100);");

            // Check if we got enough results
            int results_returned = result->GetValue(1, 0).GetValue<int>();
            if (results_returned < 100) {
                std::cout << "‼️ ONLY " << results_returned << " RESULTS RETURNED ‼️" << std::endl;
                continue;
            }
            
            // Store query results
            appender.AppendRow(
                Value(table_name),                          // dataset name 
                Value::INTEGER(iteration),                  // iteration number
                Value::INTEGER(test_query_vector_index_int), // test vector ID
                neighbor_ids,                              // ground truth neighbors
                result->GetValue(0, 0),                    // result vector IDs
                Value::FLOAT(0.0)                          // recall (calculated later)
            );
        }
    }

    static void calculateRecall(Connection& con, const std::string& table_name) {
        std::cout << "🧮 CALCULATING RECALL 🧮" << std::endl;
        con.Query("UPDATE " + table_name + "_results " + 
                  "SET recall = len(list_intersect(neighbor_vec_ids, result_vec_ids)) / 100.0;");
    }

    static void aggregateRecallStats(Connection& con, const std::string& table_name) {
        std::cout << "🧮 AGGREGATING RECALL STATS 🧮" << std::endl;
        con.Query(
            "INSERT INTO recall_stats "
            "SELECT "
            "dataset, "
            "iteration, "
            "COUNT(*) AS num_queries, "
            "favg(recall) AS mean_recall, "
            "MEDIAN(recall) AS median_recall, "
            "STDDEV_POP(recall) AS stddev_recall, "
            "VAR_POP(recall) AS var_recall, "
            "MIN(recall) AS min_recall, "
            "MAX(recall) AS max_recall, "
            "APPROX_QUANTILE(recall, 0.25) AS p25_recall, "
            "APPROX_QUANTILE(recall, 0.75) AS p75_recall, "
            "APPROX_QUANTILE(recall, 0.95) AS p95_recall "
            "FROM " + table_name + "_results GROUP BY dataset, iteration ORDER BY iteration ASC;"
        );
    }
};

// ==================== File Operations ====================
class FileOperations {
public:
    static void initConnectivitySummaryFile() {
        std::ofstream csv_file("connectivity_summary.csv", std::ios::trunc);
        csv_file << "nodes_count,unreachable_count,orphaned_count" << std::endl;
        csv_file << "0,0,0" << std::endl;
        csv_file.close();
    }
    
    static void mergeCSVFiles(const std::string& table_name) {
        std::string output_file = table_name + "_output.csv";
        std::string connectivity_file = "connectivity_summary.csv";
        std::string merged_file = table_name + "_merged_report.csv";
        
        std::ifstream output_stream(output_file);
        std::ifstream connectivity_stream(connectivity_file);
        std::ofstream merged_stream(merged_file);
        
        if (!output_stream.is_open() || !connectivity_stream.is_open() || !merged_stream.is_open()) {
            std::cerr << "Error: Could not open files for merging" << std::endl;
            return;
        }
        
        // Read and combine headers
        std::string output_header, connectivity_header;
        std::getline(output_stream, output_header);
        std::getline(connectivity_stream, connectivity_header);
        
        merged_stream << output_header << "," << connectivity_header << std::endl;
        
        // Combine the data rows
        std::string output_line, connectivity_line;
        while (std::getline(output_stream, output_line) && std::getline(connectivity_stream, connectivity_line)) {
            merged_stream << output_line << "," << connectivity_line << std::endl;
        }
        
        std::cout << "✅ Created merged report: " << merged_file << " ✅" << std::endl;
        
        output_stream.close();
        connectivity_stream.close();
        merged_stream.close();
    }
};

// ==================== Main Test Runner ====================
class RecallTestRunner {
private:
    DuckDB db;
    Connection con;
    std::vector<DatasetConfig> datasets;
    int max_iterations;

public:
    RecallTestRunner(int iterations = 120) : db(nullptr), con(db), max_iterations(iterations) {
        con.Query("SET THREADS TO 1;");
        datasets = getDatasetConfigs();
        FileOperations::initConnectivitySummaryFile();
    }

    void runTest(int datasetIdx = 0) {
        try {
            // Limit to valid dataset indices
            if (datasetIdx < 0 || datasetIdx >= (int)datasets.size()) {
                std::cerr << "Invalid dataset index: " << datasetIdx << std::endl;
                return;
            }

            const auto& dataset = datasets[datasetIdx];
            std::cout << "📊 TESTING DATASET: " << dataset.name << " 📊" << std::endl;
            
            // Setup database tables
            DatabaseSetup::initializeResultsTable(con, dataset.name);
            DatabaseSetup::setupFullDataset(con, dataset);
            
            // Create initial index
            IndexOperations::createIndex(con, dataset.name);
            
            // Get test vectors
            auto test_vectors = con.Query("SELECT * FROM " + dataset.name + "_test;");
            
            // Create appender for results
            Appender appender(con, dataset.name + "_results");
            
            // Initial query run
            QueryRunner::runTestQueries(con, dataset.name, dataset.dimensions, test_vectors, appender, 0);
            
            // Run iterations
            for (int iteration = 1; iteration <= max_iterations; iteration++) {
                std::cout << "▶️ ITERATION " << iteration << " ▶️" << std::endl;
                
                // Get sample vectors to delete and re-add
                std::string sample_vectors_str = IndexOperations::getSampleVectorsString(con, dataset.name);
                
                // Delete sample vectors
                IndexOperations::deleteSampleVectors(con, dataset.name, sample_vectors_str);
                
                // Re-add sample vectors
                IndexOperations::addSampleVectors(con, dataset.name, sample_vectors_str);
                
                // Run test queries
                QueryRunner::runTestQueries(con, dataset.name, dataset.dimensions, test_vectors, appender, iteration);
                
                std::cout << "✅ FINISHED ITERATION " << iteration << " ✅" << std::endl;
            }
            
            appender.Close();
            
            // Calculate recall and aggregate stats
            QueryRunner::calculateRecall(con, dataset.name);
            QueryRunner::aggregateRecallStats(con, dataset.name);
            
            // Export results
            DatabaseSetup::exportResultsToCSV(con, dataset.name);
            
            // Merge CSV files
            FileOperations::mergeCSVFiles(dataset.name);
            
        } catch (std::exception& e) {
            std::cerr << "Error running test: " << e.what() << std::endl;
        }
    }
};

// ==================== Main Function ====================
int main() {
    try {
        // TODO: reset state for each dataset instead of creating new runner...
        RecallTestRunner fm_runner(119);  // Run for 119 iterations
        RecallTestRunner m_runner(119);  // Run for 119 iterations
        
        // Run test on fashion_mnist
        fm_runner.runTest(0);
        //Run test on mnist
        m_runner.runTest(1);
        
        return 0;
    } catch (std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}
