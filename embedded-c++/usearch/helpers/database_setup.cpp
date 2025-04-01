#include "database_setup.h"

std::vector<DatasetConfig> DatabaseSetup::getDatasetConfigs() {
    return {
        {"fashion_mnist", 784, 16, 100},
        {"mnist", 784, 16, 100},
        {"sift", 128, 16, 200},
        {"gist", 960, 32, 600}
    };
}

void DatabaseSetup::setupTrainTable(Connection& con, const std::string& table_name, int vector_dimensionality) {
    con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train" +
              " (id INTEGER, vec FLOAT[" + std::to_string(vector_dimensionality) + "])");

    con.Query("INSERT INTO memory." + table_name + "_train" +
              " SELECT * FROM raw." + table_name + "_train;");

    auto res = con.Query("SELECT * FROM memory." + table_name + "_train" + " limit 1;");
    if (res->RowCount() != 1) {
        throw std::runtime_error("Setup failed: Expected 1 row in train table");
    }
}

void DatabaseSetup::setupTestTable(Connection& con, const std::string& table_name, int vector_dimensionality) {
    con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test" +
              " (id INTEGER, vec FLOAT[" + std::to_string(vector_dimensionality) + "], neighbor_ids INTEGER[100])");

    con.Query("INSERT INTO memory." + table_name + "_test" +
              " SELECT * FROM raw." + table_name + "_test;");

    auto test_res = con.Query("SELECT * FROM memory." + table_name + "_test" + " LIMIT 1;");
    if (test_res->RowCount() != 1) {
        throw std::runtime_error("Setup failed: Expected 1 row in test table");
    }
}

void DatabaseSetup::setupFullDataset(Connection& con, const DatasetConfig& config) {
    con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    setupTrainTable(con, config.name, config.dimensions);
    setupTestTable(con, config.name, config.dimensions);

    con.Query("DETACH raw;");
}

void DatabaseSetup::initializeResultsTable(Connection& con, const std::string& table_name) {
    std::string recall_stats_query = "CREATE OR REPLACE TABLE memory.recall_stats (" +
              std::string("dataset VARCHAR, iteration INT, num_queries INT, ") +
              std::string("mean_recall FLOAT, median_recall FLOAT, stddev_recall FLOAT, ") +
              std::string("var_recall FLOAT, min_recall FLOAT, max_recall FLOAT, ") +
              std::string("p25_recall FLOAT, p75_recall FLOAT, p95_recall FLOAT, ") +
              std::string("mean_computed_distances FLOAT, median_computed_distances FLOAT, stddev_computed_distances FLOAT, ") +
              std::string("var_computed_distances FLOAT, min_computed_distances FLOAT, max_computed_distances FLOAT, ") +
              std::string("mean_visited_members FLOAT, median_visited_members FLOAT, stddev_visited_members FLOAT, ") +
              std::string("var_visited_members FLOAT, min_visited_members FLOAT, max_visited_members FLOAT, ") +
              std::string("mean_results_count FLOAT, median_results_count FLOAT, stddev_results_count FLOAT, ") +
              std::string("var_results_count FLOAT, min_results_count FLOAT, max_results_count FLOAT);");
    con.Query(recall_stats_query);

    std::string results_query = "CREATE OR REPLACE TABLE " + table_name + "_results (" +
                std::string("dataset VARCHAR, iteration INT, test_vec_id INT, ") +
                std::string("neighbor_vec_ids INTEGER[], result_vec_ids INTEGER[], recall FLOAT, ") +
                std::string("computed_distances INT, visited_members INT, results_count INT);");
    con.Query(results_query);
}

void DatabaseSetup::initializeBMTable(Connection& con, const std::string& table_name) {
    std::string bm_stats_query = "CREATE OR REPLACE TABLE memory." + table_name + "_bm_stats (" +
            std::string("dataset VARCHAR, iteration INT, num_queries INT, num_del_add INT, ") +
            std::string("mean_time FLOAT, median_time FLOAT, stddev_time FLOAT, ") +
            std::string("var_time FLOAT, min_time FLOAT, max_time FLOAT);");
    con.Query(bm_stats_query);

    std::string bm_query = "CREATE OR REPLACE TABLE " + table_name + "_bm (" +
              std::string("dataset VARCHAR, iteration INT, ") +
              std::string("operation_time FLOAT);");
    con.Query(bm_query);
}

void DatabaseSetup::intializeEarlyTermTable(Connection& con) {
    std::string results_query = "CREATE OR REPLACE TABLE early_terminated_queries (" +
            std::string("dataset VARCHAR, iteration INT, test_vec_id INT, ") +
            std::string("neighbor_vec_ids INTEGER[], result_vec_ids INTEGER[], recall FLOAT, ") +
            std::string("computed_distances INT, visited_members INT, results_count INT);");
    con.Query(results_query);
}

void DatabaseSetup::exportResultsToCSV(Connection& con, const std::string& table_name) {
    con.Query("COPY memory.recall_stats TO '" + table_name + "_output.csv' (HEADER, DELIMITER ',');");
    con.Query("COPY memory." + table_name + "_bm TO '" + table_name + "_bm.csv' (HEADER, DELIMITER ',');");
}
