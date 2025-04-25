#pragma once

#include "duckdb.hpp"
#include <string>
#include <vector>

using namespace duckdb;

struct DatasetConfig {
    std::string name;
    int dimensions;
    int m; 
    int ef_construction;
};

class DatabaseSetup {
public:
    static std::vector<DatasetConfig> getDatasetConfigs();
    static void setupTrainTable(Connection& con, const std::string& table_name, int vector_dimensionality);
    static void setupTestTable(Connection& con, const std::string& table_name, int vector_dimensionality);
    static void setupGroundTruthTable(Connection& con, const std::string& table_name, int vector_dimensionality);
    static void setupFullDataset(Connection& con, const DatasetConfig& config);
    static void initializeResultsTable(Connection& con, const std::string& table_name);
    static void initializeBMTable(Connection& con, const std::string& table_name);
    static void intializeEarlyTermTable(Connection& con);
    static void exportResultsToCSV(Connection& con, const std::string& table_name);
    static void generateGroundTruthTable(Connection& con, const std::string& table_name, int vector_dimensionality);
};
