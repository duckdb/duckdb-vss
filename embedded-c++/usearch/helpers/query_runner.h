#pragma once

#include "duckdb.hpp"

#include <iostream>
#include <fstream>

using namespace duckdb;

class QueryRunner {
public:
    static void calculateRecall(Connection& con, const std::string& table_name);
    static void aggregateRecallStats(Connection& con, const std::string& table_name);
    static void aggregateBMStats(Connection& con, const std::string& table_name, size_t num_queries, size_t num_del_add);
    static void outputTableAsCSV(Connection& con, const std::string& full_table_name, const std::string& output_file_name);
    static unique_ptr<MaterializedQueryResult> getSampleVectors(Connection& con, const std::string& table_name, int rows = 1);
    static unique_ptr<MaterializedQueryResult> getSampleNonUnreachableVectors(Connection& con, const std::string& table_name, int rows = 5);
    static std::vector<unique_ptr<MaterializedQueryResult>> partitionDataset(Connection& con, const std::string& table_name, int num_partitions = 100);
};
