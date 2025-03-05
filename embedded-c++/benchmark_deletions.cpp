#include "duckdb.hpp"
#include <iostream>
#include <benchmark/benchmark.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

using namespace duckdb;

DuckDB db(nullptr);
Connection con(db);

void SetupTrainTable(const std::string& table_name, int& vector_dimensionality) {
    con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_train" + " (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_train" + " SELECT * FROM raw." + table_name + "_train" + ";");
    auto res = con.Query("SELECT * FROM memory." + table_name + "_train" + " limit 1;");
    assert(res->RowCount() == 1);
    
    

}

// Load the data from the raw.db file and copy it to the memory database
void SetupTable(const std::string& table_name, int& vector_dimensionality) {
	con.Query("SET threads = 10;"); // My puter has 10 cores
    con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    SetupTrainTable(table_name, vector_dimensionality);

	con.Query("CREATE OR REPLACE TABLE memory." + table_name + "_test" + " (vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + "_test" + " SELECT * FROM raw." + table_name + "_test LIMIT 100;");
    auto test_res = con.Query("SELECT * FROM memory." + table_name + "_test" + " limit 1;");
    assert(test_res->RowCount() == 1);

    con.Query("DETACH raw;");
}

std::string GetTableName(int tableIndex) {
    switch (tableIndex) {
        case 0: return "fashion_mnist";
        case 1: return "mnist";
        case 2: return "sift";
        case 3: return "gist";
        default: return "unknown";
    }
}

int GetVectorDimensionality(int tableIndex) {
    switch (tableIndex) {
        case 0: return 784;
        case 1: return 784;
        case 2: return 128;
        case 3: return 960;
        default: return 0;
    }
}

std::vector<std::string> GetDeletedVectors(const std::string& table_name, int percentage) {
    std::cout << "Fetching vectors to delete..." << std::endl;
    auto result = con.Query("SELECT * FROM memory." + table_name + "_train USING SAMPLE " + std::to_string(percentage) + ";");
    std::vector<std::string> values;
    for (idx_t i = 0; i < result->RowCount(); i++) {
        values.push_back(result->GetValue(0, i).ToString());
    }
    std::cout << "Fetched " << values.size() << " vectors for deletion." << std::endl;
    return values;
}

//duckdb::unique_ptr<duckdb::MaterializedQueryResult> GetFirstRow(const std::string& table_name) {
//    auto result = con.Query("SELECT rowid, * FROM memory." + table_name + "_train LIMIT 1;");
//    std::cout << "Row id for the row" << result->GetValue(0, 0).ToString() << std::endl;
//    if (result->ColumnCount() == 0 || result->RowCount() == 0) {
//        std::cerr << "No data found in " << table_name << "_train" << std::endl;
//        return 0;
//    }
//    return result;
    //result->GetValue(0, 0).ToString();
//}

duckdb::unique_ptr<duckdb::MaterializedQueryResult> GetRandomRow(const std::string& table_name) {
    auto result = con.Query("SELECT rowId, * FROM memory." + table_name + "_test USING SAMPLE 1;");
   
    if(result->ColumnCount() == 0 || result->RowCount() == 0) {
        std::cerr << "No data found in " << table_name << "_test" << std::endl;
        return 0;
    }
    //auto query_vector = result->GetValue(0, 0).ToString();
    //auto query_index = result->GetValue(1, 0).ToString();
    return result;
}

void PerformSearch(const std::string& table_name, const std::string& query_vector, 
    const std::string& query_index, const std::string& vec_dim_string, 
    int iteration, bool before_deletion) {
    std::cout << "Performing search after deletion and compaction..." << std::endl;
    std::cout << vec_dim_string << std::endl;

    // Open the file in append mode.
    std::ofstream results_file("deletion_results.csv", std::ios::app);

    // If the file is empty, write the header.
    if (results_file.tellp() == 0) {
    results_file << "iteration,test_query_index,top_100_result_indexes,before_deletion\n";
    }

    // Perform your query.
    auto result = con.Query("SELECT rowId, * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " 
                + query_vector + "::FLOAT[" + vec_dim_string + "]) LIMIT 100;");

    std::cout << "Iterate through results for csv" << std::endl;
    benchmark::DoNotOptimize(result);
    benchmark::ClobberMemory();

    // Write current iteration results.
    results_file << iteration << "," << query_index << ",";

    for (idx_t i = 0; i < result->RowCount(); i++){
        auto result_vector_row = result->GetValue(0, i);
        int result_vector_index_int = result_vector_row.GetValue<int>();
        results_file << result_vector_index_int << ";";
    }
    results_file << "," << before_deletion;

    results_file << "\n";
    results_file.close();
    std::cout << "Search completed." << std::endl;
}


static void BM_VSSDeleteSearchCompactReadd(benchmark::State& state) {
    static bool is_setup = false;
    static std::string table_name;
    static std::string vec_dim_string;
 

    table_name = GetTableName(state.range(0));
    auto vector_dimensionality = GetVectorDimensionality(state.range(0));
   



    if (!is_setup) {
        SetupTable(table_name, vector_dimensionality);
        std::cout << "Creating index vss_hnsw_index..." << std::endl;
        con.Query("CREATE INDEX vss_hnsw_index ON memory." + table_name + "_train USING HNSW (vec);");
        std::cout << "Index created." << std::endl;
        is_setup = true;
    }

    vec_dim_string = std::to_string(vector_dimensionality);
    auto query_result = GetRandomRow(table_name);
    auto query_vector = query_result->GetValue(1, 0).ToString(); // the vector
    auto query_index = query_result->GetValue(0, 0).ToString(); // the id

    std::cout << "Query vector: " << query_vector << std::endl;
    std::cout << "Query index: " << query_index << std::endl;

    std::int64_t iteration = 0;  // Track current iteration

    for (auto _ : state) {

        std::cout << "Starting deletion iteration..." << std::endl;
        std::vector<std::string> deleted_vectors = GetDeletedVectors(table_name, 6000);

        std::cout << "Deleting selected vectors..." << std::endl;
        auto delete_query = "DELETE FROM memory." + table_name + "_train WHERE rowid IN (SELECT rowid FROM memory." +
                            table_name + "_train LIMIT 100);";
        con.Query(delete_query);
        std::cout << "Deletion completed." << std::endl;
        benchmark::ClobberMemory();

        std::cout << "Compacting index..." << std::endl;
        con.Query("PRAGMA hnsw_compact_index('vss_hnsw_index');");
        std::cout << "Index compaction completed." << std::endl;
        benchmark::ClobberMemory();

        PerformSearch(table_name, query_vector, query_index, vec_dim_string, iteration, false);

        if (!deleted_vectors.empty()) {
            std::cout << "Re-inserting deleted vectors using prepared statements..." << std::endl;
            auto prepared = con.Prepare("INSERT INTO memory." + table_name + "_train VALUES (?);");
            for (const auto &val : deleted_vectors) {
                try {
                    prepared->Execute(val);
                } catch (std::exception &e) {
                    std::cerr << "Error during re-insertion: " << e.what() << std::endl;
                }
            }
            std::cout << "Re-insertion completed." << std::endl;
            benchmark::ClobberMemory();
        }
        iteration++;
        std::cout << "Iteration completed.\\n" << std::endl;
    }
}

void RegisterBenchmarks() {
    int tableIndex = 1; // Set to 1 (mnist) for testing purposes
    int iteration = 0;

    benchmark::RegisterBenchmark("BM_VSSDeleteSearchCompactReadd", BM_VSSDeleteSearchCompactReadd)->Repetitions(100)
        ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
            return *(std::max_element(std::begin(v), std::end(v)));
        })
        ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
            return *(std::min_element(std::begin(v), std::end(v)));
        })
        ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
        ->Args({tableIndex});
}

int main(int argc, char** argv) {
    RegisterBenchmarks();
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
