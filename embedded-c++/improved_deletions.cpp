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


struct VectorData {
    int id;                     // Store the ID
    string vec;      // Store the vector
};

std::vector<VectorData> vectors_to_delete; 

std::vector<VectorData> search_vectors;


/* This function takes in a table name and vector dimensionality
 *  and creates a table in the memory database with the same schema and data
 *  as the table in the raw database. Here table_name includes if its 
 *  the test version or train version.
 */
void SetupTableInternal(const std::string& table_name, int& vector_dimensionality) {
    std::cout << "Setting up table " << table_name << std::endl;
    con.Query("CREATE OR REPLACE TABLE memory." + table_name + " (id INTEGER, vec FLOAT[" + std::to_string(vector_dimensionality) + "])");
    con.Query("INSERT INTO memory." + table_name + " SELECT * FROM raw." + table_name + ";");
    auto res = con.Query("SELECT * FROM memory." + table_name + " limit 1;");
    assert(res->RowCount() == 1);
    std::cout << "Table " << table_name << " created" << std::endl;
}

/* This function wrapps around the SetupTableInternal function. Its function 
 * is to attach the persistent database to memory and to call the SetupTableInternal
 * function to populate the memory database with the data from the persistent database.
 */
 void SetupTable(const std::string& table_name, int& vector_dimensionality) {
	con.Query("SET threads = 1;"); 
    con.Query("ATTACH 'raw.db' AS raw (READ_ONLY);");

    SetupTableInternal(table_name + "_test", vector_dimensionality);
    SetupTableInternal(table_name + "_train", vector_dimensionality);

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


/*
* This function should be replaced by using an appender to append the results of the search test to results.db file.
* Meanwhile, the function writes the results to a csv file.
*/
void WriteToFile(duckdb::unique_ptr<duckdb::MaterializedQueryResult>& result, int iteration, int query_index, int before_deletion) {

     std::ofstream results_file("deletion_results.csv", std::ios::app);

     if (results_file.tellp() == 0) {
        results_file << "iteration,test_query_index,top_100_result_indexes,before_deletion\n";
    }

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


/* This function takes in a table name and a sample percentage
 *  and returns a vector of VectorData structs that contain the ID and vector
 *  of the vectors to be deleted.
 */
std::vector<VectorData> SelectVectorsToDelete(const std::string& table_name, int sample_percentage) {
    std::cout << "Fetching vectors to delete in all iterations" << std::endl;
    
    auto result = con.Query("SELECT id, vec FROM memory." + table_name + "_train USING SAMPLE " + std::to_string(sample_percentage) + "%;");
    std::vector<VectorData> values;

    for (idx_t i = 0; i < result->RowCount(); i++) {
        VectorData data;
        data.id = result->GetValue(0, i).GetValue<int>();  // Extract ID
        data.vec = result->GetValue(1, i).ToString();; // Extract Vector as float array
        
        values.push_back(std::move(data));
    }

    std::cout << "Fetched " << values.size() << " vectors for deletion." << std::endl;
    return values;
}


/* This function takes in a table name and a sample size
 *  and returns a vector of VectorData structs that contain the ID and vector
 *  of the vectors to be used in the search query.
 */
std::vector<VectorData> GetRandomSearchVectors(const std::string& table_name, int sample_size) {
    std::cout << "Fetching random search vectors in all iterations" << std::endl;
    auto result = con.Query("SELECT * FROM memory." + table_name + "_test USING SAMPLE " + std::to_string(sample_size) + ";");
    
    std::vector<VectorData> values;

    for (idx_t i = 0; i < result->RowCount(); i++) {
        VectorData data;
        data.id = result->GetValue(0, i).GetValue<int>();  // Extract ID
        data.vec = result->GetValue(1, i).ToString(); // Extract Vector as float array

        //std::cout << "ID: " << data.id << std::endl;
        //std::cout << "Vector: " << data.vec << std::endl;
        //std::cout << "" << std::endl;
        
        values.push_back(std::move(data));
    }
    std::cout << "Fetched " << values.size() << " search vectors." << std::endl;
    return values;
}



void Delete(benchmark::State& state){
    
    
    string table_name = GetTableName(state.range(0));

    int repetition = state.range(1);

    std::vector<VectorData> vectors_to_delete; 

    // Begin transaction to be able to ROLLBACK or COMMIT depending on what repetition we are on.
    con.Query("BEGIN TRANSACTION;");

   // Prepare the deletion statement
    auto prepared_delete = con.Prepare("DELETE FROM memory." + table_name + "_train WHERE id = ?;");

    for (auto _: state) {
        for (const auto &vec : vectors_to_delete) {
            try {
                prepared_delete->Execute(vec.id); // Delete using id
            } catch (const std::exception &e) {
                std::cerr << "Error deleting vector with ID " << vec.id << ": " << e.what() << std::endl;
            }
        }

        benchmark::ClobberMemory();

        // Compact the index after deletion
        con.Query("PRAGMA hnsw_compact_index('vss_hnsw_index');");
        benchmark::ClobberMemory();
    }
    

    // ROLLBACK the deletion if state.iteration is not last so that the repetition does not remove the whole dataset
  
    if (repetition == 100){
        // Perform search
        con.Query("COMMIT;");
    }
    
    else {
        // Perform search
        con.Query("ROLLBACK;");
    }

    repetition++;
}


void Reinsert(benchmark::State& state){

    string table_name = std::to_string(state.range(0));
    
    int repetition = state.range(1);

    // Begin transaction to be able to ROLLBACK or COMMIT depending on what repetition we are on.
    con.Query("BEGIN TRANSACTION;");

    for (auto _: state){
        // Delete vectors
        auto prepared = con.Prepare("INSERT INTO memory." + table_name + "_train VALUES (?, ?);");
        for (const auto &val : vectors_to_delete) {
            try {
                std::cout << "Inserting ID: " << val.id << " Vector: " << val.vec << std::endl;
                prepared->Execute(val.id, val.vec);  // Pass fields separately
            } catch (std::exception &e) {
                std::cerr << "Error during re-insertion: " << e.what() << std::endl;
            }
        }
        benchmark::ClobberMemory();
        
    }


    // ROLLBACK the deletion if state.repetiton is not last so that the repetition does not remove the whole dataset

    if (repetition == 100){
        con.Query("COMMIT;");
    }
    
    else {
        con.Query("ROLLBACK;");
    }
    repetition++;
}


void Search(benchmark::State& state) {

    int tableIndex = state.range(0);
    auto table_name = GetTableName(state.range(0));

    int iteration = state.range(1);



    int vector_dimensionality = GetVectorDimensionality(tableIndex);


    string query_vector = search_vectors[state.range(1)].vec;

    duckdb::unique_ptr<duckdb::MaterializedQueryResult> result;

    for (auto _: state){
        // Delete vectors
        result = con.Query("SELECT * FROM memory." + table_name + "_train ORDER BY array_distance(vec, " 
            + query_vector + "::FLOAT[" + std::to_string(vector_dimensionality)  + "]) LIMIT 100;");
    }

    // Write the results to a csv file 
    // This should be changed by appending or directyl inserting the results to the results.db file
    WriteToFile(result, iteration, search_vectors[state.range(1)].id, vectors_to_delete.size());
    

  
    benchmark::ClobberMemory();
}



int BM_VSSDeletionAndReinsertion(int cycles) {

    //int selected_tables[] = {0, 1, 2, 3};
    //for (int i = 0; i < sizeof(selected_tables) / sizeof(selected_tables[0]); i++) {
        int table_index = 1; // Only select MNSIT for testing
        auto vector_dimensionality = GetVectorDimensionality(table_index);
        auto table_name = GetTableName(table_index);
        SetupTable(table_name, vector_dimensionality);
        con.Query("CREATE INDEX vss_hnsw_index ON memory." + table_name + "_train USING HNSW (vec);");

        std::cout << "Table and index for: " << table_name << " created" << std::endl;

        vectors_to_delete = SelectVectorsToDelete(table_name, 10);
        search_vectors = GetRandomSearchVectors(table_name, 100);

        
     
    

        // Loop here for amount of cycles to be run
        for (int i = 0; i < cycles; i++){
            // Delete vectors

            std::cout << "Starting cycle" << i << std::endl;
            std::cout << "Vectors in database is " << con.Query("SELECT COUNT(*) FROM memory." + table_name + "_train;")->GetValue<int>(0, 0) << std::endl;

            int deletion_repetition = 0;
            string benchmark_name_deletion = "BM_Deletion/"+table_name+ "/" + to_string(i);
            benchmark::RegisterBenchmark(benchmark_name_deletion, Delete)->Repetitions(5)
            ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
            })
            ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                return *(std::min_element(std::begin(v), std::end(v)));
            })
            ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
            ->Args({table_index, deletion_repetition});



            // Reinsert vectors
            int reinsertion_repetition = 0;
            string benchmark_name_reinsertion = "BM_Reinsertion/"+table_name+ "/" + to_string(i);

            benchmark::RegisterBenchmark(benchmark_name_reinsertion, Reinsert)->Repetitions(5)
            ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
            })
            ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                return *(std::min_element(std::begin(v), std::end(v)));
            })
            ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
            ->Args({table_index, reinsertion_repetition});

            // Perform search
            string benchmark_name_search = "BM_Search/"+table_name+ "/" + to_string(i);
            benchmark::RegisterBenchmark(benchmark_name_search, Search)->Repetitions(5)
            ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
                return *(std::max_element(std::begin(v), std::end(v)));
            })
            ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
                return *(std::min_element(std::begin(v), std::end(v)));
            })
            ->DisplayAggregatesOnly(true)->ReportAggregatesOnly(true)
            ->Args({table_index, reinsertion_repetition});

        }
        return 0;

}

int main(int argc, char** argv) {
    // Run the benchmark

    auto cycles = 2;
    BM_VSSDeletionAndReinsertion(cycles);
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}