#pragma once

#include "duckdb.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace duckdb;

class FileOperations {
public:
    static void cleanupOutputFiles(std::filesystem::path path);
    static void copyFileTo(const std::string& source, const std::string& destination);
};
