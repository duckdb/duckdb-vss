#include "file_operations.h"

void FileOperations::cleanupOutputFiles(std::filesystem::path path = std::filesystem::current_path()) {
    try {
        std::string file_extension = ".csv";
        std::string file_extension_txt = ".txt";

        for (const auto& entry : std::filesystem::directory_iterator(path)) {
            std::string file_path_string = entry.path().string();

            // Only remove if file ends with ".csv"
            if (file_path_string.size() >= 4 &&
                (file_path_string.substr(file_path_string.size() - 4) == file_extension) || 
                ((file_path_string.substr(file_path_string.size() - 4) == file_extension_txt && file_path_string.substr(file_path_string.size() - 14) != "CMakeLists.txt"))) {
                std::filesystem::remove(entry.path());
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Filesystem error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error cleaning up files: " << e.what() << std::endl;
    }
}

void FileOperations::copyFileTo(const std::string& source, const std::string& destination) {
    std::filesystem::copy_file(source, destination);
}
