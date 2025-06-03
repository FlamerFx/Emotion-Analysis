#include "LabelUtils.h"
#include <fstream>
#include <algorithm>

// Load label strings from a file, one label per line
std::vector<std::string> load_labels(const std::string& filepath) {
    std::ifstream infile(filepath);
    std::vector<std::string> labels;
    std::string line;
    while (std::getline(infile, line)) {
        if (!line.empty())
            labels.push_back(line);
    }
    return labels;
}

// Return the index of the maximum value in a float array
size_t argmax(const float* data, size_t size) {
    return std::distance(data, std::max_element(data, data + size));
}