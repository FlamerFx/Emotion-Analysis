#pragma once
#include <vector>
#include <string>

// Load labels (class names) from a file, one label per line
std::vector<std::string> load_labels(const std::string& filepath);

// Return the index of the maximum value in a float array
size_t argmax(const float* data, size_t size);