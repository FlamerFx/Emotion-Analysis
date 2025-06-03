#pragma once

#include <string>
#include <unordered_map>
#include <vector>

// Handles text preprocessing: tokenization, cleaning, mapping to indices, and padding
class TextPreprocessor {
public:
    // Initialize with vocabulary file and max sequence length
    explicit TextPreprocessor(const std::string& vocab_file, int max_len = 100);

    // Process a string and return a padded sequence of floats (for model input)
    std::vector<float> preprocess(const std::string& text) const;

private:
    std::unordered_map<std::string, int> word_index_;
    int max_len_;

    // Clean a word: keep only alphanumeric, convert to lowercase
    static std::string clean_word(const std::string& word);

    // Tokenize text into cleaned words
    static std::vector<std::string> tokenize(const std::string& text);
};