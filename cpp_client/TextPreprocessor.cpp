#include "TextPreprocessor.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>

// Construct the preprocessor with vocabulary file and max sequence length
TextPreprocessor::TextPreprocessor(const std::string& vocab_file, int max_len)
    : max_len_(max_len) {
    std::ifstream infile(vocab_file);
    std::string word;
    int idx;
    while (infile >> word >> idx) {
        word_index_[word] = idx;
    }
}

// Preprocess input text: tokenize, map to indices, pad/truncate, convert to float vector
std::vector<float> TextPreprocessor::preprocess(const std::string& text) const {
    std::vector<std::string> words = tokenize(text);

    std::vector<int> indices;
    for (auto& w : words) {
        auto it = word_index_.find(w);
        indices.push_back(it != word_index_.end() ? it->second : 0); // 0 for OOV
    }

    if ((int)indices.size() < max_len_) {
        indices.resize(max_len_, 0); // pad with 0
    } else if ((int)indices.size() > max_len_) {
        indices.resize(max_len_);
    }

    std::vector<float> out(indices.begin(), indices.end());
    return out;
}

// Clean a word: keep only alphanumeric, convert to lowercase
std::string TextPreprocessor::clean_word(const std::string& word) {
    std::string cleaned;
    for (char c : word) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cleaned += std::tolower(static_cast<unsigned char>(c));
        }
    }
    return cleaned;
}

// Tokenize text into cleaned words
std::vector<std::string> TextPreprocessor::tokenize(const std::string& text) {
    std::istringstream iss(text);
    std::string token;
    std::vector<std::string> tokens;
    while (iss >> token) {
        std::string cleaned = clean_word(token);
        if (!cleaned.empty()) tokens.push_back(cleaned);
    }
    return tokens;
}