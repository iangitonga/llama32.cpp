#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <iostream>
#include <regex>


/// @brief A tokenizer that performs words encoding and token id decoding as-per LLama 3.2 vocabulary.
class Llama32Tokenizer {
public:
    const int eot_id = 128009;

public:

Llama32Tokenizer(const std::string& vocab_path, int n_vocab)
    : m_n_vocab{n_vocab}
{
    std::ifstream fin{vocab_path, std::ios_base::binary};
    if(!fin.is_open()) {
        std::fprintf(stderr, "Failed to open vocab file: %s\n", vocab_path.c_str());
        std::exit(EXIT_FAILURE);
    };

    std::string word;
    for (int i = 0; i < n_vocab; i++)
    {
        int32_t len;
        fin.read((char *) &len, sizeof(len));

        word.resize(len);
        fin.read((char *) word.data(), len);

        token_to_id_[word] = i;
        id_to_token_[i] = word;
    }
}

// Convert a single token id into text.
const char* decode(int32_t token_id) {
    return id_to_token_[token_id].c_str();
}

// Convert a string of arbitrary text to a sequence of tokens ids.
std::vector<int32_t> encode(const std::string& text) {
    std::vector<std::string> words;

    // first split the text into words
    std::string str = text;
    std::regex re(m_pat);
    std::smatch m;

    while (std::regex_search(str, m, re)) {
        for (auto x : m) {
            words.push_back(x);
        }
        str = m.suffix();
    }

    // find the longest tokens that form the words:
    std::vector<int32_t> tokens;
    tokens.reserve(encode_prefix.size() + words.size() + encode_suffix.size());
    // prepend prefix.
    tokens.insert(tokens.end(), encode_prefix.begin(), encode_prefix.end());

    for (const auto & word : words)
    {
        if (word.size() == 0) continue;

        int i = 0;
        int n = word.size();
        while (i < n) {
            int j = n;
            while (j > i)
            {
                auto it = token_to_id_.find(word.substr(i, j-i));
                if (it != token_to_id_.end()) {
                    tokens.push_back(it->second);
                    i = j;
                    break;
                }
                --j;
            }
            if (i == n)
                break;
            if (j == i)
            {
                auto sub = word.substr(i, 1);
                if (token_to_id_.find(sub) != token_to_id_.end())
                    tokens.push_back(token_to_id_.at(sub));
                else
                    fprintf(stderr, "%s: unknown token '%s'\n", __func__, sub.data());
                ++i;
            }
        }
    }

    // append suffix.
    tokens.reserve(tokens.size() + encode_suffix.size());
    tokens.insert(tokens.end(), encode_suffix.begin(), encode_suffix.end());

    return tokens;
}

private:
    const std::string m_pat = R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL]|'[dD])|[^\r\n[:alpha:][:digit:]]?[[:alpha:]]+|[[:digit:]]{1,3}| ?[^\s[:alpha:][:digit:]]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    // Prefix: <|begin_of_text|><|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>
    const std::vector<int> encode_prefix = {128000, 128006, 9125, 128007, 2675, 527, 264, 11190, 18328, 128009, 128006, 882, 128007};
    // Suffix: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    const std::vector<int> encode_suffix = {128009, 128006, 78191, 128007};
    std::map<std::string, int32_t> token_to_id_;
    std::map<int32_t, std::string> id_to_token_;
    int m_n_vocab;
};



// int main(int argc, char const *argv[])
// {
//     Llama32Tokenizer tok{"tokenizer.bin", 128000};

//     std::string prompt{"Who are you?"};
//     std::vector<int> toks = tok.encode(prompt);

//     for (int i = 0; i < toks.size(); i++) {
//         std::cout << toks[i] << ", ";
//     }
//     std::cout << "\n";
    
//     return 0;
// }


