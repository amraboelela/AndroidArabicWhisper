#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <sstream>

class TokenizerCreator {
private:
    std::string escapeJsonString(const std::string& str) {
        std::string escaped;
        for (char c : str) {
            switch (c) {
                case '"': escaped += "\\\""; break;
                case '\\': escaped += "\\\\"; break;
                case '\b': escaped += "\\b"; break;
                case '\f': escaped += "\\f"; break;
                case '\n': escaped += "\\n"; break;
                case '\r': escaped += "\\r"; break;
                case '\t': escaped += "\\t"; break;
                default: escaped += c; break;
            }
        }
        return escaped;
    }

    std::vector<std::string> parseVocabulary(const std::string& vocabPath) {
        std::ifstream file(vocabPath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open vocabulary.json");
        }

        std::string content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
        file.close();

        // Simple JSON array parser for vocabulary
        std::vector<std::string> vocab;
        size_t start = content.find('[');
        size_t end = content.rfind(']');

        if (start == std::string::npos || end == std::string::npos) {
            throw std::runtime_error("Invalid JSON format in vocabulary.json");
        }

        std::string arrayContent = content.substr(start + 1, end - start - 1);

        // Parse tokens (simple parser for string array)
        size_t pos = 0;
        while (pos < arrayContent.length()) {
            // Skip whitespace and commas
            while (pos < arrayContent.length() &&
                   (std::isspace(arrayContent[pos]) || arrayContent[pos] == ',')) {
                pos++;
            }

            if (pos >= arrayContent.length()) break;

            // Find start and end of quoted string
            if (arrayContent[pos] == '"') {
                pos++; // Skip opening quote
                size_t tokenStart = pos;

                // Find closing quote, handling escaped quotes
                while (pos < arrayContent.length()) {
                    if (arrayContent[pos] == '"' &&
                        (pos == 0 || arrayContent[pos-1] != '\\')) {
                        break;
                    }
                    pos++;
                }

                if (pos < arrayContent.length()) {
                    std::string token = arrayContent.substr(tokenStart, pos - tokenStart);
                    vocab.push_back(token);
                    pos++; // Skip closing quote
                }
            } else {
                pos++;
            }
        }

        return vocab;
    }

    void writeTokenizerJson(const std::string& tokenizerPath,
                           const std::vector<std::string>& vocab) {
        std::ofstream file(tokenizerPath);
        if (!file.is_open()) {
            throw std::runtime_error("Could not create tokenizer.json");
        }

        file << "{\n";
        file << "  \"version\": \"1.0\",\n";
        file << "  \"truncation\": null,\n";
        file << "  \"padding\": null,\n";
        file << "  \"added_tokens\": [],\n";
        file << "  \"normalizer\": null,\n";
        file << "  \"pre_tokenizer\": null,\n";
        file << "  \"post_processor\": null,\n";
        file << "  \"decoder\": null,\n";
        file << "  \"model\": {\n";
        file << "    \"type\": \"BPE\",\n";
        file << "    \"dropout\": null,\n";
        file << "    \"unk_token\": null,\n";
        file << "    \"continuing_subword_prefix\": null,\n";
        file << "    \"end_of_word_suffix\": null,\n";
        file << "    \"fuse_unk\": false,\n";
        file << "    \"vocab\": {\n";

        // Write vocabulary entries
        for (size_t i = 0; i < vocab.size(); ++i) {
            file << "      \"" << escapeJsonString(vocab[i]) << "\": " << i;
            if (i < vocab.size() - 1) {
                file << ",";
            }
            file << "\n";
        }

        file << "    },\n";
        file << "    \"merges\": []\n";
        file << "  }\n";
        file << "}\n";

        file.close();
    }

public:
    bool createTokenizerJson(const std::string& modelPath) {
        std::string vocabPath = modelPath + "/vocabulary.json";
        std::string tokenizerPath = modelPath + "/tokenizer.json";

        // Check if tokenizer.json already exists
        if (std::filesystem::exists(tokenizerPath)) {
            std::cout << "tokenizer.json already exists" << std::endl;
            return true;
        }

        // Check if vocabulary.json exists
        if (!std::filesystem::exists(vocabPath)) {
            std::cout << "vocabulary.json not found" << std::endl;
            return false;
        }

        try {
            // Parse vocabulary
            std::vector<std::string> vocab = parseVocabulary(vocabPath);

            // Write tokenizer.json
            writeTokenizerJson(tokenizerPath, vocab);

            std::cout << "Created basic tokenizer.json with " << vocab.size()
                      << " tokens" << std::endl;
            return true;

        } catch (const std::exception& e) {
            std::cout << "Error creating tokenizer.json: " << e.what() << std::endl;
            return false;
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    TokenizerCreator creator;
    bool success = creator.createTokenizerJson(modelPath);

    return success ? 0 : 1;
}