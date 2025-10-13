#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <fstream>
#include <sstream>

// Include the tokenizer creator functionality directly
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
};

class WhisperTester {
private:
    std::string pythonPath;
    std::string modelPath;
    TokenizerCreator tokenizerCreator;

public:
    WhisperTester() {
        pythonPath = "python3";
        modelPath = findWhisperModelPath();
    }

    std::string findWhisperModelPath() {
        std::vector<std::string> possiblePaths = {
            "../../../app/src/main/assets/whisper_ct2",
            "../../app/src/main/assets/whisper_ct2"
        };

        for (const auto& path : possiblePaths) {
            std::string configFile = path + "/config.json";
            if (std::filesystem::exists(configFile)) {
                return std::filesystem::absolute(path);
            }
        }

        throw std::runtime_error("Could not find whisper_ct2 directory with config.json");
    }

    bool fileExists(const std::string& filename) {
        return std::filesystem::exists(filename);
    }

    int executeCommand(const std::string& command) {
        std::cout << "Executing: " << command << std::endl;
        int result = std::system(command.c_str());
        return result;
    }

    bool transcribeAudio(const std::string& audioFile) {
        if (!fileExists(audioFile)) {
            std::cerr << "❌ Audio file not found: " << audioFile << std::endl;
            return false;
        }

        // Call our native C++ whisper_model_caller directly to see Arabic output
        std::string command = "./whisper_model_caller \"" + modelPath + "\" \"" + audioFile + "\"";

        int result = std::system(command.c_str());

        return (result == 0);
    }

    bool runTest() {
        try {
            // Step 1: Create tokenizer.json if needed (do this first to match Python order)

            // Canonicalize the path to remove ../ components
            std::string cleanPath = std::filesystem::canonical(modelPath).string();
            std::cout << "Testing CTranslate2 model at: " << cleanPath << std::endl;

            // Step 2: Transcribe the audio file
            if (!transcribeAudio("../../../app/src/main/assets/001.wav")) {
                std::cerr << "❌ Failed to transcribe audio file" << std::endl;
                return false;
            }

            return true;

        } catch (const std::exception& e) {
            std::cerr << "❌ Error: " << e.what() << std::endl;
            return false;
        }
    }
};

int main(int argc, char* argv[]) {
    WhisperTester tester;

    // Handle simple command line arguments
    if (argc > 1) {
        std::string command = argv[1];
        if (command == "--help") {
            std::cout << "Usage:" << std::endl;
            std::cout << "  " << argv[0] << "              # Transcribe audio file" << std::endl;
            std::cout << "  " << argv[0] << " --help       # Show this help" << std::endl;
            return 0;
        }
    }

    // Run the transcription test
    return tester.runTest() ? 0 : 1;
}
