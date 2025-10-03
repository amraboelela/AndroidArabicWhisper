#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <fstream>
#include <sstream>
#include "whisper_transcriber.hpp"

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

class WhisperTester {
private:
    std::string pythonPath;
    std::string modelPath;
    std::unique_ptr<WhisperTranscriber> transcriber;
    TokenizerCreator tokenizerCreator;

public:
    WhisperTester() {
        pythonPath = "python3";
        modelPath = findWhisperModelPath();
        transcriber = std::make_unique<WhisperTranscriber>(modelPath, pythonPath);
    }

    std::string findWhisperModelPath() {
        std::vector<std::string> possiblePaths = {
            "../whisper_ct2",
            "./whisper_ct2",
            "../../whisper_ct2"
        };

        for (const auto& path : possiblePaths) {
            std::string modelFile = path + "/model.bin";
            if (std::filesystem::exists(modelFile)) {
                std::cout << "Found whisper model at: " << path << std::endl;
                return std::filesystem::absolute(path);
            }
        }

        throw std::runtime_error("Could not find whisper_ct2 directory with model.bin");
    }

    bool fileExists(const std::string& filename) {
        return std::filesystem::exists(filename);
    }

    int executeCommand(const std::string& command) {
        std::cout << "Executing: " << command << std::endl;
        int result = std::system(command.c_str());
        return result;
    }

    bool createTokenizerJson() {
        std::cout << "\n=== Creating tokenizer.json if needed ===" << std::endl;

        return tokenizerCreator.createTokenizerJson(modelPath);
    }

    void printTranscriptionResults(const TranscriptionResult& result) {
        if (!result.success) {
            std::cerr << "❌ Transcription failed: " << result.error << std::endl;
            return;
        }

        std::cout << "\n=== Transcription Results ===" << std::endl;
        std::cout << "Language: " << result.language << std::endl;
        std::cout << "Language probability: " << std::fixed << std::setprecision(3)
                  << result.language_probability << std::endl;
        std::cout << "Duration: " << std::fixed << std::setprecision(2)
                  << result.duration << "s" << std::endl;
        std::cout << "Segments: " << result.segments.size() << std::endl;
        std::cout << std::endl;

        for (size_t i = 0; i < result.segments.size(); ++i) {
            const auto& segment = result.segments[i];

            std::cout << "Segment " << i << ":" << std::endl;
            std::cout << "  Text: '" << segment.text << "'" << std::endl;
            std::cout << "  Time: " << std::fixed << std::setprecision(2)
                      << segment.start << "s - " << segment.end << "s" << std::endl;
            std::cout << "  Confidence: " << std::fixed << std::setprecision(3)
                      << segment.avg_logprob << std::endl;

            if (!segment.words.empty()) {
                std::cout << "  Words (" << segment.words.size() << "):" << std::endl;
                for (size_t j = 0; j < std::min(size_t(10), segment.words.size()); ++j) {
                    const auto& word = segment.words[j];
                    std::cout << "    " << (j + 1) << ". '" << word.text
                              << "' (" << std::fixed << std::setprecision(2)
                              << word.start << "s-" << word.end << "s, prob="
                              << std::fixed << std::setprecision(3)
                              << word.probability << ")" << std::endl;
                }
                if (segment.words.size() > 10) {
                    std::cout << "    ... and " << (segment.words.size() - 10)
                              << " more words" << std::endl;
                }
            }
            std::cout << std::endl;
        }

        std::cout << "✅ Transcription completed successfully!" << std::endl;
    }

    bool transcribeAudio(const std::string& audioFile) {
        std::cout << "\n=== Transcribing Audio File ===" << std::endl;
        std::cout << "Audio file: " << audioFile << std::endl;

        if (!fileExists(audioFile)) {
            std::cerr << "❌ Audio file not found: " << audioFile << std::endl;
            return false;
        }

        // Call our native C++ whisper_model_caller directly to see Arabic output
        std::string command = "./whisper_model_caller \"" + modelPath + "\" \"" + audioFile + "\"";
        std::cout << "\n🔧 Calling native C++ whisper_model_caller:" << std::endl;
        std::cout << "Command: " << command << std::endl;
        std::cout << std::string(70, '=') << std::endl;

        int result = std::system(command.c_str());

        std::cout << std::string(70, '=') << std::endl;
        std::cout << "Native C++ whisper_model_caller exit code: " << result << std::endl;

        // Also call the transcriber for compatibility
        TranscriptionResult transcription_result = transcriber->transcribe(audioFile);
        printTranscriptionResults(transcription_result);

        return transcription_result.success || (result == 0);
    }

    bool runTest() {
        std::cout << "🚀 Whisper Audio Transcription Test" << std::endl;
        std::cout << "===================================" << std::endl;

        try {
            // Step 1: Create tokenizer.json if needed
            if (!createTokenizerJson()) {
                std::cerr << "❌ Failed to create tokenizer.json" << std::endl;
                return false;
            }

            // Step 2: Transcribe the audio file
            if (!transcribeAudio("data/001.wav")) {
                std::cerr << "❌ Failed to transcribe audio file" << std::endl;
                return false;
            }

            std::cout << "\n🎉 Test completed successfully!" << std::endl;
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
            std::cout << "  " << argv[0] << "              # Transcribe data/001.wav" << std::endl;
            std::cout << "  " << argv[0] << " --help       # Show this help" << std::endl;
            return 0;
        }
    }

    // Run the transcription test
    return tester.runTest() ? 0 : 1;
}