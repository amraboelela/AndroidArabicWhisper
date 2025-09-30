#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include "whisper_transcriber.hpp"

class WhisperTester {
private:
    std::string pythonPath;
    std::string modelPath;
    std::unique_ptr<WhisperTranscriber> transcriber;

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

        std::string command = pythonPath + " py_helpers/create_tokenizer.py \"" + modelPath + "\"";
        int result = executeCommand(command);
        return result == 0;
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

        TranscriptionResult result = transcriber->transcribe(audioFile);
        printTranscriptionResults(result);

        return result.success;
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