#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>

// Include the faster_whisper_cpp headers
#include "transcribe.h"
#include "audio.h"

class WhisperModelCaller {
private:
    std::string modelPath;

    std::string createJsonOutput(const std::vector<Segment>& segments, const TranscriptionInfo& info) {
        std::ostringstream json;
        json << std::fixed << std::setprecision(3);

        json << "{\n";
        json << "  \"success\": true,\n";
        json << "  \"language\": \"" << escapeJsonString(info.language) << "\",\n";
        json << "  \"language_probability\": " << info.language_probability << ",\n";
        json << "  \"duration\": " << info.duration << ",\n";
        json << "  \"segments\": [\n";

        for (size_t i = 0; i < segments.size(); ++i) {
            const auto& segment = segments[i];
            json << "    {\n";
            json << "      \"text\": \"" << escapeJsonString(segment.text) << "\",\n";
            json << "      \"start\": " << segment.start << ",\n";
            json << "      \"end\": " << segment.end << ",\n";
            json << "      \"avg_logprob\": " << segment.avg_logprob << ",\n";
            json << "      \"words\": [\n";

            if (segment.words.has_value()) {
                const auto& words = segment.words.value();
                for (size_t j = 0; j < words.size(); ++j) {
                    const auto& word = words[j];
                    json << "        {\n";
                    json << "          \"word\": \"" << escapeJsonString(word.word) << "\",\n";
                    json << "          \"start\": " << word.start << ",\n";
                    json << "          \"end\": " << word.end << ",\n";
                    json << "          \"probability\": " << word.probability << "\n";
                    json << "        }";
                    if (j < words.size() - 1) json << ",";
                    json << "\n";
                }
            }

            json << "      ]\n";
            json << "    }";
            if (i < segments.size() - 1) json << ",";
            json << "\n";
        }

        json << "  ]\n";
        json << "}\n";

        return json.str();
    }

    std::string createErrorJson(const std::string& error) {
        return "{\"success\": false, \"error\": \"" + escapeJsonString(error) + "\"}";
    }

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

public:
    WhisperModelCaller(const std::string& modelPath) : modelPath(modelPath) {}

    std::string transcribe(const std::string& audioFile) {
        try {
            std::cout << "\nLoading WhisperModel..." << std::endl;

            // Initialize WhisperModel with the same parameters as Python version
            WhisperModel model(
                modelPath,           // model_size_or_path
                "cpu",              // device
                {0},                // device_index
                "int8",             // compute_type
                0,                  // cpu_threads (0 = auto)
                1,                  // num_workers
                "",                 // download_root
                true,               // local_files_only
                {},                 // files
                "",                 // revision
                ""                  // use_auth_token
            );

            std::cout << "Model loaded successfully!\n" << std::endl;

            std::cout << "=== Testing with " << audioFile << " ===" << std::endl;
            std::cout << "Loading audio file: " << audioFile << std::endl;

            // Decode audio file to float samples
            std::vector<float> audio = AudioDecoder::decode_audio(audioFile, 16000);

            if (audio.empty()) {
                return createErrorJson("Failed to decode audio file: " + audioFile);
            }

            // Log audio statistics
            float min_val = *std::min_element(audio.begin(), audio.end());
            float max_val = *std::max_element(audio.begin(), audio.end());
            float sum = std::accumulate(audio.begin(), audio.end(), 0.0f);
            float mean = sum / audio.size();
            float sq_sum = std::inner_product(audio.begin(), audio.end(), audio.begin(), 0.0f);
            float std = std::sqrt(sq_sum / audio.size() - mean * mean);

            std::cout << "Audio loaded: " << audio.size() << " samples ("
                      << std::fixed << std::setprecision(2) << (audio.size() / 16000.0) << " seconds)" << std::endl;
            std::cout << "Audio stats: min=" << std::setprecision(6) << min_val << ", max=" << max_val
                      << ", mean=" << mean << ", std=" << std << std::endl;

            std::cout << "First 20 samples: [";
            for (size_t i = 0; i < std::min(size_t(20), audio.size()); ++i) {
                std::cout << audio[i];
                if (i < 19 && i < audio.size() - 1) std::cout << " ";
            }
            std::cout << "]" << std::endl;

            // Transcribe with Arabic language specified and word timestamps enabled
            auto [segments, info] = model.transcribe(
                audio,              // audio data
                "ar",               // language (force Arabic)
                false               // multilingual
            );

            return createJsonOutput(segments, info);

        } catch (const std::exception& e) {
            return createErrorJson(std::string("Transcription error: ") + e.what());
        }
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "{\"success\": false, \"error\": \"Usage: model_path audio_file\"}" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string audioFile = argv[2];

    try {
        WhisperModelCaller caller(modelPath);
        std::string result = caller.transcribe(audioFile);

        // Print a clear separator for the final result
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "FINAL TRANSCRIPTION RESULT:" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        std::cout << result << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"success\": false, \"error\": \"" << e.what() << "\"}" << std::endl;
        return 1;
    }
}