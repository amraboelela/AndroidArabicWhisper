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
#include <chrono>
#include <ctime>

// Include the faster_whisper_cpp headers
#include "transcribe.h"
#include "audio.h"

// Helper function to log with timestamp
std::string getCurrentTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto value = now_ms.time_since_epoch();
    auto duration = value.count();

    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << local_time->tm_hour << ":"
        << std::setfill('0') << std::setw(2) << local_time->tm_min << ":"
        << std::setfill('0') << std::setw(2) << local_time->tm_sec << "."
        << std::setfill('0') << std::setw(3) << (duration % 1000);
    return oss.str();
}

void logWithTimestamp(const std::string& message) {
    std::cout << "[" << getCurrentTimestamp() << "] " << message << std::endl;
}

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
            logWithTimestamp("Loading WhisperModel in offline mode...");

            // Initialize WhisperModel with the same parameters as Python version
            WhisperModel model(
                modelPath,           // model_size_or_path
                "cpu",              // device
                {0},                // device_index
                "int8",             // compute_type
                4,                  // cpu_threads (explicitly set to 4 for best performance)
                1,                  // num_workers
                "",                 // download_root
                true,               // local_files_only
                {},                 // files
                "",                 // revision
                ""                  // use_auth_token
            );

            logWithTimestamp("Model loaded successfully!");

            logWithTimestamp("=== Testing with " + audioFile + " ===");
            logWithTimestamp("Loading audio file: " + audioFile);

            // Decode audio file to float samples
            std::vector<float> audio = Audio::decode_audio(audioFile, 16000);

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

            std::ostringstream audio_loaded_msg;
            audio_loaded_msg << "Audio loaded: " << audio.size() << " samples ("
                      << std::fixed << std::setprecision(2) << (audio.size() / 16000.0) << " seconds)";
            logWithTimestamp(audio_loaded_msg.str());

            std::cout << "Audio stats: min=" << std::setprecision(6) << min_val << ", max=" << max_val
                      << ", mean=" << mean << ", std=" << std << std::endl;

            std::cout << "First 20 samples: [";
            for (size_t i = 0; i < std::min(size_t(20), audio.size()); ++i) {
                // Format like Python: show minimal decimals (0. instead of 0.000000)
                float val = audio[i];
                if (val == 0.0f) {
                    std::cout << "0.";
                } else {
                    std::cout << val;
                }
                if (i < 19 && i < audio.size() - 1) std::cout << " ";
            }
            std::cout << "]" << std::endl;

            // Start timing
            logWithTimestamp("Starting transcription...");
            auto start_time = std::chrono::high_resolution_clock::now();

            // Transcribe with Arabic language specified and word timestamps enabled
            auto [segments, info] = model.transcribe(
                audio,              // audio data
                "ar",               // language (force Arabic)
                false               // multilingual
            );

            // End timing
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            double seconds = duration.count() / 1000.0;
            std::cout << "\n⏱️  Transcription took: " << std::fixed << std::setprecision(2)
                      << seconds << " seconds" << std::endl;

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
