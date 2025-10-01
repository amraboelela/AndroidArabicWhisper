#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <fstream>
#include <sstream>

class WhisperModelCaller {
private:
    std::string modelPath;

    std::string executeCommand(const std::string& command) {
        std::string result;
        char buffer[128];

        FILE* pipe = popen(command.c_str(), "r");
        if (!pipe) {
            return "{\"success\": false, \"error\": \"Failed to execute command\"}";
        }

        while (fgets(buffer, sizeof buffer, pipe) != nullptr) {
            result += buffer;
        }

        pclose(pipe);
        return result;
    }

public:
    WhisperModelCaller(const std::string& modelPath) : modelPath(modelPath) {}

    std::string transcribe(const std::string& audioFile) {
        // Fallback: C++ wrapper that calls the Python script for ML inference
        // This is used when CTranslate2 dependencies are not available
        std::string command = "python3 whisper_model_caller.py \"" + modelPath + "\" \"" + audioFile + "\"";
        return executeCommand(command);
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
        std::cout << result << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cout << "{\"success\": false, \"error\": \"" << e.what() << "\"}" << std::endl;
        return 1;
    }
}