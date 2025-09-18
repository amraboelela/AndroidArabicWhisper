#ifndef MODEL_DOWNLOADER_H
#define MODEL_DOWNLOADER_H

#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <optional>

// A fictional wrapper for a C++ HTTP client library.
// This class handles the actual file download logic.
class CurlWrapper {
public:
  // Downloads a file from a given URL to a specified output path.
  static void downloadFile(const std::string& url, const std::string& outputPath);
};

// Simulates the behavior of Python's `logging.getLogger`.
void logWarning(const std::string& message);

// Returns a list of all available model names.
std::vector<std::string> availableModels();

// Downloads a CTranslate2 Whisper model from the Hugging Face Hub.
//
// Args:
//   size_or_id: The size of the model (e.g., "small", "base") or a full
//               Hugging Face repository ID (e.g., "Systran/faster-whisper-tiny.en").
//   outputDir: Directory where the model should be saved.
//   localFilesOnly: If true, only load from the local cache.
//   cacheDir: Path to the folder where cached files are stored.
//   revision: An optional Git revision ID.
//   useAuthToken: An optional Hugging Face authentication token.
//
// Returns:
//   The path to the downloaded model.
//
// Throws:
//   std::invalid_argument: if the model size is invalid.
std::string downloadModel(
    const std::string& sizeOrId,
    const std::optional<std::string>& outputDir = std::nullopt,
    bool localFilesOnly = false,
    const std::optional<std::string>& cacheDir = std::nullopt,
    const std::optional<std::string>& revision = std::nullopt,
    const std::optional<std::string>& useAuthToken = std::nullopt
);

// Formats a timestamp in seconds to "HH:MM:SS.mmm" format.
std::string formatTimestamp(
    double seconds,
    bool alwaysIncludeHours = false,
    const std::string& decimalMarker = "."
);

// Gets the 'end' timestamp of the last segment in a vector of segments.
// Returns an empty optional if the vector is empty.
std::optional<double> get_end(const std::vector<std::unordered_map<std::string, double>>& segments);

#endif // MODEL_DOWNLOADER_H

