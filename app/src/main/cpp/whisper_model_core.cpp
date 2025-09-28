/**
 * WhisperModel Core Implementation
 * Contains constructor, basic functionality, and main transcribe entry point
 * Created by Amr Aboelela
 */

#include "whisper_model.h"
#include "utils.h"
#include "tokenizer.h"
#include <ctranslate2/models/whisper.h>
#include <ctranslate2/storage_view.h>
#include <string>
#include <memory>
#include <filesystem>
#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <optional>
#include <cmath>
#include <algorithm>
#include <variant>
#include <stdexcept>
#include <numeric>
#include <cassert>
#include <set>
#include <zlib.h>
#include <cstring>
#include <utility>
#include "audio_decoder.h"
#include "feature_extractor.h"

// Forward declarations and constants

// Logger placeholder
struct Logger {
  void debug(const char* format, ...) const {
    // Simple logging implementation
  }
};
static Logger logger;

namespace fs = std::filesystem;

WhisperModel::WhisperModel(
  const std::string &model_size_or_path,
  const std::string &device,
  const std::vector<int> &device_index,
  const std::string &compute_type,
  int cpu_threads,
  int num_workers,
  const std::string &download_root,
  bool local_files_only,
  const std::map<std::string, std::string> &files,
  const std::string &revision,
  const std::string &use_auth_token
) {
  //std::string model_path;
  //std::string preprocessor_config;

  std::string model_path;
  if (!files.empty()) {
  // If model files are already provided in memory (not implemented here)
  model_path = model_size_or_path;
  } else if (std::filesystem::is_directory(model_size_or_path)) {
  model_path = model_size_or_path;
  } else {
  // In Python: download_model(...)
  // In C++: You must implement downloading manually or assume pre-downloaded
  model_path = model_size_or_path;
  }

  ctranslate2::ReplicaPoolConfig config;
  config.num_threads_per_replica = cpu_threads;   // map your params here

  // Initialize the CTranslate2 model.
  model = std::make_shared<ctranslate2::models::Whisper>(
    model_path,
    ctranslate2::Device::CPU,
    ctranslate2::ComputeType::DEFAULT,
    device_index,
    false,
    config
  );

  // Initialize tokenizer placeholder
  hf_tokenizer = nullptr;

  // -------------------
  // Tokenizer Handling
  // -------------------
  // In Python: tokenizers.Tokenizer.from_file("tokenizer.json")
  // In C++: you must implement or use a tokenizer wrapper
  std::string tokenizer_file = model_path + "/tokenizer.json";
  if (std::filesystem::exists(tokenizer_file)) {
  std::cout << "Load tokenizer from: " << tokenizer_file << std::endl;
  // TODO: integrate Hugging Face tokenizers (Rust) or your own
  } else {
  std::cerr << "Tokenizer not found, defaulting to fallback.\n";
  }

  // Placeholder for feature_kwargs logic.
  // In a real implementation, this would parse preprocessor_config.json.
  // We assume default parameters here as in the Python `FeatureExtractor`.
  feature_extractor = FeatureExtractor();

  input_stride = 2;
  num_samples_per_token = feature_extractor.hop_length * input_stride;
  frames_per_second = feature_extractor.sampling_rate() / feature_extractor.hop_length;
  tokens_per_second = feature_extractor.sampling_rate() / num_samples_per_token;
  time_precision = 0.02;
  max_length = 448;
}

std::vector<std::string> WhisperModel::supported_languages() const {
  if (model->is_multilingual()) {
    return _LANGUAGE_CODES;
  }
  return {"en"};
}

std::map<std::string, std::string> WhisperModel::get_feature_kwargs(
  const std::string &model_path,
  const std::optional<std::string> &preprocessor_bytes
) {
  std::map<std::string, std::string> config;
  try {
  std::string config_path = model_path + "/preprocessor_config.json";
  if (preprocessor_bytes.has_value()) {
    config = parse_json(preprocessor_bytes.value());
  } else if (std::filesystem::exists(config_path)) {
    config = parse_json_file(config_path);
  }

  // Optionally filter keys to match your FeatureExtractor constructor
  return config;
  } catch (const std::exception &e) {
  std::cerr << "Could not load preprocessor config: " << e.what() << std::endl;
  }
  return config;
}

std::tuple<std::vector<Segment>, TranscriptionInfo> WhisperModel::transcribe(
  const std::vector<float> &audio,
  const std::optional<std::string> &language,
  bool multilingual
) {
  if (audio.empty()) {
    throw std::runtime_error("Audio data is empty");
  }

  // Validate multilingual setting with model capability
  if (multilingual && !model->is_multilingual()) {
    multilingual = false;
  }

  // Calculate duration
  float duration = static_cast<float>(audio.size()) / feature_extractor.sampling_rate();

  // Extract features from the full audio (Python line 1082-1083)
  auto features = feature_extractor.extract_features(audio);

  // Language detection if no language is provided (Python line 1085-1103)
  std::string detected_language = "en";
  float language_probability = 1.0f;
  std::vector<std::pair<std::string, float>> all_language_probs;

  if (!language.has_value() && multilingual) {
    std::tie(detected_language, language_probability, all_language_probs) =
        detect_language(&audio, &features);
  } else if (language.has_value()) {
    detected_language = language.value();
  }

  // For English-only models, force language to English
  if (!model->is_multilingual()) {
    detected_language = "en";
  }

  // Initialize tokenizer with the detected/specified language (Python line 1105-1108)
  Tokenizer tokenizer(hf_tokenizer, model->is_multilingual(),
                     detected_language == "en" ? std::nullopt : std::make_optional(detected_language));

  // Set up transcription options with defaults (Python line 1110-1113)
  TranscriptionOptions options;
  options.multilingual = multilingual;

  // Generate segments using the features and tokenizer (Python line 1115)
  auto segments = generate_segments(features, tokenizer, options);

  // Create and return transcription info (Python line 1117-1124)
  TranscriptionInfo info;
  info.language = detected_language;
  info.language_probability = language_probability;
  info.duration = duration;
  if (!all_language_probs.empty()) {
    info.all_language_probs = all_language_probs;
  }

  return std::make_tuple(segments, info);
}

ctranslate2::StorageView WhisperModel::encode(const std::vector<std::vector<float>> &features) {
  if (features.empty() || features[0].empty()) {
    throw std::runtime_error("Features are empty");
  }

  // Convert features to CTranslate2 format
  auto features_storage = get_ctranslate2_storage(features);

  // Encode features using the model
  auto encoder_output = model->encode(features_storage, false); // to_cpu=false

  return encoder_output;
}

std::tuple<std::string, float, std::vector<std::pair<std::string, float>>>
WhisperModel::detect_language(
  const std::vector<float> *audio,
  const std::vector<std::vector<float>> *features,
  int language_detection_segments,
  float language_detection_threshold
) {
  std::string detected_language = "en";
  float language_probability = 1.0f;
  std::vector<std::pair<std::string, float>> all_language_probs;

  // Only proceed if model is multilingual
  if (!model->is_multilingual()) {
    return std::make_tuple(detected_language, language_probability, all_language_probs);
  }

  try {
    std::vector<std::vector<float>> detection_features;

    if (features) {
      detection_features = *features;
    } else if (audio) {
      // Extract features specifically for language detection
      size_t n_samples = 30 * feature_extractor.sampling_rate(); // 30 seconds
      std::vector<float> detection_audio = *audio;

      // Resize audio if needed for language detection
      if (detection_audio.size() > language_detection_segments * n_samples) {
        detection_audio.resize(language_detection_segments * n_samples);
      }

      detection_features = feature_extractor.extract_features(detection_audio);
    } else {
      throw std::runtime_error("Either audio or features must be provided for language detection");
    }

    // Process features for detection (limit to what's needed)
    size_t max_frames = 3000; // 30 seconds worth
    if (detection_features[0].size() > language_detection_segments * max_frames) {
      for (auto& feature_row : detection_features) {
        feature_row.resize(language_detection_segments * max_frames);
      }
    }

    // Encode features for language detection
    auto encoder_output = encode(detection_features);

    // Detect language using the model
    auto results = model->detect_language(encoder_output);
    if (!results.empty()) {
      auto result = results[0].get(); // Get the future result
      if (!result.empty()) {
        std::string language_token = result[0].first;
        // Extract language code (Python line 1181: language = language_token[2:-2])
        if (language_token.length() > 4) {
          detected_language = language_token.substr(2, language_token.length() - 4);
          language_probability = result[0].second;
        }

        // Store all language probabilities
        for (const auto& [token, prob] : result) {
          if (token.length() > 4) {
            std::string lang_code = token.substr(2, token.length() - 4);
            all_language_probs.emplace_back(lang_code, prob);
          }
        }
      }
    }

    // If detection confidence is too low, default to English
    if (language_probability < language_detection_threshold) {
      detected_language = "en";
      language_probability = 1.0f;
    }

  } catch (const std::exception& e) {
    std::cerr << "Language detection failed: " << e.what() << ", defaulting to English" << std::endl;
    detected_language = "en";
    language_probability = 1.0f;
  }

  return std::make_tuple(detected_language, language_probability, all_language_probs);
}